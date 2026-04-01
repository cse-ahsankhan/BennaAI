from __future__ import annotations

import logging
from typing import Dict, Any, List, Generator, Optional

import config
from ingest.embedder import embed_query
from retrieval.hybrid import hybrid_search
from llm.provider import get_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query rewriting (Claude only — too slow as a preprocessing step on Ollama)
# ---------------------------------------------------------------------------

def _rewrite_query(query: str, provider: str) -> str:
    """
    Use Claude to expand abbreviations and sharpen the query for retrieval.
    Returns the original query unchanged for non-Claude providers.
    """
    if provider.lower() != "claude":
        return query
    if not config.ANTHROPIC_API_KEY:
        return query
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            system=(
                "You are a construction document search assistant. "
                "Rewrite the user query to maximize retrieval of relevant clauses. "
                "Expand abbreviations (e.g. LD → liquidated damages, "
                "PC → provisional cost, BOQ → bill of quantities, "
                "RFI → request for information, CDP → contractor design portion). "
                "If the query is in Arabic, keep it in Arabic. "
                "Return only the rewritten query, nothing else."
            ),
            messages=[{"role": "user", "content": query}],
        )
        rewritten = response.content[0].text.strip()
        if rewritten and rewritten != query:
            logger.info("Query rewritten: %r → %r", query[:60], rewritten[:60])
        return rewritten or query
    except Exception as exc:
        logger.warning("Query rewriting failed (%s) — using original query", exc)
        return query


# ---------------------------------------------------------------------------
# Source formatting
# ---------------------------------------------------------------------------

def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append(
            {
                "file": meta.get("source_file", "unknown"),
                "page": meta.get("page_num", "?"),
                "clause_ref": meta.get("clause_ref", "") or "",
                "section_header": meta.get("section_header") or "",
                "language": meta.get("language", "unknown"),
                "doc_type": meta.get("doc_type", "general"),
                "rrf_score": round(chunk.get("rrf_score", 0.0), 4),
                "retrieval_sources": chunk.get("retrieval_sources", []),
                "text_snippet": chunk["text"][:300],
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    query_text: str,
    project_id: str,
    llm_provider: str | None = None,
    top_k_each: int = 10,
    final_top_k: int = 3,
    filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Full query pipeline: (rewrite →) embed → hybrid retrieve → LLM answer.

    Returns:
        {answer, sources, original_query, rewritten_query}
    """
    provider = (llm_provider or config.LLM_PROVIDER).lower()
    logger.info("Query [project=%s provider=%s]: %s", project_id, provider, query_text[:80])

    rewritten = _rewrite_query(query_text, provider)
    query_embedding = embed_query(rewritten)

    chunks = hybrid_search(
        query=rewritten,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k_each,
        final_top_k=final_top_k,
        filters=filters,
    )

    if not chunks:
        return {
            "answer": (
                "No relevant documents found for your query. "
                "Please ensure documents have been ingested for this project."
            ),
            "sources": [],
            "original_query": query_text,
            "rewritten_query": rewritten,
        }

    llm = get_llm(provider)
    answer = llm.generate(rewritten, chunks)

    return {
        "answer": answer,
        "sources": _build_sources(chunks),
        "original_query": query_text,
        "rewritten_query": rewritten,
    }


def query_stream(
    query_text: str,
    project_id: str,
    llm_provider: str | None = None,
    top_k_each: int = 10,
    final_top_k: int = 3,
    filters: Optional[Dict[str, str]] = None,
) -> tuple[Generator[str, None, None], List[Dict[str, Any]], str]:
    """
    Streaming variant.
    Returns (token_generator, sources_list, rewritten_query).
    """
    provider = (llm_provider or config.LLM_PROVIDER).lower()
    logger.info("Stream query [project=%s]: %s", project_id, query_text[:80])

    rewritten = _rewrite_query(query_text, provider)
    query_embedding = embed_query(rewritten)

    chunks = hybrid_search(
        query=rewritten,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k_each,
        final_top_k=final_top_k,
        filters=filters,
    )

    sources = _build_sources(chunks)

    if not chunks:
        def _empty():
            yield (
                "No relevant documents found for your query. "
                "Please ensure documents have been ingested for this project."
            )
        return _empty(), [], rewritten

    llm = get_llm(provider)
    return llm.generate_stream(rewritten, chunks), sources, rewritten
