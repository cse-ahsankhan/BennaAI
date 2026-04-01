from __future__ import annotations

import logging
from typing import Dict, Any, List, Generator

from ingest.embedder import embed_query
from retrieval.hybrid import hybrid_search
from llm.provider import get_llm

logger = logging.getLogger(__name__)


def _build_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        sources.append(
            {
                "file": meta.get("source_file", "unknown"),
                "page": meta.get("page_num", "?"),
                "clause_ref": meta.get("clause_ref", ""),
                "language": meta.get("language", "unknown"),
                "doc_type": meta.get("doc_type", "general"),
                "rrf_score": round(chunk.get("rrf_score", 0.0), 4),
                "retrieval_sources": chunk.get("retrieval_sources", []),
                "text_snippet": chunk["text"][:200],
            }
        )
    return sources


def query(
    query_text: str,
    project_id: str,
    llm_provider: str | None = None,
    top_k_each: int = 20,
    final_top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full query pipeline: embed → hybrid retrieve → LLM answer.

    Returns:
        {answer: str, sources: List[dict]}
    """
    logger.info("Query [project=%s]: %s", project_id, query_text[:80])

    # 1. Embed query
    query_embedding = embed_query(query_text)

    # 2. Hybrid retrieval
    chunks = hybrid_search(
        query=query_text,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k_each,
        final_top_k=final_top_k,
    )

    if not chunks:
        return {
            "answer": (
                "No relevant documents found for your query. "
                "Please ensure documents have been ingested for this project."
            ),
            "sources": [],
        }

    # 3. LLM answer
    llm = get_llm(llm_provider)
    answer = llm.generate(query_text, chunks)

    return {
        "answer": answer,
        "sources": _build_sources(chunks),
    }


def query_stream(
    query_text: str,
    project_id: str,
    llm_provider: str | None = None,
    top_k_each: int = 20,
    final_top_k: int = 5,
) -> tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    """
    Streaming variant. Returns (token_generator, sources_list).
    Caller iterates the generator to get streamed tokens.
    """
    logger.info("Stream query [project=%s]: %s", project_id, query_text[:80])

    query_embedding = embed_query(query_text)
    chunks = hybrid_search(
        query=query_text,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k_each,
        final_top_k=final_top_k,
    )

    sources = _build_sources(chunks)

    if not chunks:
        def _empty():
            yield (
                "No relevant documents found for your query. "
                "Please ensure documents have been ingested for this project."
            )
        return _empty(), []

    llm = get_llm(llm_provider)
    return llm.generate_stream(query_text, chunks), sources
