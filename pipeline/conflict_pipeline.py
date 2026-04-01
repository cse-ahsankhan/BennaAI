"""
Benna AI — Conflict Detection Pipeline

Retrieves relevant chunks independently from two document sets,
then asks the LLM to compare them for contradictions, alignments, or gaps.

This is an additive feature — it does NOT touch query_pipeline.py.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional

import config
from ingest.embedder import embed_query
from retrieval.hybrid import hybrid_search
from pipeline.query_pipeline import _rewrite_query  # shared rewrite logic, read-only
from llm.provider import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CONFLICT_SYSTEM = """\
You are Benna AI, a construction contracts intelligence assistant specializing \
in GCC projects. Your task is to compare two document excerpts and identify \
whether they contradict, align, or have gaps on the topic queried.

Structure your response exactly as:

VERDICT: [CONTRADICTION / ALIGNED / GAP / UNCLEAR]

DOCUMENT A SAYS:
[1-3 sentence summary of what Document A says on this topic, with clause reference]

DOCUMENT B SAYS:
[1-3 sentence summary of what Document B says on this topic, with clause reference]

ANALYSIS:
[2-4 sentences explaining the relationship — what contradicts, what aligns, \
what is missing from one but present in the other. Be specific about numbers, \
dates, or values where they differ.]

RECOMMENDATION:
[1-2 sentences on what the project team should clarify or flag for the \
contracts engineer.]

If the documents do not contain enough information to compare, say so clearly \
rather than speculating.\
"""


def _format_chunks(chunks: List[Dict[str, Any]]) -> tuple:
    """
    Returns (formatted_context: str, label: str).
    label = "filename (doc_type)" from the first chunk.
    """
    if not chunks:
        return "", "Unknown"

    first_meta = chunks[0].get("metadata", {})
    label = (
        f"{first_meta.get('source_file', 'Unknown')} "
        f"({first_meta.get('doc_type', 'general')})"
    )

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        clause = meta.get("clause_ref", "")
        clause_str = f" | Clause {clause}" if clause else ""
        parts.append(
            f"[{i}] Page {meta.get('page_num', '?')}{clause_str}\n{chunk['text']}"
        )

    return "\n\n".join(parts), label


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(analysis: str) -> str:
    """Extract structured VERDICT tag, with keyword-scan fallback."""
    match = re.search(
        r"VERDICT:\s*(CONTRADICTION|ALIGNED|GAP|UNCLEAR)",
        analysis,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()

    text = analysis.lower()
    if any(w in text for w in ("contradict", "conflict", "inconsistent", "disagree")):
        return "contradiction"
    if any(w in text for w in ("aligned", "consistent", "agree", "same")):
        return "aligned"
    if any(w in text for w in ("gap", "missing", "not addressed", "silent on")):
        return "gap"
    return "unclear"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_conflicts(
    query: str,
    project_id: str,
    doc_a_filter: Dict[str, str],
    doc_b_filter: Dict[str, str],
    llm_provider: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Compare two document sets on a given topic.

    Args:
        query:          The topic / question to compare across documents.
        project_id:     Benna AI project namespace.
        doc_a_filter:   ChromaDB-compatible filter for Document A,
                        e.g. {"doc_type": "contract"} or {"source_file": "contract.pdf"}
        doc_b_filter:   Same for Document B.
        llm_provider:   "claude" | "ollama" | None (uses env default).
        top_k:          Chunks to retrieve from each side.

    Returns:
        {
            status:         "ok" | "insufficient_data",
            verdict:        "contradiction" | "aligned" | "gap" | "unclear",
            analysis:       <full LLM response>,
            chunks_a:       List of chunk dicts,
            chunks_b:       List of chunk dicts,
            rewritten_query: query after optional rewriting,
            message:        set only when status == "insufficient_data",
        }
    """
    provider = (llm_provider or config.LLM_PROVIDER).lower()
    logger.info(
        "Conflict detection [project=%s provider=%s]: %s", project_id, provider, query[:80]
    )

    # 1. Optional query rewriting (Claude only)
    rewritten = _rewrite_query(query, provider)
    if rewritten != query:
        logger.info("Conflict query rewritten: %r → %r", query[:60], rewritten[:60])

    # 2. Embed query
    query_embedding = embed_query(rewritten)

    # 3. Retrieve independently from each side
    chunks_a = hybrid_search(
        query=rewritten,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k,
        final_top_k=top_k,
        filters=doc_a_filter,
    )
    chunks_b = hybrid_search(
        query=rewritten,
        query_embedding=query_embedding,
        project_id=project_id,
        top_k_each=top_k,
        final_top_k=top_k,
        filters=doc_b_filter,
    )

    # 4. Guard: need content from both sides
    if not chunks_a or not chunks_b:
        missing = []
        if not chunks_a:
            missing.append(f"Document A ({doc_a_filter})")
        if not chunks_b:
            missing.append(f"Document B ({doc_b_filter})")
        return {
            "status": "insufficient_data",
            "message": (
                f"No relevant content found for: {', '.join(missing)}. "
                "Ensure the documents have been ingested and the filters match."
            ),
            "chunks_a": chunks_a,
            "chunks_b": chunks_b,
            "verdict": "unclear",
            "analysis": "",
            "rewritten_query": rewritten,
        }

    # 5. Format context for both sides
    doc_a_context, doc_a_label = _format_chunks(chunks_a)
    doc_b_context, doc_b_label = _format_chunks(chunks_b)

    # 6. Build user prompt
    user_prompt = (
        f"Query: {rewritten}\n\n"
        f"--- DOCUMENT A ({doc_a_label}) ---\n{doc_a_context}\n\n"
        f"--- DOCUMENT B ({doc_b_label}) ---\n{doc_b_context}"
    )

    # 7. LLM call (non-streaming — structured response needs full text)
    llm = get_llm(provider)

    # Both providers expose generate(); use it directly for structured output
    # We construct a one-shot call by temporarily wrapping as a single chunk
    # that contains our pre-built prompt.
    class _PassthroughChunk:
        """Minimal duck-type so _format_context in provider sees a valid chunk."""
        def __init__(self, text: str) -> None:
            self.data = {"chunk_id": "conflict", "text": text, "metadata": {}}

    # Call the provider's underlying client directly to keep system prompt control
    analysis = _call_llm_direct(provider, _CONFLICT_SYSTEM, user_prompt)

    # 8. Parse verdict
    verdict = _parse_verdict(analysis)
    logger.info("Conflict verdict: %s", verdict)

    return {
        "status": "ok",
        "verdict": verdict,
        "analysis": analysis,
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "rewritten_query": rewritten,
    }


def _call_llm_direct(provider: str, system: str, user: str) -> str:
    """
    Send a direct system+user prompt to the active LLM provider.
    Bypasses the chunk-formatting layer in provider.py so we can inject
    our own structured prompt verbatim.
    """
    if provider == "claude":
        import anthropic

        if not config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    elif provider == "ollama":
        from langchain_community.llms import Ollama

        llm = Ollama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            num_ctx=4096,
            num_predict=1024,
        )
        prompt = f"<<SYS>>\n{system}\n<</SYS>>\n\n{user}\n\nResponse:"
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if "connection" in str(exc).lower() or "refused" in str(exc).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                    "Is Ollama running?"
                ) from exc
            raise

    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")
