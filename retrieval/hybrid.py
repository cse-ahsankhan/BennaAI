import logging
from typing import List, Dict, Any, Optional

import numpy as np

from retrieval import vector_store, bm25_index

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant — do not change


def _rrf_score(rank: int) -> float:
    return 1.0 / (rank + RRF_K)


def _matches_filters(chunk: Dict[str, Any], filters: Dict[str, str]) -> bool:
    """Check if a BM25 chunk dict matches all filter key/value pairs."""
    for key, val in filters.items():
        if chunk.get(key, "") != val:
            return False
    return True


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    project_id: str,
    top_k_each: int = 10,
    final_top_k: int = 3,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve from both semantic (ChromaDB) and BM25 indexes,
    fuse with Reciprocal Rank Fusion, return final_top_k chunks.

    filters: optional dict like {"doc_type": "contract"} or {"language": "ar"}
    Each returned chunk includes 'rrf_score' and 'retrieval_sources'.
    """
    # --- Semantic retrieval (with native ChromaDB filtering) ---
    semantic_hits = vector_store.similarity_search(
        query_embedding, project_id, top_k=top_k_each, filters=filters
    )
    logger.debug("Semantic hits: %d", len(semantic_hits))

    # --- BM25 retrieval (post-filter) ---
    bm25_hits: List[Dict[str, Any]] = []
    if bm25_index.index_exists(project_id):
        raw_bm25 = bm25_index.search(query, project_id, top_k=top_k_each)
        if filters:
            bm25_hits = [h for h in raw_bm25 if _matches_filters(h, filters)]
        else:
            bm25_hits = raw_bm25
    else:
        logger.warning("No BM25 index for project '%s' — using semantic only", project_id)

    logger.debug("BM25 hits after filter: %d", len(bm25_hits))

    # --- RRF fusion ---
    rrf_scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Dict[str, Any]] = {}
    chunk_sources: Dict[str, List[str]] = {}

    for rank, hit in enumerate(semantic_hits):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank)
        chunk_lookup[cid] = hit
        chunk_sources.setdefault(cid, []).append("semantic")

    for rank, hit in enumerate(bm25_hits):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank)
        if cid not in chunk_lookup:
            chunk_lookup[cid] = {
                "chunk_id": cid,
                "text": hit["text"],
                "metadata": {
                    "source_file": hit.get("source_file", ""),
                    "page_num": hit.get("page_num", 0),
                    "language": hit.get("language", "unknown"),
                    "doc_type": hit.get("doc_type", "general"),
                    "clause_ref": hit.get("clause_ref", ""),
                    "section_header": hit.get("section_header"),
                },
            }
        chunk_sources.setdefault(cid, []).append("bm25")

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in ranked[:final_top_k]:
        chunk = dict(chunk_lookup[cid])
        chunk["rrf_score"] = score
        chunk["retrieval_sources"] = chunk_sources[cid]
        origin = " + ".join(sorted(set(chunk_sources[cid])))
        logger.debug("Chunk %s | origin: %-20s | RRF: %.4f", cid[:8], origin, score)
        results.append(chunk)

    return results
