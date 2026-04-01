import logging
from typing import List, Dict, Any, Tuple

import numpy as np

from retrieval import vector_store, bm25_index

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (rank + RRF_K)


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    project_id: str,
    top_k_each: int = 20,
    final_top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve from both semantic (ChromaDB) and BM25 indexes,
    fuse results with Reciprocal Rank Fusion, return final_top_k chunks.

    Each returned chunk includes 'rrf_score', 'sources' (list of 'semantic'/'bm25').
    """
    # --- Semantic retrieval ---
    semantic_hits = vector_store.similarity_search(query_embedding, project_id, top_k=top_k_each)
    sem_ids = [h["chunk_id"] for h in semantic_hits]
    logger.debug("Semantic hits: %s", sem_ids[:5])

    # --- BM25 retrieval ---
    bm25_hits: List[Dict[str, Any]] = []
    if bm25_index.index_exists(project_id):
        bm25_hits = bm25_index.search(query, project_id, top_k=top_k_each)
    else:
        logger.warning("No BM25 index for project '%s' — using semantic only", project_id)

    bm25_ids = [h["chunk_id"] for h in bm25_hits]
    logger.debug("BM25 hits: %s", bm25_ids[:5])

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
            # BM25 hit has slightly different shape — normalise
            chunk_lookup[cid] = {
                "chunk_id": cid,
                "text": hit["text"],
                "metadata": {
                    "source_file": hit.get("source_file", ""),
                    "page_num": hit.get("page_num", 0),
                    "language": hit.get("language", "unknown"),
                    "doc_type": hit.get("doc_type", "general"),
                    "clause_ref": hit.get("clause_ref", ""),
                },
            }
        chunk_sources.setdefault(cid, []).append("bm25")

    # Sort by RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in ranked[:final_top_k]:
        chunk = dict(chunk_lookup[cid])
        chunk["rrf_score"] = score
        chunk["retrieval_sources"] = chunk_sources[cid]
        _log_source(cid, chunk_sources[cid], score)
        results.append(chunk)

    return results


def _log_source(chunk_id: str, sources: List[str], score: float) -> None:
    origin = " + ".join(sorted(set(sources)))
    logger.debug("Chunk %s | origin: %-20s | RRF score: %.4f", chunk_id[:8], origin, score)
