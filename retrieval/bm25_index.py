import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)

# In-memory cache: project_id -> (bm25, corpus_chunks)
_index_cache: Dict[str, tuple] = {}


def _tokenize(text: str) -> List[str]:
    """
    Arabic-aware tokenizer: splits on whitespace and punctuation, lowercases.
    Preserves Arabic Unicode characters.
    """
    text = text.lower()
    # Split on whitespace and common punctuation (works for both Arabic and Latin)
    tokens = re.split(r"[\s\u060C\u061B\u061F\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]+", text)
    return [t for t in tokens if t]


def _index_path(project_id: str) -> Path:
    return config.BM25_INDEX_DIR / f"{project_id}.pkl"


def build(chunks: List[Dict[str, Any]], project_id: str) -> None:
    """Build a BM25 index from chunks and persist it to disk."""
    corpus = [c["text"] for c in chunks]
    tokenized = [_tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized)

    payload = {"bm25": bm25, "chunks": chunks}
    path = _index_path(project_id)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    _index_cache[project_id] = (bm25, chunks)
    logger.info("Built BM25 index for project '%s' with %d chunks", project_id, len(chunks))


def _load(project_id: str) -> tuple:
    """Load BM25 index from disk into cache."""
    path = _index_path(project_id)
    if not path.exists():
        raise FileNotFoundError(f"No BM25 index found for project '{project_id}' at {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]
    chunks = payload["chunks"]
    _index_cache[project_id] = (bm25, chunks)
    logger.info("Loaded BM25 index for project '%s' (%d chunks)", project_id, len(chunks))
    return bm25, chunks


def search(query: str, project_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Search BM25 index for query. Returns top_k chunk dicts with added 'bm25_score'.
    """
    if project_id not in _index_cache:
        bm25, chunks = _load(project_id)
    else:
        bm25, chunks = _index_cache[project_id]

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score, sort descending
    scored = sorted(
        zip(scores, chunks), key=lambda x: x[0], reverse=True
    )

    results = []
    for score, chunk in scored[:top_k]:
        result = dict(chunk)
        result["bm25_score"] = float(score)
        results.append(result)

    return results


def index_exists(project_id: str) -> bool:
    return _index_path(project_id).exists()


def append_and_rebuild(new_chunks: List[Dict[str, Any]], project_id: str) -> None:
    """
    If an index already exists for this project, load existing chunks,
    merge with new_chunks, then rebuild.
    """
    existing_chunks: List[Dict[str, Any]] = []
    if index_exists(project_id):
        try:
            _, existing_chunks = _load(project_id)
        except Exception as exc:
            logger.warning("Could not load existing BM25 index: %s", exc)

    # Deduplicate by chunk_id
    existing_ids = {c["chunk_id"] for c in existing_chunks}
    merged = existing_chunks + [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    build(merged, project_id)
