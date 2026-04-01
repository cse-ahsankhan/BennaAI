import hashlib
import logging
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton
_cache = None  # diskcache.Cache singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model '%s' (first run may download ~1.1 GB) …",
            config.EMBEDDING_MODEL,
        )
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def _get_cache():
    global _cache
    if _cache is None:
        try:
            import diskcache

            cache_dir = config.CHROMA_PERSIST_DIR.parent / "embed_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            _cache = diskcache.Cache(str(cache_dir))
            logger.info("Embedding cache initialised at %s", cache_dir)
        except ImportError:
            logger.warning("diskcache not installed — embedding cache disabled")
            _cache = {}  # fall back to plain dict (in-memory, non-persistent)
    return _cache


def _cache_key(text: str) -> str:
    payload = f"{config.EMBEDDING_MODEL}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def embed_passages(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of passage strings with embedding cache.
    Prefixes each with 'passage: ' as required by the e5 model family.
    Returns a 2-D numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    cache = _get_cache()
    results: List[Optional[np.ndarray]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    # Phase 1: cache lookup
    for i, text in enumerate(texts):
        key = _cache_key(text)
        cached = cache.get(key) if hasattr(cache, "get") else cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    hit_count = len(texts) - len(uncached_texts)
    hit_rate = 100.0 * hit_count / len(texts) if texts else 0.0
    logger.info(
        "Embedding cache: %d hits / %d total (%.0f%% hit rate)",
        hit_count, len(texts), hit_rate,
    )

    # Phase 2: embed cache misses
    if uncached_texts:
        model = _get_model()
        prefixed = [f"passage: {t}" for t in uncached_texts]
        new_embeddings: List[np.ndarray] = []

        for i in tqdm(range(0, len(prefixed), batch_size), desc="Embedding", unit="batch"):
            batch = prefixed[i: i + batch_size]
            embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            new_embeddings.append(embs)

        stacked = np.vstack(new_embeddings)

        # Phase 3: store in cache + place in results
        for idx, text, emb in zip(uncached_indices, uncached_texts, stacked):
            key = _cache_key(text)
            try:
                cache[key] = emb
            except Exception:
                pass  # cache write failure is non-fatal
            results[idx] = emb

    return np.vstack(results)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    Prefixes with 'query: ' as required by the e5 model family.
    Returns a 1-D numpy array.
    """
    model = _get_model()
    prefixed = f"query: {query}"
    embedding = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return embedding  # type: ignore[return-value]
