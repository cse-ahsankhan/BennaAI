import logging
from typing import List

import numpy as np
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton


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


def embed_passages(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of passage strings.
    Prefixes each with 'passage: ' as required by the e5 model family.
    Returns a 2-D numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    logger.info("Embedding %d passages in batches of %d …", len(texts), batch_size)

    all_embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(prefixed), batch_size), desc="Embedding", unit="batch"):
        batch = prefixed[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embeddings)

    result = np.vstack(all_embeddings)
    logger.info("Produced embeddings of shape %s", result.shape)
    return result


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
