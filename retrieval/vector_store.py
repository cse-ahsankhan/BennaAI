from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings

import config

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised at %s", config.CHROMA_PERSIST_DIR)
    return _client


def _collection_name(project_id: str) -> str:
    return f"benna_{project_id}"


def _get_or_create_collection(project_id: str):
    client = _get_client()
    name = _collection_name(project_id)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_documents(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    project_id: str,
) -> None:
    """
    Persist chunks and their embeddings to the ChromaDB collection for project_id.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )

    collection = _get_or_create_collection(project_id)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_file": c["source_file"],
            "page_num": int(c["page_num"]),
            "language": c["language"],
            "doc_type": c["doc_type"],
            "clause_ref": c.get("clause_ref", ""),
        }
        for c in chunks
    ]
    embedding_list = embeddings.tolist()

    # ChromaDB upsert to avoid duplicates on re-ingestion
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embedding_list,
        metadatas=metadatas,
    )
    logger.info(
        "Upserted %d documents into collection '%s'",
        len(chunks),
        _collection_name(project_id),
    )


def similarity_search(
    query_embedding: np.ndarray,
    project_id: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Return top_k most similar chunks for query_embedding.

    Each result: {chunk_id, text, metadata, distance}
    """
    collection = _get_or_create_collection(project_id)
    count = collection.count()
    if count == 0:
        logger.warning("Collection '%s' is empty", _collection_name(project_id))
        return []

    actual_k = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append(
            {
                "chunk_id": doc_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def list_projects() -> List[str]:
    """Return list of project IDs that have been indexed."""
    client = _get_client()
    collections = client.list_collections()
    prefix = "benna_"
    return [c.name[len(prefix):] for c in collections if c.name.startswith(prefix)]


def document_count(project_id: str) -> int:
    """Return the number of indexed chunks for project_id."""
    collection = _get_or_create_collection(project_id)
    return collection.count()
