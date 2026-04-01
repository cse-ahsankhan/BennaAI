from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

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
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _build_where(filters: Optional[Dict[str, str]]) -> Optional[Dict]:
    """Convert a simple {key: value} filter dict into a ChromaDB where clause."""
    if not filters:
        return None
    items = list(filters.items())
    if len(items) == 1:
        k, v = items[0]
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in items]}


def add_documents(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    project_id: str,
) -> None:
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
            "clause_ref": c.get("clause_ref") or "",
            "section_header": c.get("section_header") or "",
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
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
    top_k: int = 10,
    filters: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return top_k most similar chunks for query_embedding.
    Optional filters: {doc_type: "contract"} | {language: "ar"} etc.
    """
    collection = _get_or_create_collection(project_id)
    count = collection.count()
    if count == 0:
        logger.warning("Collection '%s' is empty", _collection_name(project_id))
        return []

    actual_k = min(top_k, count)
    where = _build_where(filters)

    query_kwargs: Dict[str, Any] = dict(
        query_embeddings=[query_embedding.tolist()],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

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
    client = _get_client()
    collections = client.list_collections()
    prefix = "benna_"
    return [c.name[len(prefix):] for c in collections if c.name.startswith(prefix)]


def document_count(project_id: str) -> int:
    collection = _get_or_create_collection(project_id)
    return collection.count()


def get_indexed_files(project_id: str) -> List[str]:
    """Return sorted list of distinct source_file values in the project collection."""
    try:
        collection = _get_or_create_collection(project_id)
        results = collection.get(include=["metadatas"])
        files = {
            m.get("source_file", "")
            for m in results["metadatas"]
            if m.get("source_file")
        }
        return sorted(files)
    except Exception:
        return []


class VectorStore:
    """Thin class wrapper around module-level functions for API compatibility."""

    def get_indexed_files(self, project_id: str) -> List[str]:
        return get_indexed_files(project_id)

    def document_count(self, project_id: str) -> int:
        return document_count(project_id)

    def list_projects(self) -> List[str]:
        return list_projects()
