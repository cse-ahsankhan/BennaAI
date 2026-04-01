from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Callable

from ingest.loader import load_pdf
from ingest.chunker import chunk_pages
from ingest.embedder import embed_passages
from retrieval import vector_store, bm25_index

logger = logging.getLogger(__name__)


def ingest_document(
    file_path: Path,
    project_id: str,
    progress_callback: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    """
    Full ingest pipeline: PDF → pages → chunks → embeddings → ChromaDB + BM25.

    Args:
        file_path:          Path to the PDF file.
        project_id:         Namespace for this project's indexes.
        progress_callback:  Optional callable(message) for progress reporting.

    Returns:
        Summary dict: {file, pages_processed, chunks_created, languages_detected}
    """
    def report(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    file_path = Path(file_path)
    report(f"[1/4] Loading PDF: {file_path.name}")
    pages = load_pdf(file_path)

    if not pages:
        logger.warning("No text extracted from '%s'", file_path.name)
        return {
            "file": file_path.name,
            "pages_processed": 0,
            "chunks_created": 0,
            "languages_detected": [],
        }

    languages_detected = list({p["language"] for p in pages})

    report(f"[2/4] Chunking {len(pages)} pages …")
    chunks = chunk_pages(pages)

    if not chunks:
        logger.warning("No chunks produced from '%s'", file_path.name)
        return {
            "file": file_path.name,
            "pages_processed": len(pages),
            "chunks_created": 0,
            "languages_detected": languages_detected,
        }

    report(f"[3/4] Embedding {len(chunks)} chunks …")
    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(texts)

    report(f"[4/4] Indexing into ChromaDB and BM25 for project '{project_id}' …")
    vector_store.add_documents(chunks, embeddings, project_id)
    bm25_index.append_and_rebuild(chunks, project_id)

    summary = {
        "file": file_path.name,
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
        "languages_detected": languages_detected,
    }
    report(f"Done. {summary}")
    return summary
