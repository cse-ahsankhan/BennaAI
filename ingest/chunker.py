import logging
import re
import uuid
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

# Patterns for structure detection
_CLAUSE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+\S", re.MULTILINE)
_SECTION_HEADER_PATTERN = re.compile(r"^\s*([A-Z][A-Z\s]{3,})\s*$", re.MULTILINE)
_RFI_PATTERN = re.compile(r"\bRFI[\s\-#]*\d+\b", re.IGNORECASE)


def _detect_doc_type(text: str) -> str:
    """Heuristic document type detection."""
    text_lower = text.lower()
    rfi_hits = len(_RFI_PATTERN.findall(text))
    if rfi_hits >= 2:
        return "rfi"
    if any(kw in text_lower for kw in ("contract", "agreement", "parties", "whereas", "عقد")):
        return "contract"
    if any(kw in text_lower for kw in ("specification", "material", "standard", "مواصفة", "مواصفات")):
        return "spec"
    return "general"


def _extract_clause_ref(text: str) -> str:
    """Return the first clause reference found (e.g. '3.2.1'), or empty string."""
    match = _CLAUSE_PATTERN.search(text)
    return match.group(1) if match else ""


def _split_by_clauses(text: str) -> List[str]:
    """
    Split text at clause boundaries (e.g. '1.1 ', '2.3.4 ').
    Returns a list of clause-level segments.
    """
    positions = [m.start() for m in _CLAUSE_PATTERN.finditer(text)]
    if not positions:
        return [text]

    segments = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segments.append(text[start:end].strip())
    # Include any text before the first clause
    if positions[0] > 0:
        segments.insert(0, text[: positions[0]].strip())
    return [s for s in segments if s]


def _split_rfi(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Treat each RFI occurrence in the document as its own chunk."""
    chunks = []
    full_text = "\n".join(p["text"] for p in pages)
    # Split on RFI boundaries
    rfi_splits = re.split(r"(?=\bRFI[\s\-#]*\d+\b)", full_text, flags=re.IGNORECASE)
    first_page = pages[0] if pages else {}

    for segment in rfi_splits:
        segment = segment.strip()
        if not segment:
            continue
        chunks.append(
            {
                "text": segment,
                "source_file": first_page.get("source_file", ""),
                "page_num": first_page.get("page_num", 1),
                "chunk_id": str(uuid.uuid4()),
                "language": first_page.get("language", "unknown"),
                "doc_type": "rfi",
                "clause_ref": _extract_clause_ref(segment),
            }
        )
    return chunks


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw page dicts into structured chunk dicts.

    Each chunk:
        {text, source_file, page_num, chunk_id, language, doc_type, clause_ref}
    """
    if not pages:
        return []

    full_text = "\n".join(p["text"] for p in pages)
    doc_type = _detect_doc_type(full_text)
    logger.info("Detected document type: %s", doc_type)

    chunks: List[Dict[str, Any]] = []

    if doc_type == "rfi":
        return _split_rfi(pages)

    # Build a flat list of segments across all pages
    for page in pages:
        text = page["text"]
        base_meta = {
            "source_file": page["source_file"],
            "page_num": page["page_num"],
            "language": page["language"],
            "doc_type": doc_type,
        }

        if doc_type in ("contract", "spec") and _CLAUSE_PATTERN.search(text):
            segments = _split_by_clauses(text)
        else:
            # Fallback: RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            segments = splitter.split_text(text)

        for segment in segments:
            segment = segment.strip()
            if len(segment) < 20:  # skip noise
                continue
            # If a segment is still too large, further split it
            if len(segment) > config.CHUNK_SIZE * 2:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                )
                sub_segs = splitter.split_text(segment)
            else:
                sub_segs = [segment]

            for sub in sub_segs:
                sub = sub.strip()
                if len(sub) < 20:
                    continue
                chunks.append(
                    {
                        "text": sub,
                        "chunk_id": str(uuid.uuid4()),
                        "clause_ref": _extract_clause_ref(sub),
                        **base_meta,
                    }
                )

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks
