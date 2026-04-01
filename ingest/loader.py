import logging
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

try:
    from langdetect import detect, LangDetectException
    _langdetect_available = True
except ImportError:
    _langdetect_available = False
    logger.warning("langdetect not available — language detection disabled")


def _detect_language(text: str) -> str:
    """Detect language of a text snippet. Returns 'ar', 'en', or 'unknown'."""
    if not _langdetect_available or not text.strip():
        return "unknown"
    try:
        lang = detect(text)
        return lang if lang in ("ar", "en") else lang
    except LangDetectException:
        return "unknown"


def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF using PyMuPDF, one dict per page.

    Returns list of:
        {text, page_num, language, source_file}
    """
    file_path = Path(file_path)
    pages: List[Dict[str, Any]] = []

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", file_path, exc)
        raise

    logger.info("Loading PDF: %s (%d pages)", file_path.name, len(doc))

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")  # type: ignore[attr-defined]

        if not text.strip():
            logger.warning(
                "Page %d of '%s' has no extractable text — possibly scanned.",
                page_index + 1,
                file_path.name,
            )
            continue

        language = _detect_language(text[:500])  # sample first 500 chars

        pages.append(
            {
                "text": text,
                "page_num": page_index + 1,
                "language": language,
                "source_file": file_path.name,
            }
        )

    doc.close()
    logger.info(
        "Loaded %d pages from '%s'", len(pages), file_path.name
    )
    return pages
