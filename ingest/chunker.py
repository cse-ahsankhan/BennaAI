import logging
import re
import uuid
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Numbered clause at line start: "1.", "1.1", "1.1.1", "12.4.3"
_NUMBERED_CLAUSE = re.compile(
    r"^\s*(\d{1,3}(?:\.\d{1,3}){0,4}\.?)\s+\S",
    re.MULTILINE,
)

# FIDIC / NEC style: "Clause 4", "Sub-Clause 4.1", "Article 12"
_FIDIC_CLAUSE = re.compile(
    r"^\s*((?:Sub-?)?Clause\s+\d+(?:\.\d+)*|Article\s+\d+(?:\.\d+)*)",
    re.MULTILINE | re.IGNORECASE,
)

# RFI boundary markers
_RFI_BOUNDARY = re.compile(
    r"(?:RFI[-\s]?\d+|RFI\s+No\.?\s*\d+|Request\s+for\s+Information)",
    re.IGNORECASE,
)

# Arabic section markers at line start
_ARABIC_MARKER = re.compile(
    r"^\s*(المادة|البند|الفقرة)\s",
    re.MULTILINE,
)


def _is_allcaps_header(line: str) -> bool:
    """True if line is 10-80 chars with >80% uppercase letters."""
    line = line.strip()
    if not 10 <= len(line) <= 80:
        return False
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.8


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

def _detect_doc_type(text: str) -> str:
    text_lower = text.lower()

    if len(_RFI_BOUNDARY.findall(text)) >= 2:
        return "rfi"

    contract_kw = ("fidic", "nec", "agreement", "contractor", "employer",
                   "liquidated damages", "العقد", "المقاول")
    if any(kw in text_lower for kw in contract_kw):
        return "contract"

    spec_kw = ("specification", "material", "astm", "bs ", "saso",
                "compressive strength", "المواصفات", "مواصفة")
    if any(kw in text_lower for kw in spec_kw):
        return "spec"

    return "general"


# ---------------------------------------------------------------------------
# Structure analysis
# ---------------------------------------------------------------------------

def _is_structured(text: str) -> bool:
    """Return True if >5% of lines contain recognisable clause/section markers."""
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return False
    structured = sum(
        1 for line in lines
        if (_NUMBERED_CLAUSE.match(line)
            or _FIDIC_CLAUSE.match(line)
            or _RFI_BOUNDARY.search(line)
            or _ARABIC_MARKER.match(line)
            or _is_allcaps_header(line))
    )
    return (structured / total) > 0.05


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _extract_clause_ref(text: str) -> str:
    """Extract the first recognisable clause reference from text."""
    m = _FIDIC_CLAUSE.search(text)
    if m:
        return m.group(1).strip()
    m = _NUMBERED_CLAUSE.search(text)
    if m:
        return m.group(1).strip()
    m = _ARABIC_MARKER.search(text)
    if m:
        line_end = text.find("\n", m.start())
        return text[m.start(): line_end if line_end != -1 else m.start() + 60].strip()[:60]
    return ""


def _extract_section_header(text: str) -> Optional[str]:
    """Return the first ALL CAPS or Arabic section header found in text."""
    for line in text.splitlines():
        stripped = line.strip()
        if _is_allcaps_header(stripped):
            return stripped
        if _ARABIC_MARKER.match(line):
            return stripped[:60]
    return None


# ---------------------------------------------------------------------------
# Boundary splitting
# ---------------------------------------------------------------------------

def _split_on_boundaries(text: str) -> List[tuple]:
    """
    Split text at clause/section boundaries.
    Returns list of (segment_text, clause_ref, section_header).
    """
    boundaries = set()

    for pattern in (_NUMBERED_CLAUSE, _FIDIC_CLAUSE, _ARABIC_MARKER):
        for m in pattern.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            pre = text[line_start: m.start()]
            if m.start() == line_start or not pre.strip():
                boundaries.add(m.start())

    for m in re.finditer(r"^[^\n]{10,80}$", text, re.MULTILINE):
        if _is_allcaps_header(m.group()):
            boundaries.add(m.start())

    if not boundaries:
        return [(text, _extract_clause_ref(text), _extract_section_header(text))]

    sorted_bounds = sorted(boundaries)
    segments = []

    if sorted_bounds[0] > 0:
        pre = text[: sorted_bounds[0]].strip()
        if pre:
            segments.append((pre, _extract_clause_ref(pre), _extract_section_header(pre)))

    for i, start in enumerate(sorted_bounds):
        end = sorted_bounds[i + 1] if i + 1 < len(sorted_bounds) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append((seg, _extract_clause_ref(seg), _extract_section_header(seg)))

    return segments


def _merge_small_segments(segments: List[tuple], min_words: int = 50) -> List[tuple]:
    """Merge segments under min_words into the following sibling."""
    if not segments:
        return segments
    result = []
    i = 0
    while i < len(segments):
        seg_text, clause, header = segments[i]
        if len(seg_text.split()) < min_words and i + 1 < len(segments):
            next_text, next_clause, next_header = segments[i + 1]
            segments[i + 1] = (
                seg_text + "\n" + next_text,
                clause or next_clause,
                header or next_header,
            )
            i += 1
            continue
        result.append((seg_text, clause, header))
        i += 1
    return result


# ---------------------------------------------------------------------------
# Splitter singleton
# ---------------------------------------------------------------------------

_SPLITTER: Optional[RecursiveCharacterTextSplitter] = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _SPLITTER
    if _SPLITTER is None:
        _SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _SPLITTER


def _finalize_segment(
    seg_text: str,
    clause_ref: str,
    section_header: Optional[str],
    base_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split oversized segment further, emit chunk dicts."""
    max_chars = config.CHUNK_SIZE * 6  # ~1.5x token limit in chars
    sub_segs = _get_splitter().split_text(seg_text) if len(seg_text) > max_chars else [seg_text]
    chunks = []
    for sub in sub_segs:
        sub = sub.strip()
        if len(sub) < 30:
            continue
        chunks.append(
            {
                "text": sub,
                "chunk_id": str(uuid.uuid4()),
                "clause_ref": clause_ref or _extract_clause_ref(sub),
                "section_header": section_header,
                **base_meta,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# RFI splitting
# ---------------------------------------------------------------------------

def _split_rfi(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full_text = "\n".join(p["text"] for p in pages)
    parts = re.split(
        r"(?=(?:RFI[-\s]?\d+|RFI\s+No\.?\s*\d+|Request\s+for\s+Information))",
        full_text,
        flags=re.IGNORECASE,
    )
    first_page = pages[0] if pages else {}
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < 30:
            continue
        chunks.append(
            {
                "text": part,
                "source_file": first_page.get("source_file", ""),
                "page_num": first_page.get("page_num", 1),
                "chunk_id": str(uuid.uuid4()),
                "language": first_page.get("language", "unknown"),
                "doc_type": "rfi",
                "clause_ref": _extract_clause_ref(part),
                "section_header": _extract_section_header(part),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert raw page dicts into structured chunk dicts.

    Each chunk:
        {text, source_file, page_num, chunk_id, language, doc_type,
         clause_ref, section_header}
    """
    if not pages:
        return []

    full_text = "\n".join(p["text"] for p in pages)
    doc_type = _detect_doc_type(full_text)
    logger.info("Detected document type: %s", doc_type)

    if doc_type == "rfi":
        chunks = _split_rfi(pages)
        logger.info("Created %d RFI chunks from %d pages", len(chunks), len(pages))
        return chunks

    chunks: List[Dict[str, Any]] = []
    splitter = _get_splitter()

    for page in pages:
        text = page["text"]
        base_meta = {
            "source_file": page["source_file"],
            "page_num": page["page_num"],
            "language": page["language"],
            "doc_type": doc_type,
        }

        if _is_structured(text):
            raw_segs = _split_on_boundaries(text)
            raw_segs = _merge_small_segments(raw_segs, min_words=50)
            for seg_text, clause_ref, section_header in raw_segs:
                chunks.extend(_finalize_segment(seg_text, clause_ref, section_header, base_meta))
        else:
            for seg in splitter.split_text(text):
                seg = seg.strip()
                if len(seg) < 30:
                    continue
                chunks.append(
                    {
                        "text": seg,
                        "chunk_id": str(uuid.uuid4()),
                        "clause_ref": _extract_clause_ref(seg),
                        "section_header": _extract_section_header(seg),
                        **base_meta,
                    }
                )

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


# Alias for import compatibility
chunk_documents = chunk_pages
