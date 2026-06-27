"""services/pdf_ingest.py - PDF document ingestor for the fact-checker pipeline.

Converts PDF files into a list of TranscriptSegment objects that can flow
through the standard claim-extraction and fact-checking pipeline.

Ingest strategy (tried in order):
  1. PyMuPDF (fitz)  - fastest, best layout preservation, handles complex
                       multi-column layouts and embedded text layers.
  2. pdfminer.six    - pure-Python fallback; solid for text-heavy PDFs.
  3. pypdf           - second pure-Python fallback; broad format support.
  4. Raw byte decode  - last resort; extracts readable ASCII/UTF-8 runs.

Each PDF page is extracted as a separate chunk; pages longer than
MAX_CHARS_PER_SEGMENT are further split at sentence boundaries.

Dependencies (all optional - at least one should be installed)::

    pip install pymupdf        # PyMuPDF  (recommended)
    pip install pdfminer.six   # fallback
    pip install pypdf          # second fallback
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple
from uuid import UUID

from ..models import IngestSource, TranscriptSegment
from .file_router import _text_to_segments

log = logging.getLogger(__name__)

# Maximum characters per emitted segment before hard-wrapping
MAX_CHARS_PER_SEGMENT = 1500

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    _PDFMINER_AVAILABLE = True
except ImportError:
    _PDFMINER_AVAILABLE = False

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_pdf(
    job_id: UUID,
    pdf_path: Path,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Extract text from a PDF file and return as TranscriptSegments.

    Tries available PDF libraries in preference order.  The returned
    segments use ``start_sec=0.0`` / ``end_sec=0.0`` since PDFs have no
    audio timeline.  Each segment is prefixed with its page number
    (e.g. ``[Page 3] ...``) for downstream provenance tracking.

    Args:
        job_id:   UUID of the owning pipeline job.
        pdf_path: Path to the PDF file to ingest.

    Returns:
        Tuple of ``(List[TranscriptSegment], IngestSource.DOCUMENT)``.

    Raises:
        RuntimeError: If every extraction backend fails.
    """
    pdf_path = Path(pdf_path)
    log.info("[pdf_ingest] Ingesting: %s", pdf_path.name)

    pages: List[str] = []

    if _FITZ_AVAILABLE:
        pages = _extract_fitz(pdf_path)
        log.info("[pdf_ingest] PyMuPDF: %d pages from %s", len(pages), pdf_path.name)
    elif _PDFMINER_AVAILABLE:
        pages = _extract_pdfminer(pdf_path)
        log.info("[pdf_ingest] pdfminer: %d pages from %s", len(pages), pdf_path.name)
    elif _PYPDF_AVAILABLE:
        pages = _extract_pypdf(pdf_path)
        log.info("[pdf_ingest] pypdf: %d pages from %s", len(pages), pdf_path.name)
    else:
        log.warning("[pdf_ingest] No PDF library available - raw byte decode.")
        pages = _extract_raw(pdf_path)

    if not pages:
        log.warning("[pdf_ingest] No text extracted from %s", pdf_path.name)
        return [], IngestSource.DOCUMENT

    segments: List[TranscriptSegment] = []
    for page_num, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        labelled = f"[Page {page_num}] {page_text}"
        page_segs = _text_to_segments(
            job_id=job_id,
            text=labelled,
            source_name=f"{pdf_path.name}:p{page_num}",
            max_chars=MAX_CHARS_PER_SEGMENT,
        )
        segments.extend(page_segs)

    log.info(
        "[pdf_ingest] %s -> %d segments from %d pages",
        pdf_path.name, len(segments), len(pages),
    )
    return segments, IngestSource.DOCUMENT


# ---------------------------------------------------------------------------
# Extraction backends
# ---------------------------------------------------------------------------

def _extract_fitz(pdf_path: Path) -> List[str]:
    """Extract per-page text using PyMuPDF (fitz).

    PyMuPDF preserves reading order better than pdfminer for multi-column
    layouts and handles embedded image-text layers via the text layer.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        List of page-text strings, one entry per page.
    """
    pages: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            # "text" mode sorts by reading order (left-right, top-bottom)
            pages.append(page.get_text("text"))
    return pages


def _extract_pdfminer(pdf_path: Path) -> List[str]:
    """Extract per-page text using pdfminer.six.

    Iterates through the layout tree page-by-page and concatenates all
    text from LTTextContainer elements.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        List of page-text strings.
    """
    pages: List[str] = []
    try:
        for page_layout in extract_pages(str(pdf_path)):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_text += element.get_text()
            pages.append(page_text)
    except Exception as exc:
        log.warning("[pdf_ingest] pdfminer error: %s", exc)
    return pages


def _extract_pypdf(pdf_path: Path) -> List[str]:
    """Extract per-page text using pypdf.

    pypdf is a pure-Python library with broad PDF version support.
    Text quality may be lower than PyMuPDF for complex layouts.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        List of page-text strings.
    """
    pages: List[str] = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception as exc:
        log.warning("[pdf_ingest] pypdf error: %s", exc)
    return pages


def _extract_raw(pdf_path: Path) -> List[str]:
    """Last-resort: read raw PDF bytes and decode printable characters.

    This heuristic works surprisingly well for simple PDFs that store text
    as literal ASCII/UTF-8 strings.  Returns a single page entry containing
    all extractable text.

    Args:
        pdf_path: Path to the source PDF.

    Returns:
        Single-element list with extracted text, or empty list on failure.
    """
    try:
        raw = pdf_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        # Strip PDF binary operator noise, keep printable text runs
        printable = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text
        )
        return [printable]
    except Exception as exc:
        log.error("[pdf_ingest] Raw decode failed: %s", exc)
        return []
