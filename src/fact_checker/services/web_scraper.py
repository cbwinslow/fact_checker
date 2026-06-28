"""services/web_scraper.py - Web article scraper for the fact-checker pipeline.

Fetches and cleans text content from web article URLs so they can flow
through the standard claim-extraction and fact-checking pipeline.

Extraction strategy (tried in order):
  1. trafilatura  - best main-content extraction; removes boilerplate,
                   navigation, ads and side-bars automatically.
  2. BeautifulSoup (bs4) - HTML parser fallback; strips noise elements
                           manually and extracts body text.
  3. Regex tag stripper   - minimal last-resort; removes all HTML tags.

The scraper uses a browser-like User-Agent to avoid basic bot detection
and caps downloads at 5 MB to prevent runaway memory usage.

Dependencies (all optional - at least one should be installed)::

    pip install trafilatura         # recommended
    pip install beautifulsoup4 lxml # fallback
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple
from uuid import UUID

import httpx

from ..models import IngestSource, TranscriptSegment
from .file_router import _text_to_segments

# Import search provider utilities for quote extraction
try:
    from .search_providers import extract_quotes, Quote
    _SEARCH_PROVIDERS_AVAILABLE = True
except ImportError:
    _SEARCH_PROVIDERS_AVAILABLE = False
    class _Quote:
        pass
    Quote = _Quote  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import trafilatura
    _TRAFILATURA_AVAILABLE = True
except ImportError:
    _TRAFILATURA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (compatible; FactCheckerBot/1.0; "
    "+https://github.com/cbwinslow/fact_checker)"
)
_REQUEST_TIMEOUT_SEC = 30
_MAX_CONTENT_BYTES   = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def scrape_article(
    job_id: UUID,
    url: str,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Fetch and extract main text from a web article URL.

    Downloads the page HTML, extracts the main article body using the
    best available library, and converts the result to a list of
    TranscriptSegment objects ready for claim extraction.

    Args:
        job_id: UUID of the owning pipeline job.
        url:    Full URL of the article to scrape.

    Returns:
        Tuple of ``(List[TranscriptSegment], IngestSource.WEB_ARTICLE)``.
        Returns an empty list if no text could be extracted.

    Raises:
        httpx.HTTPStatusError:  If the server returns a 4xx/5xx response.
        httpx.TimeoutException: If the request exceeds the timeout.
    """
    log.info("[web_scraper] Fetching: %s", url)
    html = await _fetch_html(url)
    text, quotes = html_to_text(html, url=url, claim_text="")
    
    if not text.strip():
        log.warning("[web_scraper] No text extracted from %s", url)
        return [], IngestSource.WEB_ARTICLE
    
    segments = _text_to_segments(job_id=job_id, text=text, source_name=url)
    log.info(
        "[web_scraper] %d segments from %s (%.1f kB)",
        len(segments), url, len(text) / 1024,
    )
    return segments, IngestSource.WEB_ARTICLE


# ---------------------------------------------------------------------------
# HTML fetching
# ---------------------------------------------------------------------------

async def _fetch_html(url: str) -> str:
    """Download page HTML with a browser-like User-Agent header.

    Caps the download body at ``_MAX_CONTENT_BYTES`` to prevent runaway
    memory usage on large or misconfigured pages.

    Args:
        url: The page URL to fetch.

    Returns:
        Decoded HTML string.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx HTTP responses.
    """
    headers = {"User-Agent": _USER_AGENT}
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_SEC,
        follow_redirects=True,
        headers=headers,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content  = resp.content[:_MAX_CONTENT_BYTES]
        encoding = resp.encoding or "utf-8"
        return content.decode(encoding, errors="replace")


# ---------------------------------------------------------------------------
# HTML -> plain text  (also exported for use by file_router local HTML path)
# ---------------------------------------------------------------------------

def html_to_text(html: str, url: str = "", claim_text: str = "") -> tuple[str, list]:
    """Extract main article text from an HTML string, optionally extracting quotes.
    
    Tries trafilatura first (best boilerplate removal), then
    BeautifulSoup with noise-element stripping, then a simple regex
    tag stripper as a last resort.
    
    This function is also used by :mod:`file_router` for local ``.html``
    files, so it must not make any network calls.
    
    Args:
        html: Raw HTML string.
        url:  Source URL hint passed to trafilatura for better extraction
              (optional; does not trigger a network request).
        claim_text: Optional claim text to extract relevant quotes for.
    
    Returns:
        Tuple of (cleaned plain-text string, list of Quote objects).
    """
    extracted_text = ""
    quotes = []
    
    if _TRAFILATURA_AVAILABLE:
        extracted = trafilatura.extract(
            html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if extracted and len(extracted.strip()) > 100:
            log.debug("[web_scraper] trafilatura: %d chars", len(extracted))
            extracted_text = extracted.strip()
    
    if not extracted_text and _BS4_AVAILABLE:
        extracted = _bs4_extract(html)
        if extracted and len(extracted.strip()) > 50:
            log.debug("[web_scraper] BeautifulSoup: %d chars", len(extracted))
            extracted_text = extracted.strip()
    
    if not extracted_text:
        log.debug("[web_scraper] Falling back to regex tag stripper")
        extracted_text = _strip_tags(html).strip()
    
    # Extract quotes if claim_text provided and search_providers available
    if claim_text and _SEARCH_PROVIDERS_AVAILABLE and extracted_text:
        quotes = extract_quotes(extracted_text, claim_text, max_quotes=3)
        log.debug("[web_scraper] Extracted %d quotes for claim", len(quotes))
    
    return extracted_text, quotes


def _bs4_extract(html: str) -> str:
    """Extract body text from HTML using BeautifulSoup.

    Removes script, style, navigation, header, footer, and aside
    elements before extracting remaining text with whitespace
    normalisation.  Attempts to find the main article container first
    (``<article>``, ``<main>``, or elements with content-related IDs/
    classes) to reduce boilerplate.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned text with normalised whitespace.
    """
    parser = "lxml" if _lxml_available() else "html.parser"
    soup = BeautifulSoup(html, parser)

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                      "aside", "form", "noscript", "iframe", "button"]):
        tag.decompose()

    # Prefer a clearly-scoped article container
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|article|main", re.I))
        or soup.find(class_=re.compile(r"content|article|post[-_]body", re.I))
        or soup.body
    )
    root = main if main is not None else soup

    text = root.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _strip_tags(html: str) -> str:
    """Minimal regex-based HTML tag and entity stripper.

    Used only as a last resort when no HTML parsing library is available.
    Handles the most common HTML entities and collapses whitespace.

    Args:
        html: Raw HTML string.

    Returns:
        Text with tags and entities removed, whitespace collapsed.
    """
    text = re.sub(r"<[^>]+>", " ", html)
    entity_map = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
    }
    for entity, replacement in entity_map.items():
        text = text.replace(entity, replacement)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _lxml_available() -> bool:
    """Return True if the lxml library is importable."""
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False
