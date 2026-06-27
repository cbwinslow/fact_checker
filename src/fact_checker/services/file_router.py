"""services/file_router.py - Universal media/file input router for the fact-checker pipeline.

The MediaRouter is the single entry point for ALL input types. Given any
combination of a URL string, a local file path, or a list of image paths,
it detects the media type and dispatches to the appropriate ingest layer:

  URL          ──► YouTube / web-article / social-media
  .pdf         ──► pdf_ingest
  .mp3/.m4a/…  ──► audio_ingest  (Whisper ASR)
  .mp4/.mkv/…  ──► services/ingest (yt-dlp + Whisper)
  .jpg/.png/…  ──► vision pipeline (image_analyst)
  .txt/.md/…   ──► plain-text reader
  .docx        ──► python-docx reader
  .html/.htm   ──► web_scraper (local HTML)
  http(s)://   ──► web_scraper or YouTube ingest

All ingestors return (List[TranscriptSegment], IngestSource) so the harness
never needs to know which ingestor was used.

Dependencies:
    pip install httpx python-docx trafilatura beautifulsoup4 lxml
"""
from __future__ import annotations

import logging
import mimetypes
import re
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

from ..models import IngestSource, TranscriptSegment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIME / extension groups
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".wmv", ".ts", ".3gp"}
_AUDIO_EXTS = {".mp3", ".m4a", ".ogg", ".flac", ".wav", ".aac", ".opus", ".wma", ".aiff"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic"}
_PDF_EXTS   = {".pdf"}
_TEXT_EXTS  = {".txt", ".md", ".rst", ".csv", ".log"}
_DOCX_EXTS  = {".docx", ".doc"}
_HTML_EXTS  = {".html", ".htm"}

_YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?("
    r"youtube\.com/(watch\?v=|shorts/|embed/)|"
    r"youtu\.be/"
    r")[a-zA-Z0-9_-]{11}"
)


class MediaRouter:
    """Detect input type and route to the correct ingest pipeline.

    This is the single front-door for all fact-checker input types.  It
    normalises the variety of input formats (URLs, local files, images) into
    the common ``(List[TranscriptSegment], IngestSource)`` tuple that every
    downstream pipeline stage expects.

    The router is stateless and safe to instantiate once at application
    startup and reuse across requests.

    Example::

        router = MediaRouter()

        # YouTube video
        segs, source = await router.route(job_id, url="https://youtu.be/dQw4w9WgXcQ")

        # Local PDF
        segs, source = await router.route(job_id, local_path=Path("report.pdf"))

        # Web article
        segs, source = await router.route(job_id, url="https://apnews.com/article/...")
    """

    async def route(
        self,
        job_id: UUID,
        url: Optional[str] = None,
        local_path: Optional[Path] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route the input to the correct ingest layer.

        Priority order when multiple inputs are supplied:
          1. ``local_path`` - most explicit, processed first.
          2. ``url``        - remote resource.
          3. ``image_paths``- vision-only job; returns empty transcript.

        Args:
            job_id:      UUID for the current pipeline job.
            url:         Remote URL (YouTube, web article, direct media link).
            local_path:  Path to a local file of any supported type.
            image_paths: List of image file paths for vision-only jobs.
                         Returns an empty segment list; caller passes
                         image_paths directly to the harness vision stage.

        Returns:
            Tuple of ``(List[TranscriptSegment], IngestSource)``.

        Raises:
            ValueError:      If no input is provided or the type is unsupported.
            FileNotFoundError: If ``local_path`` does not exist.
        """
        if local_path is not None:
            return await self._route_local(job_id, Path(local_path))
        if url is not None:
            return await self._route_url(job_id, url)
        if image_paths:
            log.info(
                "[router] Image-only job %s (%d images) - no transcript.",
                job_id, len(image_paths),
            )
            return [], IngestSource.IMAGE
        raise ValueError(
            "MediaRouter.route() requires at least one of: url, local_path, image_paths"
        )

    # ------------------------------------------------------------------
    # Local file routing
    # ------------------------------------------------------------------

    async def _route_local(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Detect file type from extension and MIME type, then dispatch.

        Args:
            job_id: UUID for the pipeline job.
            path:   Absolute or relative path to the local file.

        Returns:
            ``(segments, source)`` tuple from the appropriate ingestor.
        """
        if not path.exists():
            raise FileNotFoundError(f"[router] Local file not found: {path}")

        ext = path.suffix.lower()
        log.info("[router] Local file: %s (ext=%s)", path.name, ext)

        if ext in _VIDEO_EXTS:
            return await self._ingest_video(job_id, path)
        if ext in _AUDIO_EXTS:
            return await self._ingest_audio(job_id, path)
        if ext in _PDF_EXTS:
            return await self._ingest_pdf(job_id, path)
        if ext in _TEXT_EXTS:
            return await self._ingest_text(job_id, path)
        if ext in _DOCX_EXTS:
            return await self._ingest_docx(job_id, path)
        if ext in _HTML_EXTS:
            return await self._ingest_local_html(job_id, path)
        if ext in _IMAGE_EXTS:
            log.info("[router] Image file - returning empty transcript (vision-only).")
            return [], IngestSource.IMAGE

        # Fallback: MIME sniffing
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            if mime.startswith("video/"):
                return await self._ingest_video(job_id, path)
            if mime.startswith("audio/"):
                return await self._ingest_audio(job_id, path)
            if mime == "application/pdf":
                return await self._ingest_pdf(job_id, path)
            if mime.startswith("text/"):
                return await self._ingest_text(job_id, path)

        raise ValueError(
            f"[router] Unsupported file type: {path.suffix!r} (MIME: {mime})"
        )

    # ------------------------------------------------------------------
    # URL routing
    # ------------------------------------------------------------------

    async def _route_url(
        self, job_id: UUID, url: str
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route a URL to the correct remote ingestor.

        YouTube URLs go through the 3-layer video ingest pipeline.
        Direct media file URLs are downloaded and ingested appropriately.
        Everything else is treated as a web article and scraped.

        Args:
            job_id: UUID for the pipeline job.
            url:    Remote resource URL.

        Returns:
            ``(segments, source)`` tuple.
        """
        if _YOUTUBE_RE.search(url):
            log.info("[router] YouTube URL -> video ingest")
            return await self._ingest_video_url(job_id, url)

        # Check if URL path ends with a known media extension
        url_path = url.split("?")[0].lower()
        for ext in _VIDEO_EXTS:
            if url_path.endswith(ext):
                return await self._ingest_video_url(job_id, url)
        for ext in _AUDIO_EXTS:
            if url_path.endswith(ext):
                return await self._ingest_audio_url(job_id, url)
        for ext in _PDF_EXTS:
            if url_path.endswith(ext):
                return await self._ingest_pdf_url(job_id, url)

        # Default: web article
        log.info("[router] Web article URL -> web scraper")
        from .web_scraper import scrape_article
        return await scrape_article(job_id=job_id, url=url)

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    async def _ingest_video(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route a local video file through the existing 3-layer ingest pipeline."""
        from .ingest import ingest
        return await ingest(job_id=job_id, local_path=path)

    async def _ingest_video_url(
        self, job_id: UUID, url: str
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route a remote video/YouTube URL through the existing ingest pipeline."""
        from .ingest import ingest
        return await ingest(job_id=job_id, url=url)

    async def _ingest_audio(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route a local audio file through the dedicated audio ingestor."""
        from .audio_ingest import ingest_audio_file
        return await ingest_audio_file(job_id=job_id, audio_path=path)

    async def _ingest_audio_url(
        self, job_id: UUID, url: str
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Download a remote audio file and ingest via Whisper ASR."""
        import tempfile
        import httpx
        suffix = Path(url.split("?")[0]).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                  try:
                tmp_path = Path(tmp.name)
            async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.get(url, follow_redirects=True)
                    resp.raise_for_status()
                    tmp_path.write_bytes(resp.content)
            log.info("[router] Downloaded audio %s -> %s", url, tmp_path.name)
            from .audio_ingest import ingest_audio_file
            return await ingest_audio_file(job_id=job_id, audio_path=tmp_path)
                            finally:
            # Clean up temporary file
            try:
                tmp_path.unlink()
                log.debug(f"[router] Cleaned up temp file: {tmp_path}")
            except Exception as e:
                log.warning(f"[router] Failed to cleanup temp file {tmp_path}: {e}")

    async def _ingest_pdf(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Route a local PDF file through the PDF ingestor."""
        from .pdf_ingest import ingest_pdf
        return await ingest_pdf(job_id=job_id, pdf_path=path)

    async def _ingest_pdf_url(
        self, job_id: UUID, url: str
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Download a remote PDF and ingest it."""
        import tempfile
        import httpx
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                  try:
                tmp_path = Path(tmp.name)
            async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.get(url, follow_redirects=True)
                    resp.raise_for_status()
                    tmp_path.write_bytes(resp.content)
            from .pdf_ingest import ingest_pdf
            return await ingest_pdf(job_id=job_id, pdf_path=tmp_path)
                            finally:
            # Clean up temporary file
            try:
                tmp_path.unlink()
                log.debug(f"[router] Cleaned up temp file: {tmp_path}")
            except Exception as e:
                log.warning(f"[router] Failed to cleanup temp file {tmp_path}: {e}")

    async def _ingest_text(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Ingest a plain-text file by splitting into paragraph segments."""
        log.info("[router] Plain text ingest: %s", path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
        segments = _text_to_segments(job_id, text, source_name=path.name)
        log.info(
            "[router] Text ingest: %d segments from %s",
            len(segments), path.name,
        )
        return segments, IngestSource.TEXT_FILE

    async def _ingest_docx(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Ingest a .docx file via python-docx, falling back to raw text."""
        log.info("[router] DOCX ingest: %s", path.name)
        try:
            import docx  # python-docx
            doc = docx.Document(str(path))
            full_text = "\n\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )
        except ImportError:
            log.warning(
                "[router] python-docx not installed; reading .docx as raw text."
            )
            full_text = path.read_text(encoding="utf-8", errors="replace")
        segments = _text_to_segments(job_id, full_text, source_name=path.name)
        log.info(
            "[router] DOCX ingest: %d segments from %s",
            len(segments), path.name,
        )
        return segments, IngestSource.DOCUMENT

    async def _ingest_local_html(
        self, job_id: UUID, path: Path
    ) -> Tuple[List[TranscriptSegment], IngestSource]:
        """Ingest a local HTML file by stripping tags and extracting text."""
        log.info("[router] Local HTML ingest: %s", path.name)
        html = path.read_text(encoding="utf-8", errors="replace")
        from .web_scraper import html_to_text
        text = html_to_text(html)
        segments = _text_to_segments(job_id, text, source_name=path.name)
        return segments, IngestSource.WEB_ARTICLE


# ---------------------------------------------------------------------------
# Shared text segmentation utility  (used by multiple ingestors)
# ---------------------------------------------------------------------------

def _text_to_segments(
    job_id: UUID,
    text: str,
    source_name: str = "text",
    max_chars: int = 1500,
) -> List[TranscriptSegment]:
    """Split a plain-text string into TranscriptSegment objects.

    Splits on double-newlines (paragraphs) first, then hard-wraps any
    paragraph that exceeds ``max_chars`` at sentence boundaries to avoid
    overloading the LLM context window.

    Each segment receives synthetic ``start_sec=0.0`` / ``end_sec=0.0``
    timestamps because text documents have no audio timeline.

    Args:
        job_id:      UUID of the owning pipeline job.
        text:        Full document text.
        source_name: Descriptive label used in log messages.
        max_chars:   Maximum characters per segment before hard-wrapping.

    Returns:
        List of :class:`~fact_checker.models.TranscriptSegment` objects.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    segments: List[TranscriptSegment] = []

    for para in paragraphs:
        while len(para) > max_chars:
            chunk = para[:max_chars]
            # Prefer to break at the last sentence boundary in the chunk
            last_boundary = max(
                chunk.rfind("."),
                chunk.rfind("!"),
                chunk.rfind("?"),
            )
            if last_boundary > max_chars // 2:
                chunk = para[: last_boundary + 1]
            segments.append(TranscriptSegment(
                job_id=job_id,
                start_sec=0.0,
                end_sec=0.0,
                text=chunk.strip(),
            ))
            para = para[len(chunk) :].strip()
        if para:
            segments.append(TranscriptSegment(
                job_id=job_id,
                start_sec=0.0,
                end_sec=0.0,
                text=para,
            ))

    return segments
