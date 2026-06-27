"""ingest_skills.py - Reusable skill functions for media ingest and chunking.

File: src/fact_checker/skills/ingest_skills.py

Provides stateless utility functions used by the MediaRouter and the harness
pipeline to detect media types, intelligently chunk text and transcript
segments, merge overly-short segments, and estimate the computational cost
of processing a given input before committing to the full pipeline run.

All functions are pure (no I/O, no LLM calls) and safe to unit-test
without external dependencies.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Types and constants
# ---------------------------------------------------------------------------

MediaType = Literal[
    "video",
    "audio",
    "image",
    "pdf",
    "docx",
    "text",
    "html",
    "url_youtube",
    "url_web",
    "unknown",
]

# Extension-to-media-type mapping
_EXT_MAP: dict[str, MediaType] = {
    # Video
    ".mp4": "video", ".mkv": "video", ".mov": "video",
    ".avi": "video", ".webm": "video", ".flv": "video",
    # Audio
    ".mp3": "audio", ".m4a": "audio", ".ogg": "audio",
    ".flac": "audio", ".wav": "audio", ".aac": "audio",
    ".opus": "audio", ".wma": "audio",
    # Image
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".gif": "image", ".bmp": "image", ".webp": "image",
    ".tiff": "image", ".tif": "image",
    # Document
    ".pdf": "pdf",
    ".docx": "docx", ".doc": "docx",
    # Text
    ".txt": "text", ".md": "text", ".rst": "text", ".csv": "text",
    # HTML
    ".html": "html", ".htm": "html",
}

# YouTube URL patterns
_YOUTUBE_PATTERNS = (
    r"youtube\.com/watch",
    r"youtu\.be/",
    r"youtube\.com/shorts/",
    r"youtube\.com/live/",
)

# Approximate tokens per second of audio/video for cost estimation
_TOKENS_PER_AUDIO_SECOND = 3.5  # Whisper output density
_TOKENS_PER_TEXT_CHAR = 0.25    # Rough token/char ratio


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_media_type(source: str) -> MediaType:
    """Detect the media type of a file path or URL string.

    Detection order:
    1. YouTube URL patterns -> ``url_youtube``.
    2. Other http/https URLs -> ``url_web``.
    3. File extension lookup -> specific media type.
    4. Unknown if no match.

    Args:
        source: File path string or URL string.

    Returns:
        One of the ``MediaType`` literal values.
    """
    s = source.strip()

    # URL detection first
    if s.startswith(("http://", "https://", "//")):
        for pattern in _YOUTUBE_PATTERNS:
            if re.search(pattern, s, re.IGNORECASE):
                return "url_youtube"
        return "url_web"

    # File extension lookup
    suffix = Path(s).suffix.lower()
    return _EXT_MAP.get(suffix, "unknown")


def chunk_text_segments(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 150,
    sentence_boundary: bool = True,
) -> List[dict]:
    """Split a long text string into overlapping chunks for embedding.

    Chunks are created by splitting at sentence boundaries when possible,
    with an optional overlap window to preserve context across chunk edges.
    Each chunk is returned as a dict matching the ``TranscriptSegment`` schema
    so chunks can flow through the standard pipeline.

    Args:
        text:              Input text string to chunk.
        max_chars:         Maximum characters per chunk.  Default 1500.
        overlap_chars:     Characters of overlap between adjacent chunks.
                           Default 150.
        sentence_boundary: If True, attempt to break at sentence boundaries
                           (period/exclamation/question mark followed by space).
                           Default True.

    Returns:
        List of segment dicts with keys:
        ``text`` (str), ``start_sec`` (float, 0.0 for text), ``end_sec`` (float, 0.0),
        ``chunk_index`` (int), ``char_start`` (int), ``char_end`` (int).
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if sentence_boundary and end < len(text):
            # Search backwards for the nearest sentence boundary
            boundary_match = None
            for m in re.finditer(r"[.!?]\s", text[start:end]):
                boundary_match = m
            if boundary_match:
                end = start + boundary_match.end()

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_sec": 0.0,
                "end_sec": 0.0,
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
            })
            chunk_index += 1

        # Move forward with overlap
        start = max(start + 1, end - overlap_chars)

    return chunks


def merge_short_segments(
    segments: List[dict],
    min_words: int = 8,
    max_merged_chars: int = 800,
) -> List[dict]:
    """Merge overly-short transcript segments into their neighbours.

    Short segments (below ``min_words`` word count) produce poor claim
    extraction results because they lack sufficient context.  This function
    greedily merges short segments with the following segment, respecting
    the ``max_merged_chars`` ceiling.

    Args:
        segments:        List of segment dicts with ``text``, ``start_sec``,
                         and ``end_sec`` keys.
        min_words:       Segments with fewer than this many words are
                         candidates for merging.  Default 8.
        max_merged_chars: Maximum characters the merged segment may have.
                          Default 800.

    Returns:
        New list of segment dicts with short segments merged in.
        ``start_sec`` and ``end_sec`` span the merged range.
    """
    if not segments:
        return []

    merged: list[dict] = []
    buffer: Optional[dict] = None

    for seg in segments:
        if buffer is None:
            buffer = dict(seg)
            continue

        word_count = len(buffer.get("text", "").split())
        combined_len = len(buffer.get("text", "")) + len(seg.get("text", "")) + 1

        if word_count < min_words and combined_len <= max_merged_chars:
            # Merge: extend buffer
            buffer["text"] = buffer["text"].rstrip() + " " + seg["text"].lstrip()
            buffer["end_sec"] = seg.get("end_sec", buffer["end_sec"])
        else:
            merged.append(buffer)
            buffer = dict(seg)

    if buffer is not None:
        merged.append(buffer)

    return merged


def estimate_processing_cost(
    source: str,
    duration_sec: Optional[float] = None,
    text_length: Optional[int] = None,
) -> dict:
    """Estimate the token and time cost of processing a given input.

    Used by the harness to decide whether to apply cost-saving optimisations
    (e.g. fewer keyframes, skip deep research for low-priority claims) or to
    warn the user about expensive long-form inputs.

    Estimation model:
    - Video/audio:  tokens = duration_sec * _TOKENS_PER_AUDIO_SECOND.
    - Text/PDF/URL: tokens = text_length * _TOKENS_PER_TEXT_CHAR.
    - Unknown:      returns a conservative ``medium`` estimate.

    Cost tiers:
    - ``low``:    < 5,000 estimated tokens
    - ``medium``: 5,000 - 25,000 tokens
    - ``high``:   > 25,000 tokens

    Args:
        source:       File path or URL string (used for media type detection).
        duration_sec: Duration in seconds for audio/video inputs.
        text_length:  Character count for text-based inputs.

    Returns:
        Dict with keys:
        ``media_type``, ``estimated_tokens`` (int), ``cost_tier`` (str),
        ``estimated_minutes`` (float), ``recommendations`` (list[str]).
    """
    media_type = detect_media_type(source)
    estimated_tokens = 0
    recommendations: list[str] = []

    if media_type in ("video", "audio", "url_youtube") and duration_sec is not None:
        estimated_tokens = int(duration_sec * _TOKENS_PER_AUDIO_SECOND)
        if duration_sec > 3600:
            recommendations.append("Input exceeds 1 hour; consider processing in segments.")
    elif text_length is not None:
        estimated_tokens = int(text_length * _TOKENS_PER_TEXT_CHAR)
    else:
        estimated_tokens = 10_000  # Conservative default
        recommendations.append("Could not estimate cost precisely; using conservative default.")

    if estimated_tokens < 5_000:
        tier = "low"
    elif estimated_tokens < 25_000:
        tier = "medium"
    else:
        tier = "high"
        recommendations.append(
            "High token estimate; deep research will be limited to top-priority claims."
        )

    # Rough wall-clock estimate: ~500 tokens/second LLM throughput
    estimated_minutes = round(estimated_tokens / 500 / 60, 2)

    return {
        "media_type": media_type,
        "estimated_tokens": estimated_tokens,
        "cost_tier": tier,
        "estimated_minutes": estimated_minutes,
        "recommendations": recommendations,
    }
