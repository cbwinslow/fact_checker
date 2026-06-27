"""image_skills.py - Reusable skill functions for image and frame processing.

File: src/fact_checker/skills/image_skills.py

Provides stateless utility functions used by the ImageAnalystAgent and the
harness pipeline to select informative keyframes, post-process raw OCR text,
tag structured objects from vision-LLM analysis output, and correlate image
timestamps with transcript segments for provenance tracking.

All functions are pure (no I/O, no LLM calls) and safe to unit-test
without external dependencies.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def select_keyframes(
    frame_paths: List[str],
    total_duration_sec: float,
    max_frames: int = 12,
    min_interval_sec: float = 30.0,
) -> List[str]:
    """Select a representative subset of video frames for analysis.

    Strategy: distribute frames evenly across the video duration while
    respecting a minimum inter-frame interval to avoid redundant nearby
    frames.  The first and last frames are always included if available.

    Args:
        frame_paths: Ordered list of extracted frame file paths.  Assumes
                     frames are named with their timestamp, e.g.
                     ``frame_0030.jpg`` (seconds), or in order of extraction.
        total_duration_sec: Total video duration in seconds.
        max_frames:         Maximum number of frames to return.  Default 12.
        min_interval_sec:   Minimum gap (seconds) between selected frames.
                            Default 30 seconds.

    Returns:
        Subset list of frame paths (at most ``max_frames`` entries).
    """
    if not frame_paths:
        return []

    n = len(frame_paths)
    if n <= max_frames:
        return frame_paths

    # Always include first and last
    selected: list[str] = [frame_paths[0]]
    last_ts = 0.0

    if total_duration_sec > 0 and n > 1:
        interval = total_duration_sec / (n - 1)
    else:
        interval = 1.0

    for i in range(1, n - 1):
        current_ts = i * interval
        if (current_ts - last_ts) >= min_interval_sec:
            selected.append(frame_paths[i])
            last_ts = current_ts
            if len(selected) >= max_frames - 1:
                break

    if frame_paths[-1] not in selected:
        selected.append(frame_paths[-1])

    return selected[:max_frames]


def postprocess_ocr_text(raw_ocr: str) -> str:
    """Clean and normalise raw OCR text extracted from an image.

    Applies the following transformations:
    1. Strip surrounding whitespace.
    2. Collapse multiple consecutive newlines to a single newline.
    3. Remove common OCR artefacts: isolated characters between spaces,
       control characters, and non-printable Unicode.
    4. Normalise common OCR digit/letter confusions:
       - l -> 1 when surrounded by digits (e.g. '2l3' -> '213')
       - O -> 0 when between digits
    5. Collapse multiple spaces within a line to a single space.

    Args:
        raw_ocr: Raw OCR string as returned by the vision LLM or a local
                 OCR engine.

    Returns:
        Cleaned OCR string, or empty string if input was empty/whitespace.
    """
    if not raw_ocr or not raw_ocr.strip():
        return ""

    text = raw_ocr.strip()
    # Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix OCR digit/letter confusion: l between digits
    text = re.sub(r"(?<=\d)l(?=\d)", "1", text)
    # Fix O between digits
    text = re.sub(r"(?<=\d)O(?=\d)", "0", text)
    # Collapse multiple spaces on a line
    text = re.sub(r"[ \t]+", " ", text)
    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def tag_objects_from_analysis(analysis: dict) -> List[Dict[str, object]]:
    """Extract and enrich a structured object tag list from an ImageAnalysis dict.

    Takes the raw dict output of the vision LLM (as parsed by
    ``ImageAnalystAgent``) and returns a flat list of object tags suitable
    for indexing in the vector store or for display in the UI.

    Enrichment steps:
    - Filter out objects with confidence < 0.3 (likely hallucinations).
    - Normalise label strings: strip punctuation, lower-case.
    - Flag objects that contain visible text (text_content is not empty).
    - Attach the source image path if provided in the analysis dict.

    Args:
        analysis: Dict with at least an ``objects`` key containing a list
                  of object dicts (``label``, ``confidence``,
                  ``text_content``).

    Returns:
        List of enriched object tag dicts with keys:
        ``label``, ``confidence``, ``has_text``, ``text_content``,
        ``source_image``.
    """
    raw_objects: list[dict] = analysis.get("objects", [])
    source_image: str = analysis.get("source_image", "")

    tags: list[dict] = []
    for obj in raw_objects:
        confidence = float(obj.get("confidence", 0.0))
        if confidence < 0.3:
            continue
        label = re.sub(r"[^\w\s-]", "", str(obj.get("label", ""))).strip().lower()
        if not label:
            continue
        text_content = str(obj.get("text_content") or "").strip()
        tags.append({
            "label": label,
            "confidence": round(confidence, 3),
            "has_text": bool(text_content),
            "text_content": text_content,
            "source_image": source_image,
        })

    # Sort by confidence descending
    tags.sort(key=lambda x: x["confidence"], reverse=True)
    return tags


def correlate_frames_to_transcript(
    frame_timestamps: List[float],
    segments: List[dict],
    window_sec: float = 5.0,
) -> Dict[int, List[int]]:
    """Map each video frame to the nearest overlapping transcript segments.

    For each frame timestamp, find all transcript segments whose time window
    overlaps within ``window_sec`` of the frame.  This correlation allows the
    pipeline to attach visual evidence to spoken claims.

    Args:
        frame_timestamps: List of frame timestamps in seconds (one per frame).
        segments:         List of transcript segment dicts, each with
                          ``start_sec`` and ``end_sec`` float keys.
        window_sec:       Seconds of tolerance on either side of a frame
                          timestamp when searching for matching segments.
                          Default 5.0 seconds.

    Returns:
        Dict mapping frame index (int) -> list of matching segment indices
        (int).  Frames with no matching segments map to an empty list.
    """
    correlation: dict[int, list[int]] = {}

    for fi, frame_ts in enumerate(frame_timestamps):
        matching: list[int] = []
        for si, seg in enumerate(segments):
            seg_start = float(seg.get("start_sec", 0.0))
            seg_end = float(seg.get("end_sec", 0.0))
            # Check if frame falls within expanded segment window
            if (seg_start - window_sec) <= frame_ts <= (seg_end + window_sec):
                matching.append(si)
        correlation[fi] = matching

    return correlation
