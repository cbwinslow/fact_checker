"""services/vision.py - Image/frame utilities for the fact-checker pipeline.

Provides three capabilities:
  1. extract_frames()   - Pull key frames from a video file using ffmpeg.
  2. read_image_metadata() - Extract EXIF / file-level metadata from an image
                             using Pillow (+ piexif for raw EXIF tags).
  3. image_to_data_url() - Encode an image file as a base64 data URL for
                           passing to vision-capable LLMs via the OpenAI
                           messages format.

All functions are designed to be import-safe: if optional deps (Pillow,
piexif, ffmpeg-python) are missing the module still loads and returns graceful
fallbacks or raises descriptive ImportError messages at call time.
"""
from __future__ import annotations

import base64
import logging
import math
import subprocess
from pathlib import Path
from typing import List, Optional

from ..models import ImageMetadata

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False

try:
    import piexif
    _PIEXIF_AVAILABLE = True
except ImportError:
    _PIEXIF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    interval_sec: float = 30.0,
    max_frames: int = 20,
    format: str = "jpg",
    quality: int = 85,
) -> List[Path]:
    """Extract evenly-spaced key frames from a video using ffmpeg.

    Args:
        video_path:   Path to the source video file.
        output_dir:   Directory where extracted frames will be written.
        interval_sec: Extract one frame every N seconds (default 30).
        max_frames:   Hard cap on total frames extracted (default 20).
        format:       Output image format - "jpg" or "png" (default "jpg").
        quality:      JPEG quality 1-95 (ignored for PNG).

    Returns:
        Sorted list of Paths to the extracted frame files.

    Raises:
        FileNotFoundError: If ffmpeg binary is not on PATH.
        RuntimeError:      If ffmpeg exits non-zero.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Probe video duration so we can cap frames
    duration = _probe_duration(video_path)
    if duration and duration > 0:
        n_frames = min(max_frames, math.ceil(duration / interval_sec))
    else:
        n_frames = max_frames

    pattern = str(output_dir / f"frame_%04d.{format}")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_sec}",
        "-frames:v", str(n_frames),
    ]
    if format == "jpg":
        cmd += ["-q:v", str(max(1, min(31, int(31 * (1 - quality / 95)))))]
    cmd.append(pattern)

    log.info("[vision] Extracting up to %d frames from %s", n_frames, video_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[vision] ffmpeg frame extraction failed:\n{result.stderr[-2000:]}"
        )

    frames = sorted(output_dir.glob(f"frame_*.{format}"))
    log.info("[vision] Extracted %d frames to %s", len(frames), output_dir)
    return frames


def _probe_duration(video_path: Path) -> Optional[float]:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def read_image_metadata(image_path: str | Path) -> ImageMetadata:
    """Extract file-level and EXIF metadata from an image.

    Uses Pillow for basic properties and piexif for raw EXIF tags.
    Falls back gracefully if either library is unavailable or EXIF is absent.

    Args:
        image_path: Path to the image file.

    Returns:
        Populated ImageMetadata instance (fields default to None if unavailable).
    """
    image_path = Path(image_path)
    meta = ImageMetadata()

    # File size
    try:
        meta.file_size_bytes = image_path.stat().st_size
    except OSError:
        pass

    if not _PILLOW_AVAILABLE:
        log.warning("[vision] Pillow not installed - skipping image metadata.")
        return meta

    try:
        with PILImage.open(image_path) as img:
            meta.width  = img.width
            meta.height = img.height
            meta.format = img.format or image_path.suffix.lstrip(".").upper()
            meta.mode   = img.mode

            # --- Pillow EXIF (simple path) ---
            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif:
                exif_map = {TAGS.get(k, k): v for k, v in raw_exif.items()}
                meta.camera_make  = _str_or_none(exif_map.get("Make"))
                meta.camera_model = _str_or_none(exif_map.get("Model"))
                meta.software     = _str_or_none(exif_map.get("Software"))
                meta.datetime_original = _str_or_none(
                    exif_map.get("DateTimeOriginal") or exif_map.get("DateTime")
                )
                # GPS
                gps_info = exif_map.get("GPSInfo")
                if gps_info and isinstance(gps_info, dict):
                    meta.gps_latitude  = _dms_to_decimal(gps_info.get(2), gps_info.get(1))
                    meta.gps_longitude = _dms_to_decimal(gps_info.get(4), gps_info.get(3))

            # --- piexif deep dive for extra tags ---
            if _PIEXIF_AVAILABLE:
                try:
                    exif_bytes = img.info.get("exif", b"")
                    if exif_bytes:
                        exif_dict = piexif.load(exif_bytes)
                        # Flatten useful Exif IFD tags into meta.extra
                        for ifd_name, ifd in exif_dict.items():
                            if isinstance(ifd, dict):
                                for tag_id, value in ifd.items():
                                    tag_name = piexif.TAGS.get(ifd_name, {}).get(tag_id, {}).get("name", str(tag_id))
                                    if isinstance(value, bytes):
                                        try:
                                            value = value.decode("utf-8", errors="replace").strip("\x00")
                                        except Exception:
                                            value = repr(value)
                                    meta.extra[tag_name] = value
                except Exception as exc:
                    log.debug("[vision] piexif parse error for %s: %s", image_path.name, exc)

    except Exception as exc:
        log.warning("[vision] Could not read metadata for %s: %s", image_path.name, exc)

    return meta


def _str_or_none(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip() or None


def _dms_to_decimal(
    dms_tuple,
    ref: Optional[str],
) -> Optional[float]:
    """Convert a GPS DMS tuple from EXIF to decimal degrees."""
    try:
        d, m, s = dms_tuple
        # IFDRational or tuple of (numerator, denominator)
        def to_float(v):
            if isinstance(v, tuple):
                return v[0] / v[1] if v[1] != 0 else 0.0
            return float(v)
        decimal = to_float(d) + to_float(m) / 60 + to_float(s) / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return round(decimal, 6)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data-URL encoding for vision LLMs
# ---------------------------------------------------------------------------

def image_to_data_url(image_path: str | Path, *, max_side: int = 1024) -> str:
    """Encode an image as a base64 data URL suitable for vision LLM messages.

    Optionally downscales the image so the longest side is <= max_side pixels,
    keeping aspect ratio and reducing token usage.

    Args:
        image_path: Path to the source image file.
        max_side:   Max pixel dimension for downscaling (default 1024).
                    Set to 0 to skip resizing.

    Returns:
        A string like "data:image/jpeg;base64,<encoded>"
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(suffix, "image/jpeg")

    if _PILLOW_AVAILABLE and max_side > 0:
        try:
            with PILImage.open(image_path) as img:
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, PILImage.LANCZOS)
                    # Convert RGBA -> RGB for JPEG compatibility
                    if img.mode in ("RGBA", "P") and mime == "image/jpeg":
                        img = img.convert("RGB")
                    import io
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG" if mime == "image/jpeg" else "PNG", quality=85)
                    encoded = base64.b64encode(buf.getvalue()).decode()
                    return f"data:{mime};base64,{encoded}"
        except Exception as exc:
            log.warning("[vision] Pillow resize failed for %s: %s", image_path.name, exc)

    # Fallback: read raw bytes
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Convenience: fetch a remote image to a local temp file
# ---------------------------------------------------------------------------

async def fetch_image_url(url: str, dest_dir: str | Path) -> Path:
    """Download an image from a URL to dest_dir and return the local Path.

    Uses httpx async client so it integrates cleanly with the async pipeline.
    """
    import httpx
    from uuid import uuid4

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    dest = dest_dir / f"{uuid4().hex}{suffix}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    log.info("[vision] Fetched image %s -> %s (%d bytes)", url, dest.name, dest.stat().st_size)
    return dest
