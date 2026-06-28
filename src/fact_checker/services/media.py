"""MediaService - FFmpeg/ffprobe utilities for media normalization."""
from __future__ import annotations
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import get_settings

log = logging.getLogger(__name__)


def _require_binary(name: str) -> str:
    """Resolve binary path or raise a clear error."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found on PATH. Install ffmpeg: https://ffmpeg.org/download.html"
        )
    return path


async def probe(media_path: Path) -> dict:
    """Run ffprobe and return parsed JSON metadata."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(media_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
    return json.loads(stdout)


async def extract_audio(media_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Extract audio as 16kHz mono WAV (optimal for Whisper ASR)."""
    ffmpeg = _require_binary("ffmpeg")
    out_dir = output_dir or get_settings().media_cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{media_path.stem}_audio.wav"

    if out_path.exists():
        log.debug("Audio already extracted: %s", out_path)
        return out_path

    cmd = [
        ffmpeg, "-y",
        "-i", str(media_path),
        "-vn",                    # no video
        "-ar", "16000",           # 16 kHz sample rate
        "-ac", "1",               # mono
        "-c:a", "pcm_s16le",      # 16-bit PCM
        str(out_path),
    ]
    log.info("Extracting audio: %s -> %s", media_path.name, out_path.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr.decode()[-500:]}")
    return out_path


async def get_duration(media_path: Path) -> float:
    """Return duration in seconds."""
    meta = await probe(media_path)
    return float(meta.get("format", {}).get("duration", 0.0))
