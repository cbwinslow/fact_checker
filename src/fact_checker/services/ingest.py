"""IngestService - 3-layer transcript pipeline.

Layer 1: yt-dlp captions (fastest, no compute)
Layer 2: youtube-transcript-api (YouTube-only fallback)
Layer 3: faster-whisper ASR (universal, local compute)
"""
from __future__ import annotations
import asyncio
import logging
import re
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..config import get_settings
from ..models import IngestSource, TranscriptSegment

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 - yt-dlp captions
# ---------------------------------------------------------------------------

async def _fetch_ytdlp_captions(url: str, job_id: UUID) -> list[TranscriptSegment]:
    """Download auto/manual captions via yt-dlp as VTT, parse to segments."""
    import tempfile
    out_dir = Path(tempfile.mkdtemp())
    cmd = [
        "yt-dlp",
        "--write-auto-subs", "--write-subs",
        "--sub-langs", "en",
        "--sub-format", "vtt",
        "--skip-download",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    vtt_files = list(out_dir.glob("*.vtt"))
    if not vtt_files:
        raise ValueError("No VTT captions found via yt-dlp")
    return _parse_vtt(vtt_files[0], job_id)


def _parse_vtt(path: Path, job_id: UUID) -> list[TranscriptSegment]:
    """Parse a WebVTT file into TranscriptSegment list."""
    segments: list[TranscriptSegment] = []
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\s*\n([\s\S]+?)(?=\n\n|\Z)"
    )
    for m in pattern.finditer(content):
        start = _vtt_time(m.group(1))
        end = _vtt_time(m.group(2))
        text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        if text:
            segments.append(TranscriptSegment(job_id=job_id, start_sec=start, end_sec=end, text=text))
    return segments


def _vtt_time(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------------------------
# Layer 2 - youtube-transcript-api
# ---------------------------------------------------------------------------

async def _fetch_yt_transcript_api(url: str, job_id: UUID) -> list[TranscriptSegment]:
    """Use youtube-transcript-api as a second fallback for YouTube URLs."""
    from youtube_transcript_api import YouTubeTranscriptApi
    import re as _re
    vid_match = _re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not vid_match:
        raise ValueError("Cannot extract YouTube video ID from URL")
    vid_id = vid_match.group(1)
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(None, lambda: YouTubeTranscriptApi.get_transcript(vid_id))
    return [
        TranscriptSegment(
            job_id=job_id,
            start_sec=e["start"],
            end_sec=e["start"] + e["duration"],
            text=e["text"],
        )
        for e in raw
    ]


# ---------------------------------------------------------------------------
# Layer 3 - faster-whisper ASR
# ---------------------------------------------------------------------------

async def _transcribe_whisper(audio_path: Path, job_id: UUID) -> list[TranscriptSegment]:
    """Run faster-whisper locally on extracted audio."""
    from faster_whisper import WhisperModel
    loop = asyncio.get_event_loop()

    def _run():
        model = WhisperModel(
            get_settings().whisper_model_size,
            device=get_settings().whisper_device,
            compute_type=get_settings().whisper_compute_type,
        )
        segs, _ = model.transcribe(str(audio_path), beam_size=5)
        return list(segs)

    raw_segs = await loop.run_in_executor(None, _run)
    return [
        TranscriptSegment(
            job_id=job_id,
            start_sec=s.start,
            end_sec=s.end,
            text=s.text.strip(),
        )
        for s in raw_segs
        if s.text.strip()
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest(
    job_id: UUID,
    url: Optional[str] = None,
    local_path: Optional[Path] = None,
) -> tuple[list[TranscriptSegment], IngestSource]:
    """Try each layer in order; return segments + the source that worked."""
    if url:
        # Layer 1
        try:
            segs = await _fetch_ytdlp_captions(url, job_id)
            if segs:
                log.info("[ingest] Layer 1 (yt-dlp captions) succeeded: %d segments", len(segs))
                return segs, IngestSource.YOUTUBE_CAPTIONS
        except Exception as e:
            log.warning("[ingest] Layer 1 failed: %s", e)

        # Layer 2
        try:
            segs = await _fetch_yt_transcript_api(url, job_id)
            if segs:
                log.info("[ingest] Layer 2 (youtube-transcript-api) succeeded: %d segments", len(segs))
                return segs, IngestSource.YOUTUBE_TRANSCRIPT_API
        except Exception as e:
            log.warning("[ingest] Layer 2 failed: %s", e)

        # Layer 3: download audio then ASR
        log.info("[ingest] Falling back to Layer 3 (Whisper ASR) for URL: %s", url)
        from .media import extract_audio
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        dl_cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", str(tmp / "audio.wav"), url]
        proc = await asyncio.create_subprocess_exec(*dl_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        wav_files = list(tmp.glob("*.wav"))
        if not wav_files:
            raise RuntimeError("Could not download audio for Whisper ASR")
        segs = await _transcribe_whisper(wav_files[0], job_id)
        return segs, IngestSource.WHISPER_ASR

    elif local_path:
        # Local file: extract audio then ASR directly
        from .media import extract_audio
        audio = await extract_audio(local_path)
        segs = await _transcribe_whisper(audio, job_id)
        return segs, IngestSource.WHISPER_ASR

    else:
        raise ValueError("Either url or local_path must be provided")
