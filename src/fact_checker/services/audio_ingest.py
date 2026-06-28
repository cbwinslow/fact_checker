"""services/audio_ingest.py - Audio-only file ingestor for the fact-checker pipeline.

Handles standalone audio files (MP3, M4A, OGG, FLAC, WAV, AAC, OPUS, AIFF, WMA)
by normalising them to 16 kHz mono WAV via ffmpeg and then running
faster-whisper ASR to produce TranscriptSegment lists.

For video files use :mod:`services.ingest` (which also calls Whisper
internally and additionally handles yt-dlp caption extraction).
This module is specifically for audio-only inputs where there is no video
track and no possibility of pre-existing captions.

Pipeline::

    audio file  -->  ffmpeg normalise (16kHz mono WAV)  -->  faster-whisper ASR
                     (skipped for native .wav/.flac/.ogg)         |
                                                                   v
                                                        List[TranscriptSegment]

Dependencies::

    pip install faster-whisper   # required
    ffmpeg must be on PATH       # required for normalisation
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
from uuid import UUID

from ..config import get_settings
from ..models import IngestSource, TranscriptSegment

log = logging.getLogger(__name__)

# Formats accepted natively by faster-whisper without re-encoding
_WHISPER_NATIVE_FORMATS = {".wav", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_audio_file(
    job_id: UUID,
    audio_path: Path,
) -> Tuple[List[TranscriptSegment], IngestSource]:
    """Transcribe an audio file using faster-whisper ASR.

    The audio is first normalised to 16 kHz mono WAV using ffmpeg when the
    input is not already in a natively-supported format.  Normalisation
    runs in a thread-pool executor so it does not block the async event
    loop.

    Args:
        job_id:     UUID of the owning pipeline job.
        audio_path: Path to the input audio file (any format supported
                    by ffmpeg).

    Returns:
        Tuple of ``(List[TranscriptSegment], IngestSource.WHISPER_ASR)``.

    Raises:
        FileNotFoundError: If ``audio_path`` does not exist on disk.
        RuntimeError:      If ffmpeg or faster-whisper encounters an error.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(
            f"[audio_ingest] File not found: {audio_path}"
        )

    log.info("[audio_ingest] Ingesting: %s", audio_path.name)

    loop = asyncio.get_event_loop()
    wav_path = await loop.run_in_executor(None, _normalise_to_wav, audio_path)
    segments  = await _transcribe_whisper(wav_path, job_id)

    log.info(
        "[audio_ingest] %d segments from %s via Whisper",
        len(segments), audio_path.name,
    )
    return segments, IngestSource.WHISPER_ASR


# ---------------------------------------------------------------------------
# Audio normalisation
# ---------------------------------------------------------------------------

def _normalise_to_wav(audio_path: Path) -> Path:
    """Convert an audio file to 16 kHz mono PCM WAV using ffmpeg.

    Files already in a natively-supported format (``.wav``, ``.flac``,
    ``.ogg``) are returned unchanged to avoid unnecessary re-encoding.

    The output WAV is written to a temporary directory and will persist
    for the duration of the process (Python's ``tempfile`` will clean it
    up on interpreter shutdown unless the OS reclaims it first).

    Args:
        audio_path: Source audio file.

    Returns:
        Path to the normalised 16 kHz mono WAV file.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    if audio_path.suffix.lower() in _WHISPER_NATIVE_FORMATS:
        log.debug(
            "[audio_ingest] Native format %s - skipping normalisation.",
            audio_path.suffix,
        )
        return audio_path

    tmp_dir  = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / f"{audio_path.stem}_16k_mono.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i",    str(audio_path),
        "-ar",   "16000",      # 16 kHz sample rate required by Whisper
        "-ac",   "1",           # mono
        "-c:a",  "pcm_s16le",  # 16-bit signed little-endian PCM
        str(wav_path),
    ]
    log.debug("[audio_ingest] ffmpeg: %s -> %s", audio_path.name, wav_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[audio_ingest] ffmpeg failed:\n{result.stderr[-2000:]}"
        )
    log.info("[audio_ingest] Normalised -> %s", wav_path.name)
    return wav_path


# ---------------------------------------------------------------------------
# Whisper ASR
# ---------------------------------------------------------------------------

async def _transcribe_whisper(
    wav_path: Path,
    job_id: UUID,
) -> List[TranscriptSegment]:
    """Run faster-whisper on a 16 kHz mono WAV and return TranscriptSegments.

    Model inference runs in a thread-pool executor via
    ``asyncio.get_event_loop().run_in_executor`` so it does not block the
    async event loop during the (potentially slow) CPU/GPU inference.

    Model configuration is read from :data:`~fact_checker.config.settings`:

    - ``whisper_model_size``   - e.g. ``"base"``, ``"small"``, ``"medium"``
    - ``whisper_device``       - ``"cpu"`` or ``"cuda"``
    - ``whisper_compute_type`` - ``"int8"``, ``"float16"``, etc.

    Args:
        wav_path: Path to the normalised 16 kHz mono WAV file.
        job_id:   UUID for the owning pipeline job.

    Returns:
        List of :class:`~fact_checker.models.TranscriptSegment` objects
        with ``start_sec``, ``end_sec``, and ``text`` populated.
    """
    loop = asyncio.get_event_loop()

    def _run() -> List[TranscriptSegment]:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            get_settings().whisper_model_size,
            device=get_settings().whisper_device,
            compute_type=get_settings().whisper_compute_type,
        )
        raw_segs, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            language="en",
            vad_filter=True,   # skip silent regions
            vad_parameters={"min_silence_duration_ms": 500},
        )
        log.info(
            "[audio_ingest] Whisper: lang=%s prob=%.2f duration=%.1fs",
            info.language, info.language_probability, info.duration,
        )
        result: List[TranscriptSegment] = []
        for seg in raw_segs:
            text = seg.text.strip()
            if text:
                result.append(TranscriptSegment(
                    job_id=job_id,
                    start_sec=seg.start,
                    end_sec=seg.end,
                    text=text,
                ))
        return result

    return await loop.run_in_executor(None, _run)
