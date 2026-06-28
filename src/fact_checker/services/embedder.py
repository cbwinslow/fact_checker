"""services/embedder.py - Text chunking and embedding for the fact-checker pipeline.

Converts raw TranscriptSegment lists (or any plain text) into vector
embeddings that can be stored in a vector store for semantic retrieval.

Responsibilities:
  1. Chunking   - Split text into overlapping chunks that fit within the
                  embedding model's token limit.
  2. Embedding  - Encode each chunk into a dense float vector using the
                  configured embedding model.
  3. Packaging  - Return typed EmbeddedChunk objects that bundle the
                  original text, its vector, and provenance metadata.

Embedding backend resolution order:
  1. OpenAI embeddings via openrouter (if OPENROUTER_API_KEY is set)
  2. sentence-transformers local model (if installed, zero API cost)
  3. Mock embedding (random unit vector, for offline/test use)

Dependencies (all optional - at least one should be installed)::

    pip install openai                   # OpenAI-compatible embeddings
    pip install sentence-transformers    # local embeddings (no API key)
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List
from uuid import UUID, uuid4

from ..models import EmbeddedChunk, TranscriptSegment
from ..config import get_settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

# Target token budget per chunk (conservative estimate: 1 token ~ 4 chars)
DEFAULT_CHUNK_CHARS    = 800
DEFAULT_OVERLAP_CHARS  = 150  # overlap between consecutive chunks

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Module-level cache so the sentence-transformer model is only loaded once
_st_model_cache: dict = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def embed_segments(
    job_id: UUID,
    segments: List[TranscriptSegment],
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[EmbeddedChunk]:
    """Chunk and embed a list of TranscriptSegments.

    Segments are concatenated in order, then split into overlapping
    character-level chunks.  Each chunk is embedded and returned as an
    :class:`~fact_checker.models.EmbeddedChunk`.

    Args:
        job_id:        UUID of the owning pipeline job.
        segments:      Ordered list of TranscriptSegment objects.
        chunk_chars:   Target characters per chunk (default 800).
        overlap_chars: Overlap characters between adjacent chunks (default 150).

    Returns:
        List of EmbeddedChunk objects with vectors populated.
    """
    if not segments:
        return []

    # Concatenate all segment text with timestamps for context
    full_text = "\n".join(
        f"[{s.start_sec:.1f}s] {s.text}" for s in segments
    )
    chunks = _chunk_text(full_text, chunk_chars, overlap_chars)
    log.info(
        "[embedder] %d segments -> %d chunks for job %s",
        len(segments), len(chunks), job_id,
    )

    vectors = await _embed_texts(chunks)

    return [
        EmbeddedChunk(
            id=uuid4(),
            job_id=job_id,
            text=chunk,
            vector=vec,
            chunk_index=i,
            source_hash=_sha256(chunk),
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]


async def embed_texts(
    job_id: UUID,
    texts: List[str],
) -> List[EmbeddedChunk]:
    """Embed a list of plain text strings directly (no chunking).

    Useful for embedding individual claims or evidence snippets.

    Args:
        job_id: UUID of the owning pipeline job.
        texts:  List of strings to embed.

    Returns:
        List of EmbeddedChunk objects, one per input string.
    """
    if not texts:
        return []
    vectors = await _embed_texts(texts)
    return [
        EmbeddedChunk(
            id=uuid4(),
            job_id=job_id,
            text=text,
            vector=vec,
            chunk_index=i,
            source_hash=_sha256(text),
        )
        for i, (text, vec) in enumerate(zip(texts, vectors))
    ]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
) -> List[str]:
    """Split text into overlapping character-level chunks.

    Attempts to split at sentence boundaries (period/question/exclamation)
    rather than mid-sentence when the chunk boundary falls within a
    sentence.  Falls back to a hard character split if no boundary is found.

    Args:
        text:          Input text to chunk.
        chunk_chars:   Maximum characters per chunk.
        overlap_chars: Characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunk strings.
    """
    if len(text) <= chunk_chars:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to find sentence boundary near the end of the window
        window = text[start:end]
        best = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind("\n"),
        )
        split_at = (best + 1) if best > chunk_chars // 2 else end
        chunk = text[start : start + split_at].strip()
        if chunk:
            chunks.append(chunk)
        # Advance with overlap
        start = start + split_at - overlap_chars

    return chunks


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

async def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings using the best available backend.

    Resolution order:
      1. OpenAI-compatible API (via openrouter) if key is set.
      2. sentence-transformers local model if installed.
      3. Mock random unit-vector embedding for offline/test use.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input string.
    """
    s = get_settings()
    if s.openrouter_api_key.strip() and _OPENAI_AVAILABLE:
        try:
            return await _embed_openai(texts)
        except Exception as exc:
            log.warning("[embedder] OpenAI embedding failed: %s - falling back", exc)

    if _ST_AVAILABLE:
        return _embed_sentence_transformers(texts)

    log.warning("[embedder] No embedding backend available - using mock vectors.")
    return [_mock_vector(t) for t in texts]


async def _embed_openai(texts: List[str]) -> List[List[float]]:
    """Embed texts using the OpenAI-compatible embeddings API via openrouter.

    Uses ``text-embedding-3-small`` by default (1536-dim, low cost).
    The model can be overridden via ``settings.embedding_model``.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors from the API response.
    """
    import asyncio
    s = get_settings()
    client = _OpenAIClient(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
    )
    model = getattr(s, "embedding_model", "text-embedding-3-small")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.embeddings.create(input=texts, model=model),
    )
    return [item.embedding for item in response.data]


def _embed_sentence_transformers(texts: List[str]) -> List[List[float]]:
    """Embed texts using a local sentence-transformers model.

    The model is loaded once and cached in ``_st_model_cache`` for reuse.
    Default model: ``all-MiniLM-L6-v2`` (384-dim, ~90 MB, fast on CPU).
    Override via ``settings.st_model_name``.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors.
    """
    s = get_settings()
    model_name = getattr(s, "st_model_name", "all-MiniLM-L6-v2")
    if model_name not in _st_model_cache:
        log.info("[embedder] Loading sentence-transformer model: %s", model_name)
        _st_model_cache[model_name] = _ST(model_name)
    model = _st_model_cache[model_name]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [vec.tolist() for vec in embeddings]


def _mock_vector(text: str, dim: int = 384) -> List[float]:
    """Generate a deterministic pseudo-random unit vector for offline testing.

    The vector is derived from the SHA-256 hash of the input text so the
    same string always produces the same vector (useful for test assertions).

    Args:
        text: Input string.
        dim:  Vector dimensionality (default 384 to match MiniLM).

    Returns:
        Normalised float vector of length ``dim``.
    """
    seed_bytes = hashlib.sha256(text.encode()).digest()
    # Expand seed to dim floats by cycling through hash bytes
    floats = [
        ((seed_bytes[i % len(seed_bytes)] / 255.0) * 2 - 1)
        for i in range(dim)
    ]
    # L2 normalise
    magnitude = math.sqrt(sum(x * x for x in floats)) or 1.0
    return [x / magnitude for x in floats]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
