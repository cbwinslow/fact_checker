"""services/vector_store.py - Vector store for semantic context retrieval.

Provides a ChromaDB-backed vector store that indexes EmbeddedChunk objects
and supports semantic similarity search.  Used by the DeepResearchAgent to
retrieve the most relevant context chunks for a given claim before issuing
web-search queries.

Design decisions:
  - ChromaDB is used as the default backend (zero-config, runs in-process
    or as a server, supports both ephemeral and persistent collections).
  - Each pipeline job gets its own isolated ChromaDB collection named by
    its UUID, preventing cross-job data leakage.
  - A simple in-memory Python fallback (cosine similarity over a list) is
    provided for environments where ChromaDB is not installed.

Dependencies (optional - in-memory fallback used if not installed)::

    pip install chromadb
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional
from uuid import UUID

from ..models import EmbeddedChunk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VectorStore:
    """Semantic search index over EmbeddedChunk objects.

    Each instance manages one ChromaDB collection (or an in-memory
    fallback list) scoped to a single pipeline job.  Create a new
    VectorStore per job to keep context isolated.

    Example::

        store = VectorStore(job_id=job_id)
        store.add(embedded_chunks)
        results = store.search("unemployment rate 2024", top_k=5)

    Attributes:
        job_id:     UUID of the owning pipeline job.
        collection: Underlying ChromaDB collection (None if using fallback).
    """

    def __init__(
        self,
        job_id: UUID,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialise the vector store for a given job.

        Args:
            job_id:            UUID of the owning pipeline job.  Used as the
                               ChromaDB collection name.
            persist_directory: Optional filesystem path for ChromaDB to
                               persist its data.  When ``None`` an ephemeral
                               in-memory ChromaDB instance is used.
        """
        self.job_id = job_id
        self._chunks: List[EmbeddedChunk] = []  # in-memory fallback store
        self._collection = None

        if _CHROMA_AVAILABLE:
            try:
                if persist_directory:
                    client = chromadb.PersistentClient(
                        path=persist_directory,
                        settings=ChromaSettings(anonymized_telemetry=False),
                    )
                else:
                    client = chromadb.EphemeralClient(
                        settings=ChromaSettings(anonymized_telemetry=False),
                    )
                self._collection = client.get_or_create_collection(
                    name=f"job_{str(job_id).replace('-', '_')}",
                    metadata={"hnsw:space": "cosine"},
                )
                log.info("[vector_store] ChromaDB collection ready for job %s", job_id)
            except Exception as exc:
                log.warning(
                    "[vector_store] ChromaDB init failed (%s) - using in-memory fallback", exc
                )
                self._collection = None
        else:
            log.info("[vector_store] chromadb not installed - using in-memory cosine search")

    def add(self, chunks: List[EmbeddedChunk]) -> None:
        """Index a list of EmbeddedChunk objects.

        Chunks with no vector are silently skipped.  Safe to call multiple
        times; duplicate IDs are handled by ChromaDB's upsert semantics.

        Args:
            chunks: List of :class:`~fact_checker.models.EmbeddedChunk` to index.
        """
        valid = [c for c in chunks if c.vector]
        if not valid:
            return

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[str(c.id) for c in valid],
                    embeddings=[c.vector for c in valid],
                    documents=[c.text for c in valid],
                    metadatas=[{"chunk_index": c.chunk_index, "job_id": str(c.job_id)} for c in valid],
                )
                log.debug("[vector_store] Upserted %d chunks into ChromaDB", len(valid))
                return
            except Exception as exc:
                log.warning("[vector_store] ChromaDB upsert failed: %s - falling back", exc)

        # In-memory fallback
        self._chunks.extend(valid)
        log.debug("[vector_store] Stored %d chunks in memory", len(valid))

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[EmbeddedChunk]:
        """Return the top-k most similar chunks for a query vector.

        Uses ChromaDB's HNSW cosine similarity when available, otherwise
        falls back to a brute-force cosine scan over the in-memory list.

        Args:
            query_vector: Dense float query vector (same dimensionality as
                          the stored chunk vectors).
            top_k:        Maximum number of results to return (default 5).
            min_score:    Minimum cosine similarity score threshold (0-1).
                          Results below this score are excluded.

        Returns:
            List of EmbeddedChunk objects sorted by descending similarity,
            filtered by ``min_score``.
        """
        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_embeddings=[query_vector],
                    n_results=min(top_k, max(1, self._collection.count())),
                    include=["documents", "distances", "metadatas"],
                )
                chunks: List[EmbeddedChunk] = []
                docs      = results.get("documents", [[]])[0]
                distances = results.get("distances",  [[]])[0]
                ids       = results.get("ids",         [[]])[0]
                for doc, dist, cid in zip(docs, distances, ids):
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    # Convert to similarity score 0-1
                    score = 1.0 - (dist / 2.0)
                    if score >= min_score:
                        chunks.append(EmbeddedChunk(
                            id=cid,
                            job_id=self.job_id,
                            text=doc,
                            vector=[],   # not returned by query
                            chunk_index=0,
                            source_hash="",
                            similarity_score=score,
                        ))
                return chunks
            except Exception as exc:
                log.warning("[vector_store] ChromaDB query failed: %s - using fallback", exc)

        # In-memory brute-force cosine search
        return self._memory_search(query_vector, top_k, min_score)

    def _memory_search(
        self,
        query_vector: List[float],
        top_k: int,
        min_score: float,
    ) -> List[EmbeddedChunk]:
        """Brute-force cosine similarity search over the in-memory chunk list.

        Time complexity O(n * d) where n = number of chunks and d = vector
        dimensionality.  Suitable for small jobs (< 10k chunks).

        Args:
            query_vector: Dense float query vector.
            top_k:        Maximum results to return.
            min_score:    Minimum cosine similarity to include.

        Returns:
            Top-k EmbeddedChunk objects sorted by descending similarity.
        """
        scored: List[tuple[float, EmbeddedChunk]] = []
        q_norm = _l2_norm(query_vector)
        if q_norm == 0:
            return []

        for chunk in self._chunks:
            if not chunk.vector:
                continue
            score = _cosine(query_vector, chunk.vector, q_norm)
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:top_k]:
            # Return a copy with the similarity score attached
            results.append(chunk.model_copy(update={"similarity_score": score}))
        return results

    @property
    def count(self) -> int:
        """Return the number of indexed chunks."""
        if self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._chunks)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _l2_norm(v: List[float]) -> float:
    """Compute the L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _cosine(
    a: List[float],
    b: List[float],
    a_norm: Optional[float] = None,
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a:      First vector.
        b:      Second vector.
        a_norm: Pre-computed L2 norm of ``a`` (optional optimisation).

    Returns:
        Cosine similarity in range [-1, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = a_norm if a_norm is not None else _l2_norm(a)
    norm_b = _l2_norm(b)
    denom  = norm_a * norm_b
    return dot / denom if denom != 0 else 0.0
