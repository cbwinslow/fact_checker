"""Async PostgreSQL connection pool via asyncpg + SQLAlchemy ORM table definitions."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Numeric, String, Text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from .config import settings

log = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Table Definitions
# ---------------------------------------------------------------------------

class VideoJobRow(Base):
    __tablename__ = "video_jobs"

    id            = Column(PGUUID(as_uuid=True), primary_key=True)
    url           = Column(Text, nullable=True)
    local_path    = Column(Text, nullable=True)
    status        = Column(String(32), default="pending", nullable=False)
    ingest_source = Column(String(64), nullable=True)
    error         = Column(Text, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    segments = relationship("TranscriptSegmentRow", back_populates="job", cascade="all, delete-orphan")
    claims   = relationship("ClaimRow",             back_populates="job", cascade="all, delete-orphan")
    verdicts = relationship("VerdictRow",           back_populates="job", cascade="all, delete-orphan")


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"

    id        = Column(PGUUID(as_uuid=True), primary_key=True)
    job_id    = Column(PGUUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), nullable=False)
    start_sec = Column(Numeric(10, 3), nullable=True)
    end_sec   = Column(Numeric(10, 3), nullable=True)
    text      = Column(Text, nullable=False)
    speaker   = Column(Text, nullable=True)

    job = relationship("VideoJobRow", back_populates="segments")


class ClaimRow(Base):
    __tablename__ = "claims"

    id           = Column(PGUUID(as_uuid=True), primary_key=True)
    job_id       = Column(PGUUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), nullable=False)
    segment_id   = Column(PGUUID(as_uuid=True), nullable=True)
    text         = Column(Text, nullable=False)
    is_checkable = Column(Boolean, default=True)
    confidence   = Column(Float, default=1.0)
    context      = Column(Text, nullable=True)

    job            = relationship("VideoJobRow",     back_populates="claims")
    verdicts       = relationship("VerdictRow",       back_populates="claim",    cascade="all, delete-orphan")
    evidence_items = relationship("EvidenceItemRow", back_populates="claim",    cascade="all, delete-orphan")


class EvidenceItemRow(Base):
    __tablename__ = "evidence_items"

    id                  = Column(PGUUID(as_uuid=True), primary_key=True)
    claim_id            = Column(PGUUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    source_url          = Column(Text, nullable=False)
    title               = Column(Text, nullable=True)
    snippet             = Column(Text, nullable=True)
    relevance_score     = Column(Numeric(4, 3), default=0.0)
    is_factcheck_source = Column(Boolean, default=False)
    created_at          = Column(DateTime, default=datetime.utcnow)

    claim = relationship("ClaimRow", back_populates="evidence_items")


class VerdictRow(Base):
    __tablename__ = "verdicts"

    id                    = Column(PGUUID(as_uuid=True), primary_key=True)
    job_id                = Column(PGUUID(as_uuid=True), ForeignKey("video_jobs.id", ondelete="CASCADE"), nullable=False)
    claim_id              = Column(PGUUID(as_uuid=True), ForeignKey("claims.id",     ondelete="CASCADE"), nullable=False)
    verdict               = Column(String(32), nullable=False)
    explanation           = Column(Text, nullable=False)
    confidence            = Column(Float, default=0.0)
    requires_human_review = Column(Boolean, default=False)
    reviewed_by           = Column(String(128), nullable=True)
    reviewed_at           = Column(DateTime, nullable=True)

    job   = relationship("VideoJobRow", back_populates="verdicts")
    claim = relationship("ClaimRow",    back_populates="verdicts")


# ---------------------------------------------------------------------------
# Session dependency (FastAPI / general use)
# ---------------------------------------------------------------------------

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a transactional async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables (dev / test only — use Alembic migrations in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("[db] Database tables initialised.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

async def save_pipeline_result(session: AsyncSession, result) -> None:
    """Persist a full PipelineResult (job + segments + claims + evidence + verdicts)."""
    job = result.job
    session.add(VideoJobRow(
        id=job.id,
        url=job.url,
        local_path=job.local_path,
        status=job.status.value,
        ingest_source=job.ingest_source.value if job.ingest_source else None,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    ))

    for seg in result.segments:
        session.add(TranscriptSegmentRow(
            id=seg.id,
            job_id=job.id,
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            text=seg.text,
            speaker=seg.speaker,
        ))

    for claim in result.claims:
        session.add(ClaimRow(
            id=claim.id,
            job_id=job.id,
            segment_id=claim.segment_id,
            text=claim.text,
            is_checkable=claim.is_checkable,
            confidence=claim.confidence,
            context=claim.context,
        ))

    for ev in result.evidence:
        session.add(EvidenceItemRow(
            id=ev.id,
            claim_id=ev.claim_id,
            source_url=ev.source_url,
            title=ev.title,
            snippet=ev.snippet,
            relevance_score=ev.relevance_score,
            is_factcheck_source=ev.is_factcheck_source,
        ))

    for verdict in result.verdicts:
        session.add(VerdictRow(
            id=verdict.id,
            job_id=job.id,
            claim_id=verdict.claim_id,
            verdict=verdict.verdict.value,
            explanation=verdict.explanation,
            confidence=verdict.confidence,
            requires_human_review=verdict.requires_human_review,
        ))

    await session.flush()
    log.info("[db] Persisted pipeline result for job %s (%d segments, %d claims, %d evidence, %d verdicts)",
             job.id, len(result.segments), len(result.claims), len(result.evidence), len(result.verdicts))


async def get_job_row(session: AsyncSession, job_id) -> VideoJobRow | None:
    """Fetch a VideoJobRow by UUID. Returns None if not found."""
    return await session.get(VideoJobRow, job_id)
