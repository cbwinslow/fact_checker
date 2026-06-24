"""Pydantic domain models shared across the pipeline."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    TRANSCRIBING = "transcribing"
    EXTRACTING = "extracting"
    RETRIEVING = "retrieving"
    VERDICTING = "verdicting"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient_evidence"
    MISLEADING = "misleading"
    UNVERIFIABLE = "unverifiable"


class IngestSource(str, Enum):
    YOUTUBE_CAPTIONS = "youtube_captions"
    YOUTUBE_TRANSCRIPT_API = "youtube_transcript_api"
    WHISPER_ASR = "whisper_asr"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class VideoJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: Optional[str] = None
    local_path: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    ingest_source: Optional[IngestSource] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class TranscriptSegment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    start_sec: float
    end_sec: float
    text: str
    speaker: Optional[str] = None


class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    segment_id: Optional[UUID] = None
    text: str
    is_checkable: bool = True
    confidence: float = 1.0  # model confidence that this is a real claim
    context: Optional[str] = None  # surrounding text for grounding


class EvidenceItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    source_url: str
    title: Optional[str] = None
    snippet: str
    relevance_score: float = 0.0
    is_factcheck_source: bool = False


class VerdictResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    verdict: Verdict
    explanation: str
    confidence: float
    evidence_ids: list[UUID] = Field(default_factory=list)
    requires_human_review: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class PipelineResult(BaseModel):
    """Top-level result returned by the harness."""
    job: VideoJob
    segments: list[TranscriptSegment] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    verdicts: list[VerdictResult] = Field(default_factory=list)
