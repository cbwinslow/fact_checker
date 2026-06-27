"""Pydantic domain models shared across the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING      = "pending"
    INGESTING    = "ingesting"
    TRANSCRIBING = "transcribing"
    ANALYZING    = "analyzing"   # NEW: image/frame analysis stage
    EXTRACTING   = "extracting"
    RETRIEVING   = "retrieving"
    VERDICTING   = "verdicting"
    REVIEW       = "review"
    DONE         = "done"
    FAILED       = "failed"


class Verdict(str, Enum):
    SUPPORTED   = "supported"
    REFUTED     = "refuted"
    INSUFFICIENT = "insufficient_evidence"
    MISLEADING  = "misleading"
    UNVERIFIABLE = "unverifiable"


class IngestSource(str, Enum):
    YOUTUBE_CAPTIONS      = "youtube_captions"
    YOUTUBE_TRANSCRIPT_API = "youtube_transcript_api"
    WHISPER_ASR           = "whisper_asr"
    IMAGE                 = "image"       # NEW: static image input
    SCREENSHOT            = "screenshot"  # NEW: screenshot / web capture


# ---------------------------------------------------------------------------
# NEW: Image / vision enums
# ---------------------------------------------------------------------------

class ImageSourceType(str, Enum):
    """Where the image came from."""
    VIDEO_FRAME  = "video_frame"   # extracted from video at timestamp
    UPLOAD       = "upload"        # user-uploaded image
    URL          = "url"           # fetched from a web URL
    SCREENSHOT   = "screenshot"   # full-page or element screenshot


class ManipulationRisk(str, Enum):
    """Rough signal of how likely an image has been manipulated."""
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class VideoJob(BaseModel):
    id:            UUID             = Field(default_factory=uuid4)
    url:           Optional[str]    = None
    local_path:    Optional[str]    = None
    status:        JobStatus        = JobStatus.PENDING
    ingest_source: Optional[IngestSource] = None
    created_at:    datetime         = Field(default_factory=datetime.utcnow)
    updated_at:    datetime         = Field(default_factory=datetime.utcnow)
    error:         Optional[str]    = None


class TranscriptSegment(BaseModel):
    id:        UUID            = Field(default_factory=uuid4)
    job_id:    UUID
    start_sec: float
    end_sec:   float
    text:      str
    speaker:   Optional[str]  = None


class Claim(BaseModel):
    id:           UUID           = Field(default_factory=uuid4)
    job_id:       UUID
    segment_id:   Optional[UUID] = None
    image_id:     Optional[UUID] = None   # NEW: link claim to source image
    text:         str
    is_checkable: bool           = True
    confidence:   float          = 1.0
    context:      Optional[str]  = None


class EvidenceItem(BaseModel):
    id:                  UUID           = Field(default_factory=uuid4)
    claim_id:            UUID
    source_url:          str
    title:               Optional[str]  = None
    snippet:             str
    relevance_score:     float          = 0.0
    is_factcheck_source: bool           = False


class VerdictResult(BaseModel):
    id:                  UUID           = Field(default_factory=uuid4)
    claim_id:            UUID
    verdict:             Verdict
    explanation:         str
    confidence:          float
    evidence_ids:        List[UUID]     = Field(default_factory=list)
    requires_human_review: bool         = False
    reviewed_by:         Optional[str]  = None
    reviewed_at:         Optional[datetime] = None


# ---------------------------------------------------------------------------
# NEW: Image analysis models
# ---------------------------------------------------------------------------

class ImageMetadata(BaseModel):
    """EXIF / file-level metadata extracted from an image."""
    width:          Optional[int]   = None
    height:         Optional[int]   = None
    format:         Optional[str]   = None   # e.g. "JPEG", "PNG"
    mode:           Optional[str]   = None   # e.g. "RGB", "RGBA"
    file_size_bytes: Optional[int]  = None
    # EXIF fields (populated when available)
    camera_make:    Optional[str]   = None
    camera_model:   Optional[str]   = None
    datetime_original: Optional[str] = None
    gps_latitude:   Optional[float] = None
    gps_longitude:  Optional[float] = None
    software:       Optional[str]   = None   # editing software tag
    # Video-frame context
    frame_timestamp_sec: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)  # raw EXIF overflow


class DetectedObject(BaseModel):
    """A single object or text element detected in an image."""
    label:       str
    confidence:  float = 0.0
    bounding_box: Optional[Dict[str, float]] = None  # {x, y, w, h} normalised 0-1
    text_content: Optional[str] = None  # for OCR-detected text regions


class ImageAnalysis(BaseModel):
    """Full vision analysis result for one image / frame."""
    id:              UUID               = Field(default_factory=uuid4)
    job_id:          UUID
    source_type:     ImageSourceType
    source_path:     str                # local file path or URL
    frame_sec:       Optional[float]    = None  # for video frames

    # Metadata
    metadata:        ImageMetadata      = Field(default_factory=ImageMetadata)

    # Vision LLM outputs
    description:     str                = ""    # natural-language scene description
    objects:         List[DetectedObject] = Field(default_factory=list)
    text_in_image:   str                = ""    # OCR / on-screen text
    visible_claims:  List[str]          = Field(default_factory=list)  # claims visible in frame
    context_notes:   str                = ""    # analyst notes (e.g. "edited watermark")

    # Manipulation signal
    manipulation_risk:    ManipulationRisk = ManipulationRisk.UNKNOWN
    manipulation_reason:  str              = ""

    # Timestamps
    analyzed_at:     datetime           = Field(default_factory=datetime.utcnow)


class PipelineResult(BaseModel):
    """Top-level result returned by the harness."""
    job:      VideoJob
    segments: List[TranscriptSegment] = Field(default_factory=list)
    images:   List[ImageAnalysis]     = Field(default_factory=list)  # NEW
    claims:   List[Claim]             = Field(default_factory=list)
    evidence: List[EvidenceItem]      = Field(default_factory=list)
    verdicts: List[VerdictResult]     = Field(default_factory=list)
