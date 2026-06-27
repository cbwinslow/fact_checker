"""models.py - Pydantic domain models shared across the fact-checker pipeline.

This module defines the full type hierarchy used throughout every pipeline
stage.  All models use Pydantic v2 for validation, serialisation, and
JSON schema generation.

Model groups:
  - Job / status enums        : JobStatus, Verdict, IngestSource
  - Vision enums              : ImageSourceType, ManipulationRisk
  - Core pipeline models      : VideoJob, TranscriptSegment, Claim,
                                EvidenceItem, VerdictResult
  - Image analysis models     : ImageMetadata, DetectedObject, ImageAnalysis
  - Embedding / vector models : EmbeddedChunk
  - Research models           : ResearchResult
  - Context packet            : AnalysisContext
  - Top-level result          : PipelineResult
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ===========================================================================
# Enums
# ===========================================================================

class JobStatus(str, Enum):
    """Lifecycle states of a fact-checking pipeline job."""
    PENDING      = "pending"
    INGESTING    = "ingesting"
    TRANSCRIBING = "transcribing"
    ANALYZING    = "analyzing"     # image / frame analysis stage
    EMBEDDING    = "embedding"     # vectorisation stage
    EXTRACTING   = "extracting"    # claim extraction stage
    RESEARCHING  = "researching"   # deep research stage
    RETRIEVING   = "retrieving"    # evidence retrieval stage
    VERDICTING   = "verdicting"    # verdict drafting stage
    REVIEW       = "review"        # requires human review
    DONE         = "done"
    FAILED       = "failed"


class Verdict(str, Enum):
    """Possible verdict labels for a fact-checked claim."""
    SUPPORTED    = "supported"
    REFUTED      = "refuted"
    INSUFFICIENT = "insufficient_evidence"
    MISLEADING   = "misleading"
    UNVERIFIABLE = "unverifiable"


class IngestSource(str, Enum):
    """The ingest pathway that produced a set of transcript segments."""
    YOUTUBE_CAPTIONS       = "youtube_captions"
    YOUTUBE_TRANSCRIPT_API = "youtube_transcript_api"
    WHISPER_ASR            = "whisper_asr"
    IMAGE                  = "image"        # static image input
    SCREENSHOT             = "screenshot"   # screenshot / web capture
    TEXT_FILE              = "text_file"    # plain text / markdown / CSV
    DOCUMENT               = "document"     # PDF / DOCX
    WEB_ARTICLE            = "web_article"  # scraped web page


class ImageSourceType(str, Enum):
    """Where an image originated from."""
    VIDEO_FRAME = "video_frame"   # extracted from video at a timestamp
    UPLOAD      = "upload"        # user-uploaded image file
    URL         = "url"           # fetched from a remote URL
    SCREENSHOT  = "screenshot"    # full-page or element screenshot


class ManipulationRisk(str, Enum):
    """Rough signal of how likely an image has been digitally manipulated."""
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    UNKNOWN = "unknown"


# ===========================================================================
# Core pipeline models
# ===========================================================================

class VideoJob(BaseModel):
    """Top-level record representing one fact-checking job."""
    id:            UUID              = Field(default_factory=uuid4)
    url:           Optional[str]     = None
    local_path:    Optional[str]     = None
    status:        JobStatus         = JobStatus.PENDING
    ingest_source: Optional[IngestSource] = None
    created_at:    datetime          = Field(default_factory=datetime.utcnow)
    updated_at:    datetime          = Field(default_factory=datetime.utcnow)
    error:         Optional[str]     = None


class TranscriptSegment(BaseModel):
    """One timestamped segment of transcribed or extracted text."""
    id:        UUID           = Field(default_factory=uuid4)
    job_id:    UUID
    start_sec: float
    end_sec:   float
    text:      str
    speaker:   Optional[str] = None


class Claim(BaseModel):
    """A single atomic, verifiable factual claim extracted from content."""
    id:           UUID           = Field(default_factory=uuid4)
    job_id:       UUID
    segment_id:   Optional[UUID] = None
    image_id:     Optional[UUID] = None   # link claim to source image
    text:         str
    is_checkable: bool           = True
    confidence:   float          = 1.0
    context:      Optional[str]  = None


class EvidenceItem(BaseModel):
    """A single piece of evidence retrieved for a claim."""
    id:                  UUID          = Field(default_factory=uuid4)
    claim_id:            UUID
    source_url:          str
    title:               Optional[str] = None
    snippet:             str
    relevance_score:     float         = 0.0
    credibility_score:   float         = 0.5   # domain-reputation score 0-1
    is_factcheck_source: bool          = False


class VerdictResult(BaseModel):
    """The verdict and explanation for one claim."""
    id:                    UUID            = Field(default_factory=uuid4)
    claim_id:              UUID
    verdict:               Verdict
    explanation:           str
    confidence:            float
    evidence_ids:          List[UUID]      = Field(default_factory=list)
    requires_human_review: bool            = False
    reviewed_by:           Optional[str]   = None
    reviewed_at:           Optional[datetime] = None


# ===========================================================================
# Image analysis models
# ===========================================================================

class ImageMetadata(BaseModel):
    """EXIF and file-level metadata extracted from an image."""
    width:               Optional[int]   = None
    height:              Optional[int]   = None
    format:              Optional[str]   = None   # e.g. "JPEG", "PNG"
    mode:                Optional[str]   = None   # e.g. "RGB", "RGBA"
    file_size_bytes:     Optional[int]   = None
    camera_make:         Optional[str]   = None
    camera_model:        Optional[str]   = None
    datetime_original:   Optional[str]   = None
    gps_latitude:        Optional[float] = None
    gps_longitude:       Optional[float] = None
    software:            Optional[str]   = None   # editing software EXIF tag
    frame_timestamp_sec: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)  # raw EXIF overflow


class DetectedObject(BaseModel):
    """A single object or text element detected in an image."""
    label:        str
    confidence:   float                       = 0.0
    bounding_box: Optional[Dict[str, float]]  = None  # {x, y, w, h} normalised 0-1
    text_content: Optional[str]               = None  # OCR-detected text content


class ImageAnalysis(BaseModel):
    """Full vision LLM analysis result for one image or video frame."""
    id:                   UUID                = Field(default_factory=uuid4)
    job_id:               UUID
    source_type:          ImageSourceType
    source_path:          str                 # local file path or URL
    frame_sec:            Optional[float]     = None  # video frame timestamp

    metadata:             ImageMetadata       = Field(default_factory=ImageMetadata)

    description:          str                 = ""    # natural-language scene description
    objects:              List[DetectedObject] = Field(default_factory=list)
    text_in_image:        str                 = ""    # OCR / on-screen text
    visible_claims:       List[str]           = Field(default_factory=list)
    context_notes:        str                 = ""    # analyst notes

    manipulation_risk:    ManipulationRisk    = ManipulationRisk.UNKNOWN
    manipulation_reason:  str                 = ""

    analyzed_at:          datetime            = Field(default_factory=datetime.utcnow)


# ===========================================================================
# Embedding / vector models
# ===========================================================================

class EmbeddedChunk(BaseModel):
    """A text chunk with its dense vector embedding.

    Produced by :mod:`services.embedder` and stored in the VectorStore for
    semantic retrieval by the DeepResearchAgent.
    """
    id:               UUID         = Field(default_factory=uuid4)
    job_id:           UUID
    text:             str          # the original text chunk
    vector:           List[float]  = Field(default_factory=list)  # embedding vector
    chunk_index:      int          = 0       # position in the document
    source_hash:      str          = ""      # SHA-256 of the text for dedup
    similarity_score: float        = 0.0     # populated during retrieval


# ===========================================================================
# Research models
# ===========================================================================

class ResearchResult(BaseModel):
    """The packaged output of the DeepResearchAgent for one claim.

    Bundles the evidence list together with semantic context snippets
    and aggregate quality metrics so the VerdictAgent has a complete
    picture for its reasoning.
    """
    claim_id:            UUID
    evidence:            List[EvidenceItem] = Field(default_factory=list)
    context_snippets:    List[str]          = Field(default_factory=list)
    avg_credibility:     float              = 0.0
    evidence_count:      int                = 0
    has_factcheck_source: bool              = False


# ===========================================================================
# AnalysisContext - typed context packet passed between pipeline stages
# ===========================================================================

class AnalysisContext(BaseModel):
    """Structured context packet that flows through the entire pipeline.

    Replaces the ad-hoc raw Python lists previously passed between stages.
    Bundles all intermediate artefacts in one typed object so every agent
    has full visibility into the job's state and can make informed decisions.

    The ``vector_store`` field holds a live VectorStore instance and is
    excluded from Pydantic serialisation (``exclude=True``) because it
    contains a non-serialisable ChromaDB client object.
    """
    model_config = {"arbitrary_types_allowed": True}

    job_id:     UUID

    # Ingest outputs
    segments:    List[TranscriptSegment] = Field(default_factory=list)
    ingest_source: Optional[IngestSource] = None

    # Vision outputs
    images:      List[ImageAnalysis]     = Field(default_factory=list)

    # Embedding outputs
    chunks:      List[EmbeddedChunk]     = Field(default_factory=list)
    vector_store: Optional[Any]          = Field(default=None, exclude=True)

    # Claim extraction outputs
    claims:      List[Claim]             = Field(default_factory=list)

    # Research outputs
    research_results: List[ResearchResult] = Field(default_factory=list)

    # Evidence + verdict outputs
    evidence:    List[EvidenceItem]      = Field(default_factory=list)
    verdicts:    List[VerdictResult]     = Field(default_factory=list)


# ===========================================================================
# Top-level pipeline result
# ===========================================================================

class PipelineResult(BaseModel):
    """Top-level result object returned by the pipeline harness.

    Aggregates all stage outputs for the API response and DB persistence.
    """
    job:      VideoJob
    segments: List[TranscriptSegment] = Field(default_factory=list)
    images:   List[ImageAnalysis]     = Field(default_factory=list)
    chunks:   List[EmbeddedChunk]     = Field(default_factory=list)
    claims:   List[Claim]             = Field(default_factory=list)
    research: List[ResearchResult]    = Field(default_factory=list)
    evidence: List[EvidenceItem]      = Field(default_factory=list)
    verdicts: List[VerdictResult]     = Field(default_factory=list)
