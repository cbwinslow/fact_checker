from pydantic import BaseModel


class Claim(BaseModel):
    claim_id: str
    video_id: str
    text: str
    start_ts: float
    end_ts: float
    speaker: str | None = None
    claim_type: str
    confidence: float
    status: str = "extracted"


class Verdict(BaseModel):
    claim_id: str
    label: str
    confidence: float
    rationale: str
    evidence_ids: list[str]
    review_required: bool = False
