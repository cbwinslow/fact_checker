from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from fact_checker.domain.models import Claim
from fact_checker.utils.openrouter import build_chat_model

_SYSTEM_PROMPT = """
You are a precise claim-extraction engine for a video fact-checking pipeline.
Given a transcript chunk, extract every distinct, verifiable factual claim.

Return a JSON array of objects. Each object must have these keys:
  - claim_id:   a unique UUID v4 string
  - text:       the verbatim or near-verbatim claim text
  - claim_type: one of ["statistic", "historical", "scientific", "political", "other"]
  - confidence: float 0.0-1.0 representing how verifiable this claim appears
  - speaker:    speaker name if identifiable, else null
  - start_ts:   approximate start timestamp in seconds (0.0 if unknown)
  - end_ts:     approximate end timestamp in seconds (0.0 if unknown)

Output ONLY valid JSON. No markdown, no explanation.
"""


class ClaimExtractionAgent:
    """Extracts verifiable claims from transcript chunks using gpt-oss-120b.

    This model is chosen for its strong structured-output and function-calling
    capabilities, making it ideal for producing typed JSON claim objects.
    """

    def __init__(self) -> None:
        # Use the 'extraction' task slot -> openai/gpt-oss-120b:free
        self.model = build_chat_model(task="extraction", temperature=0)

    def extract(self, transcript_chunk: str, video_id: str = "") -> list[Claim]:
        """Extract claims from a single transcript chunk.

        Args:
            transcript_chunk: Raw transcript text to analyse.
            video_id: Optional video identifier to populate on each Claim.

        Returns:
            List of Claim domain objects (empty list on parse failure).
        """
        if not transcript_chunk.strip():
            return []

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"TRANSCRIPT:\n{transcript_chunk}"),
        ]

        try:
            response = self.model.invoke(messages)
            raw: list[dict[str, Any]] = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError, Exception):
            return []

        claims: list[Claim] = []
        for item in raw:
            try:
                claims.append(
                    Claim(
                        claim_id=item.get("claim_id", str(uuid.uuid4())),
                        video_id=video_id,
                        text=item["text"],
                        start_ts=float(item.get("start_ts", 0.0)),
                        end_ts=float(item.get("end_ts", 0.0)),
                        speaker=item.get("speaker"),
                        claim_type=item.get("claim_type", "other"),
                        confidence=float(item.get("confidence", 0.5)),
                        status="extracted",
                    )
                )
            except (KeyError, TypeError):
                continue

        return claims
