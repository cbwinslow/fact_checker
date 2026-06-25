from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from fact_checker.domain.models import Claim, Verdict
from fact_checker.extractors.claim_extraction import ClaimExtractionAgent
from fact_checker.utils.openrouter import build_chat_model

_VERIFICATION_SYSTEM = """
You are a rigorous fact-verification engine. Given a single factual claim,
determine whether it is TRUE, FALSE, MISLEADING, or UNVERIFIABLE.

Return a JSON object with exactly these keys:
  - label:           one of ["TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"]
  - confidence:      float 0.0-1.0
  - rationale:       1-3 sentence explanation of your verdict
  - evidence_ids:    list of strings (empty list if none)
  - review_required: true if human review is recommended, else false

Output ONLY valid JSON. No markdown, no extra text.
"""

_ORCHESTRATION_SYSTEM = """
You are a pipeline orchestrator for a video fact-checking system.
Given a list of claim verification results, produce a concise executive summary.

Return a JSON object with:
  - total_claims:    integer
  - true_count:      integer
  - false_count:     integer
  - misleading_count: integer
  - unverifiable_count: integer
  - overall_credibility: float 0.0-1.0 (1.0 = fully credible)
  - summary:         2-4 sentence narrative summary
  - flags:           list of strings describing the most serious issues found

Output ONLY valid JSON.
"""


class FactCheckHarness:
    """Multi-model orchestration harness for the video fact-checking pipeline.

    Model assignments:
      - Claim extraction:   openai/gpt-oss-120b:free   (structured JSON output)
      - Claim verification: nvidia/nemotron-3-ultra:free (deep reasoning, 1M ctx)
      - Final summary:      nvidia/nemotron-3-super:free (agent orchestration)
    """

    def __init__(self) -> None:
        self.extractor = ClaimExtractionAgent()
        # Verification model - deep reasoning over individual claims
        self.verifier = build_chat_model(task="verification", temperature=0)
        # Orchestration model - synthesises results across all claims
        self.orchestrator = build_chat_model(task="orchestration", temperature=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source: str) -> dict[str, Any]:
        """Run the full fact-checking pipeline on a transcript or text source.

        Args:
            source: Raw transcript text (or URL/path to be resolved upstream).

        Returns:
            A dict containing source, status, claims, verdicts, and summary.
        """
        video_id = str(uuid.uuid4())

        # --- Stage 1: Claim Extraction (gpt-oss-120b) ---
        claims: list[Claim] = self.extractor.extract(
            transcript_chunk=source,
            video_id=video_id,
        )

        if not claims:
            return {
                "source": source,
                "video_id": video_id,
                "status": "no_claims_found",
                "claims": [],
                "verdicts": [],
                "summary": None,
            }

        # --- Stage 2: Claim Verification (nemotron-3-ultra) ---
        verdicts: list[Verdict] = []
        for claim in claims:
            verdict = self._verify_claim(claim)
            verdicts.append(verdict)

        # --- Stage 3: Pipeline Orchestration / Summary (nemotron-3-super) ---
        summary = self._synthesise(claims, verdicts)

        return {
            "source": source,
            "video_id": video_id,
            "status": "complete",
            "claims": [c.model_dump() for c in claims],
            "verdicts": [v.model_dump() for v in verdicts],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_claim(self, claim: Claim) -> Verdict:
        """Verify a single claim using nemotron-3-ultra (1M context window)."""
        messages = [
            SystemMessage(content=_VERIFICATION_SYSTEM),
            HumanMessage(content=f"CLAIM:\n{claim.text}"),
        ]
        try:
            response = self.verifier.invoke(messages)
            data: dict[str, Any] = json.loads(response.content)
            return Verdict(
                claim_id=claim.claim_id,
                label=data.get("label", "UNVERIFIABLE"),
                confidence=float(data.get("confidence", 0.0)),
                rationale=data.get("rationale", ""),
                evidence_ids=data.get("evidence_ids", []),
                review_required=bool(data.get("review_required", True)),
            )
        except Exception:
            return Verdict(
                claim_id=claim.claim_id,
                label="UNVERIFIABLE",
                confidence=0.0,
                rationale="Verification failed due to an internal error.",
                evidence_ids=[],
                review_required=True,
            )

    def _synthesise(
        self,
        claims: list[Claim],
        verdicts: list[Verdict],
    ) -> dict[str, Any]:
        """Produce an executive summary using nemotron-3-super (orchestration)."""
        payload = [
            {"claim": c.text, "verdict": v.label, "rationale": v.rationale}
            for c, v in zip(claims, verdicts)
        ]
        messages = [
            SystemMessage(content=_ORCHESTRATION_SYSTEM),
            HumanMessage(content=f"RESULTS:\n{json.dumps(payload, indent=2)}"),
        ]
        try:
            response = self.orchestrator.invoke(messages)
            return json.loads(response.content)
        except Exception:
            return {
                "total_claims": len(claims),
                "summary": "Summary generation failed.",
                "flags": [],
            }
