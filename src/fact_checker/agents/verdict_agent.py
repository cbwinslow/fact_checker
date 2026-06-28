"""VerdictDraftAgent - generates evidence-backed verdicts for each claim."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import build_chat_model
from ..models import Claim, Citation, EvidenceItem, Verdict, VerdictResult

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "verdict_draft.md"

LOW_CONFIDENCE_THRESHOLD = 0.5

_FALLBACK_PROMPT = """You are an expert fact-checker. Given a CLAIM and a list of EVIDENCE items, produce a JSON verdict object.

Respond with ONLY a JSON object with these keys:
- verdict: one of "supported", "refuted", "insufficient_evidence", "misleading", "unverifiable"
- confidence: float 0.0-1.0
- explanation: concise explanation citing evidence (2-4 sentences) with [doc_id] markers
- requires_human_review: boolean
- citations: array of objects with evidence_id, quote, claim_fragment

Only return the JSON object, no other text."""


def _load_prompt() -> str:
    """Load prompt from file, fall back to inline default if missing."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning(
            "[verdict_agent] Prompt file not found at %s, using fallback", PROMPT_PATH
        )
        return _FALLBACK_PROMPT


def _build_llm() -> BaseChatModel:
    """Use verification task slot -> nvidia/nemotron-3-ultra:free for deep reasoning."""
    return build_chat_model(task="verification", temperature=0.0, max_tokens=2048)


def _extract_json(content: str) -> dict:
    """Extract JSON from LLM response, handling markdown code fences."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(content)


async def draft_verdicts(
    claims: list[Claim],
    evidence: list[EvidenceItem],
) -> list[VerdictResult]:
    """Generate a verdict for each claim grounded in retrieved evidence.

    Args:
        claims: List of Claim objects to verdict.
        evidence: All EvidenceItem objects from the research stage.

    Returns:
        List of VerdictResult objects, one per claim.
    """
    if not claims:
        return []

    llm = _build_llm()
    system_prompt = _load_prompt()

    # Build evidence map by claim_id
    evidence_by_claim: dict = {}
    for ev in evidence:
        evidence_by_claim.setdefault(str(ev.claim_id), []).append(ev)

    verdicts: list[VerdictResult] = []
    for claim in claims:
        ev_items = evidence_by_claim.get(str(claim.id), [])

        # Format evidence with doc_ids for citation
        evidence_lines = []
        for i, ev in enumerate(ev_items):
            doc_id = i + 1
            source = ev.title or ev.domain or "Source"
            quote = f' Quote: "{ev.quote_text}"' if ev.quote_text else ""
            evidence_lines.append(
                f"[doc_{doc_id}] [{source}]({ev.source_url}): {ev.snippet}{quote}"
            )
        evidence_text = "\n".join(evidence_lines) or "No evidence retrieved."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"CLAIM:\n{claim.text}\n\n"
                    f"EVIDENCE:\n{evidence_text}"
                )
            ),
        ]

        try:
            response = llm.invoke(messages)
            data = _extract_json(response.content)

            verdict_label = Verdict(data.get("verdict", "unverifiable"))
            confidence = float(data.get("confidence", 0.0))

            # Parse citations from LLM response
            citations = []
            for cite_data in data.get("citations", []):
                try:
                    citations.append(Citation(
                        evidence_id=UUID(cite_data.get("evidence_id", "")),
                        quote=cite_data.get("quote", ""),
                        claim_fragment=cite_data.get("claim_fragment", ""),
                    ))
                except Exception as exc:
                    log.warning("[verdict_agent] Failed to parse citation: %s", exc)

            verdicts.append(
                VerdictResult(
                    claim_id=claim.id,
                    verdict=verdict_label,
                    explanation=data.get("explanation", ""),
                    confidence=confidence,
                    evidence_ids=[ev.id for ev in ev_items],
                    citations=citations,
                    requires_human_review=(
                        data.get("requires_human_review", False)
                        or confidence < LOW_CONFIDENCE_THRESHOLD
                    ),
                )
            )
        except Exception as exc:
            log.error(
                "[verdict_agent] Failed to draft verdict for claim %s: %s",
                claim.id, exc,
            )
            verdicts.append(
                VerdictResult(
                    claim_id=claim.id,
                    verdict=Verdict.UNVERIFIABLE,
                    explanation="Verdict generation failed due to an internal error.",
                    confidence=0.0,
                    evidence_ids=[],
                    citations=[],
                    requires_human_review=True,
                )
            )

    log.info("[verdict_agent] Drafted %d verdicts", len(verdicts))
    return verdicts