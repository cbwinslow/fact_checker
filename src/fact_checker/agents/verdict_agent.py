"""VerdictDraftAgent - generates evidence-backed verdicts for each claim."""
from __future__ import annotations
import json
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import build_chat_model
from ..models import Claim, EvidenceItem, Verdict, VerdictResult

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "verdict_draft.md"

LOW_CONFIDENCE_THRESHOLD = 0.5


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_llm() -> ChatOpenAI:
    """Use verification task slot -> nvidia/nemotron-3-ultra:free for deep reasoning."""
    return build_chat_model(task="verification", temperature=0.0, max_tokens=2048)


async def draft_verdicts(
    claims: list[Claim],
    evidence: list[EvidenceItem],
) -> list[VerdictResult]:
    """Generate a verdict for each claim grounded in retrieved evidence."""
    if not claims:
        return []

    llm = _build_llm()
    system_prompt = _load_prompt()
    evidence_by_claim: dict = {}
    for ev in evidence:
        evidence_by_claim.setdefault(str(ev.claim_id), []).append(ev)

    verdicts: list[VerdictResult] = []
    for claim in claims:
        ev_items = evidence_by_claim.get(str(claim.id), [])
        evidence_text = "\n".join(
            f"- [{ev.title or 'Source'}]({ev.source_url}): {ev.snippet}"
            for ev in ev_items
        ) or "No evidence retrieved."

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
            data: dict = json.loads(response.content)
            verdict_label = Verdict(data.get("verdict", "unverifiable"))
            confidence = float(data.get("confidence", 0.0))
            verdicts.append(
                VerdictResult(
                    claim_id=claim.id,
                    verdict=verdict_label,
                    explanation=data.get("explanation", ""),
                    confidence=confidence,
                    evidence_ids=[ev.id for ev in ev_items],
                    requires_human_review=(
                        data.get("requires_human_review", False)
                        or confidence < LOW_CONFIDENCE_THRESHOLD
                    ),
                )
            )
        except Exception as exc:
            log.error("[verdict_agent] Failed to draft verdict for claim %s: %s", claim.id, exc)
            verdicts.append(
                VerdictResult(
                    claim_id=claim.id,
                    verdict=Verdict.UNVERIFIABLE,
                    explanation="Verdict generation failed due to an internal error.",
                    confidence=0.0,
                    evidence_ids=[],
                    requires_human_review=True,
                )
            )

    log.info("[verdict_agent] Drafted %d verdicts", len(verdicts))
    return verdicts
