"""VerdictDraftAgent - generates evidence-backed verdicts for each claim."""
from __future__ import annotations
import json
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..models import Claim, EvidenceItem, Verdict, VerdictResult

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "verdict_draft.md"

LOW_CONFIDENCE_THRESHOLD = 0.5


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=2048,
    )


def _evidence_for_claim(claim: Claim, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return [e for e in evidence if e.claim_id == claim.id]


async def draft_verdicts(
    claims: list[Claim],
    evidence: list[EvidenceItem],
) -> list[VerdictResult]:
    """For each claim, build a verdict using LLM + evidence snippets."""
    llm = _build_llm()
    system_prompt = _load_prompt()
    results: list[VerdictResult] = []

    for claim in claims:
        if not claim.is_checkable:
            continue
        claim_evidence = _evidence_for_claim(claim, evidence)
        evidence_block = "\n".join(
            f"- [{e.title or 'Source'}] {e.snippet} ({e.source_url})"
            for e in claim_evidence[:6]
        ) or "No evidence retrieved."

        user_msg = (
            f"CLAIM: {claim.text}\n\n"
            f"CONTEXT: {claim.context or 'N/A'}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Return a JSON object with keys: verdict, explanation, confidence (0-1), requires_human_review."
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)]
        log.info("[VerdictAgent] Verdicting: %s", claim.text[:80])
        response = await llm.ainvoke(messages)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
            verdict_str = data.get("verdict", "insufficient_evidence").lower().replace(" ", "_")
            try:
                verdict = Verdict(verdict_str)
            except ValueError:
                verdict = Verdict.INSUFFICIENT
            confidence = float(data.get("confidence", 0.5))
            requires_review = data.get("requires_human_review", confidence < LOW_CONFIDENCE_THRESHOLD)
            result = VerdictResult(
                claim_id=claim.id,
                verdict=verdict,
                explanation=data.get("explanation", ""),
                confidence=confidence,
                evidence_ids=[e.id for e in claim_evidence],
                requires_human_review=bool(requires_review),
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("[VerdictAgent] Parse error: %s | raw=%s", e, raw[:200])
            result = VerdictResult(
                claim_id=claim.id,
                verdict=Verdict.INSUFFICIENT,
                explanation="Failed to parse model output.",
                confidence=0.0,
                requires_human_review=True,
            )
        results.append(result)

    log.info("[VerdictAgent] Generated %d verdicts", len(results))
    return results
