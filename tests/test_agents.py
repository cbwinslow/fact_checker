"""Unit tests for claim extraction, verdict, and evidence agents."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from fact_checker.models import Claim, Verdict


# ---------------------------------------------------------------------------
# ClaimExtractionAgent tests
# ---------------------------------------------------------------------------

class TestClaimExtraction:
    """Tests for fact_checker.agents.claim_extractor.extract_claims"""

    @pytest.mark.asyncio
    async def test_extract_claims_returns_claims(self, sample_segments, job_id):
        """extract_claims returns a list of Claim objects on valid LLM response."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {
                "text": "The Earth is approximately 4.5 billion years old.",
                "is_checkable": True,
                "confidence": 0.95,
                "context": None,
            }
        ])

        with patch("fact_checker.agents.claim_extractor._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.claim_extractor import extract_claims
            result = await extract_claims(job_id=job_id, segments=sample_segments)

        assert len(result) == 1
        assert isinstance(result[0], Claim)
        assert "4.5 billion" in result[0].text
        assert result[0].job_id == job_id
        assert result[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_extract_claims_empty_segments(self, job_id):
        """extract_claims returns empty list when no segments provided."""
        from fact_checker.agents.claim_extractor import extract_claims
        result = await extract_claims(job_id=job_id, segments=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_claims_handles_bad_json(self, sample_segments, job_id):
        """extract_claims returns empty list when LLM returns invalid JSON."""
        mock_response = MagicMock()
        mock_response.content = "not valid json at all {{{"

        with patch("fact_checker.agents.claim_extractor._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.claim_extractor import extract_claims
            result = await extract_claims(job_id=job_id, segments=sample_segments)

        assert result == []

    @pytest.mark.asyncio
    async def test_extract_claims_skips_malformed_items(self, sample_segments, job_id):
        """extract_claims skips items missing required 'text' key."""
        mock_response = MagicMock()
        mock_response.content = json.dumps([
            {"is_checkable": True},  # missing 'text' key
            {"text": "Valid claim here.", "confidence": 0.8},
        ])

        with patch("fact_checker.agents.claim_extractor._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.claim_extractor import extract_claims
            result = await extract_claims(job_id=job_id, segments=sample_segments)

        assert len(result) == 1
        assert result[0].text == "Valid claim here."


# ---------------------------------------------------------------------------
# VerdictAgent tests
# ---------------------------------------------------------------------------

class TestVerdictAgent:
    """Tests for fact_checker.agents.verdict_agent.draft_verdicts"""

    @pytest.mark.asyncio
    async def test_draft_verdicts_returns_verdict_per_claim(self, sample_claims, sample_evidence):
        """draft_verdicts returns one VerdictResult per Claim."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "supported",
            "explanation": "Confirmed by multiple sources.",
            "confidence": 0.92,
            "requires_human_review": False,
        })

        with patch("fact_checker.agents.verdict_agent._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.verdict_agent import draft_verdicts
            result = await draft_verdicts(claims=sample_claims, evidence=sample_evidence)

        assert len(result) == len(sample_claims)
        assert result[0].verdict == Verdict.SUPPORTED
        assert result[0].confidence == 0.92
        assert result[0].claim_id == sample_claims[0].id

    @pytest.mark.asyncio
    async def test_draft_verdicts_empty_claims(self, sample_evidence):
        """draft_verdicts returns empty list when no claims provided."""
        from fact_checker.agents.verdict_agent import draft_verdicts
        result = await draft_verdicts(claims=[], evidence=sample_evidence)
        assert result == []

    @pytest.mark.asyncio
    async def test_draft_verdicts_handles_llm_failure(self, sample_claims):
        """draft_verdicts returns UNVERIFIABLE + requires_human_review on LLM error."""
        with patch("fact_checker.agents.verdict_agent._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM timeout")
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.verdict_agent import draft_verdicts
            result = await draft_verdicts(claims=sample_claims, evidence=[])

        assert len(result) == len(sample_claims)
        for v in result:
            assert v.verdict == Verdict.UNVERIFIABLE
            assert v.requires_human_review is True
            assert v.confidence == 0.0

    @pytest.mark.asyncio
    async def test_draft_verdicts_low_confidence_flags_review(self, sample_claims):
        """draft_verdicts sets requires_human_review when confidence < 0.5."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "insufficient_evidence",
            "explanation": "Not enough evidence.",
            "confidence": 0.3,
            "requires_human_review": False,
        })

        with patch("fact_checker.agents.verdict_agent._build_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_llm_factory.return_value = mock_llm

            from fact_checker.agents.verdict_agent import draft_verdicts
            result = await draft_verdicts(claims=[sample_claims[0]], evidence=[])

        assert result[0].requires_human_review is True


# ---------------------------------------------------------------------------
# Config / model registry tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for fact_checker.config MODEL_REGISTRY and build_chat_model."""

    def test_model_registry_has_all_tasks(self):
        from fact_checker.config import MODEL_REGISTRY
        expected_tasks = {"extraction", "verification", "orchestration", "multimodal", "tooluse", "fast"}
        assert expected_tasks == set(MODEL_REGISTRY.keys())

    def test_build_chat_model_returns_chat_openai(self):
        from fact_checker.config import build_chat_model
        from langchain_openai import ChatOpenAI
        with patch("fact_checker.config.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.openrouter_api_key = "test-key"
            mock_s.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_s.openrouter_model = "openai/gpt-oss-120b:free"
            mock_s.model_extraction = None
            mock_settings.return_value = mock_s
            model = build_chat_model(task="extraction", temperature=0)
        assert isinstance(model, ChatOpenAI)

    def test_build_chat_model_env_override(self):
        from fact_checker.config import build_chat_model
        with patch("fact_checker.config.get_settings") as mock_settings:
            mock_s = MagicMock()
            mock_s.openrouter_api_key = "test-key"
            mock_s.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_s.openrouter_model = "openai/gpt-oss-120b:free"
            mock_s.model_extraction = "custom/model:free"
            mock_settings.return_value = mock_s
            model = build_chat_model(task="extraction")
        assert model.model_name == "custom/model:free"
