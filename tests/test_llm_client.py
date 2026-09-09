"""Tests for LLM client and verifier."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from rag_adviser.llm.client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    detect_llm_config,
)
from rag_adviser.llm.verifier import RecommendationVerifier
from rag_adviser.models import (
    Recommendations,
    RetrievalRecommendation,
    UserAnswers,
)


class TestDetectLLMConfig:
    """Tests for auto-detecting API keys from environment."""

    def test_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            assert detect_llm_config() is None

    def test_anthropic_key_preferred(self):
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-oai-test"},
        ):
            config = detect_llm_config()
            assert config is not None
            assert config.provider == LLMProvider.ANTHROPIC
            assert config.api_key == "sk-ant-test"

    def test_openai_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai-test"}, clear=True):
            config = detect_llm_config()
            assert config is not None
            assert config.provider == LLMProvider.OPENAI

    def test_empty_key_ignored(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "  "}, clear=True):
            assert detect_llm_config() is None

    def test_custom_model_from_env(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test", "RAGADVISOR_LLM_MODEL": "gpt-4o"},
        ):
            config = detect_llm_config()
            assert config.model == "gpt-4o"


class TestLLMClient:
    """Tests for LLM client HTTP calls."""

    def test_openai_complete(self):
        config = LLMConfig(
            provider=LLMProvider.OPENAI, api_key="test", model="gpt-4o-mini"
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = client.complete("system", "user")
            assert result == "test response"
            mock_post.assert_called_once()

    def test_anthropic_complete(self):
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC, api_key="test", model="claude-opus-5"
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "test response"},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = client.complete("system", "user")
            assert result == "test response"
            # Verify Anthropic-specific headers
            call_kwargs = mock_post.call_args
            assert "x-api-key" in call_kwargs.kwargs["headers"]
            # Current Claude models reject sampling params; none must be sent.
            payload = call_kwargs.kwargs["json"]
            assert "temperature" not in payload
            assert payload["model"] == "claude-opus-5"

    def test_anthropic_refusal_raises(self):
        from rag_adviser.llm.client import LLMError

        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC, api_key="test", model="claude-opus-5"
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"stop_reason": "refusal", "content": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            try:
                client.complete("system", "user")
            except LLMError as e:
                assert "refusal" in str(e)
            else:
                raise AssertionError("expected LLMError")


class TestRecommendationVerifier:
    """Tests for the LLM recommendation verifier."""

    def _make_verifier(self, response_text: str) -> RecommendationVerifier:
        client = MagicMock()
        client.config = LLMConfig(
            provider=LLMProvider.OPENAI, api_key="test", model="gpt-4o-mini"
        )
        client.complete.return_value = response_text
        return RecommendationVerifier(client)

    def test_parse_json_response(self):
        response = json.dumps({
            "summary": "Good recommendations overall.",
            "agreements": ["Embedding model choice is appropriate"],
            "refinements": ["Consider increasing chunk overlap"],
            "additional_considerations": ["Test with real queries"],
        })
        verifier = self._make_verifier(response)
        result = verifier.verify(UserAnswers(), Recommendations())

        assert result.summary == "Good recommendations overall."
        assert len(result.agreements) == 1
        assert len(result.refinements) == 1
        assert len(result.additional_considerations) == 1

    def test_parse_markdown_wrapped_json(self):
        response = (
            '```json\n{"summary": "OK", "agreements": [], "refinements": [], '
            '"additional_considerations": []}\n```'
        )
        verifier = self._make_verifier(response)
        result = verifier.verify(UserAnswers(), Recommendations())
        assert result.summary == "OK"

    def test_fallback_on_invalid_json(self):
        verifier = self._make_verifier("This is not JSON at all.")
        result = verifier.verify(UserAnswers(), Recommendations())
        assert "This is not JSON" in result.summary

    def test_prompt_includes_user_scenario(self):
        client = MagicMock()
        client.config = LLMConfig(
            provider=LLMProvider.OPENAI, api_key="test", model="test"
        )
        client.complete.return_value = (
            '{"summary":"ok","agreements":[],"refinements":[],'
            '"additional_considerations":[]}'
        )

        verifier = RecommendationVerifier(client)
        answers = UserAnswers()
        recs = Recommendations()
        recs.retrieval = RetrievalRecommendation(top_k=10)
        verifier.verify(answers, recs)

        # Check that the prompt was constructed
        call_args = client.complete.call_args
        user_prompt = call_args[0][1]
        assert "question_answering" in user_prompt
        assert "top_k=10" in user_prompt
