"""Lightweight LLM client — OpenAI and Anthropic via httpx."""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass

import httpx

from rag_adviser.models import RagAdvisorError

logger = logging.getLogger(__name__)


class LLMError(RagAdvisorError):
    """LLM API call failed."""


class LLMProvider(enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models. Override with RAGADVISOR_LLM_MODEL.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class LLMConfig:
    """Configuration for an LLM API call."""

    provider: LLMProvider
    api_key: str
    model: str
    base_url: str | None = None
    max_tokens: int = 2000
    temperature: float = 0.3


def detect_llm_config() -> LLMConfig | None:
    """Auto-detect available API keys from environment variables.

    Checks ANTHROPIC_API_KEY first (preferred), then OPENAI_API_KEY.
    Returns None if no keys are found.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        return LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=anthropic_key,
            model=os.environ.get("RAGADVISOR_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=openai_key,
            model=os.environ.get("RAGADVISOR_LLM_MODEL", DEFAULT_OPENAI_MODEL),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )

    return None


class LLMClient:
    """Thin wrapper over OpenAI / Anthropic HTTP APIs using httpx."""

    TIMEOUT = 60.0

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a completion request and return the text response."""
        if self.config.provider == LLMProvider.OPENAI:
            return self._complete_openai(system_prompt, user_prompt)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._complete_anthropic(system_prompt, user_prompt)
        raise LLMError(f"Unsupported provider: {self.config.provider}")

    def _complete_openai(self, system: str, user: str) -> str:
        base = self.config.base_url or "https://api.openai.com"
        url = f"{base.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        try:
            resp = httpx.post(
                url, json=payload, headers=headers, timeout=self.TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"OpenAI API error {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except (httpx.RequestError, KeyError, IndexError) as e:
            raise LLMError(f"OpenAI API request failed: {e}") from e

    def _complete_anthropic(self, system: str, user: str) -> str:
        base = self.config.base_url or "https://api.anthropic.com"
        url = f"{base.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        # Current Claude models (Opus 5, Sonnet 5, 4.6+) reject sampling
        # parameters such as `temperature` with a 400, so none are sent.
        payload = {
            "model": self.config.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.config.max_tokens,
        }
        try:
            resp = httpx.post(
                url, json=payload, headers=headers, timeout=self.TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("stop_reason") == "refusal":
                raise LLMError("Anthropic API declined the request (stop_reason=refusal)")
            # Skip non-text blocks (e.g. thinking) and join the text ones.
            texts = [
                b.get("text", "")
                for b in data["content"]
                if b.get("type", "text") == "text"
            ]
            if not texts:
                raise LLMError("Anthropic API returned no text content")
            return "".join(texts)
        except httpx.HTTPStatusError as e:
            raise LLMError(
                f"Anthropic API error {e.response.status_code}: {e.response.text[:300]}"
            ) from e
        except (httpx.RequestError, KeyError, IndexError) as e:
            raise LLMError(f"Anthropic API request failed: {e}") from e
