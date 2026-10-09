"""Thin wrapper around the OpenAI-compatible Chat Completions client.

`LLMClient` is intentionally minimal. It:
  1. accepts an explicit ``LLMConfig`` or falls back to environment variables;
  2. builds an `openai.OpenAI` client;
  3. exposes a single `chat(...)` method that forwards arguments to the SDK
     and returns the raw SDK response.

No agent loop, no tool execution, no retry / memory / cache logic lives here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# Single, shared registry of required environment variables and their hints.
# Both the text LLM client and the vision client read from here so the project
# keeps ONE config mechanism instead of a second, parallel one.
_ENV_VAR_HINTS: dict[str, str] = {
    "LLM_API_KEY": "API key for the OpenAI-compatible endpoint.",
    "LLM_BASE_URL": "Base URL of the OpenAI-compatible endpoint, e.g. https://api.openai.com/v1",
    "LLM_MODEL": "Model name to send to the endpoint, e.g. gpt-4o-mini.",
    "VISION_API_KEY": "API key for the OpenAI-compatible vision endpoint.",
    "VISION_BASE_URL": "Base URL of the OpenAI-compatible vision endpoint, e.g. https://api.openai.com/v1",
    "VISION_MODEL": "Vision-language model name, e.g. gpt-4o-mini.",
}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        hint = _ENV_VAR_HINTS.get(name, "")
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"{hint} "
            f"Please set it in your shell or in a `.env` file "
            f"(see `.env.example` for reference)."
        )
    return value


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Validated configuration for one OpenAI-compatible text model."""

    api_key: str = field(repr=False)
    base_url: str
    model: str

    def __post_init__(self) -> None:
        for name in ("api_key", "base_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLM {name} must not be empty")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> LLMConfig:
        """Resolve the existing ``LLM_*`` variables from a mapping."""
        values = os.environ if environ is None else environ
        resolved: dict[str, str] = {}
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            value = values.get(name, "").strip()
            if not value:
                hint = _ENV_VAR_HINTS[name]
                raise RuntimeError(
                    f"Missing required environment variable {name!r}. "
                    f"{hint} Please set it in your shell or in a `.env` file "
                    f"(see `.env.example` for reference)."
                )
            resolved[name] = value
        return cls(
            api_key=resolved["LLM_API_KEY"],
            base_url=resolved["LLM_BASE_URL"],
            model=resolved["LLM_MODEL"],
        )


class LLMClient:
    """A thin OpenAI-compatible Chat Completions client.

    Attributes:
        model:  the configured model name.
        client: the underlying ``openai.OpenAI`` instance.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        resolved = config if config is not None else LLMConfig.from_env()
        self.model: str = resolved.model
        self.client: OpenAI = OpenAI(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
        )
        self._closed = False

    def close(self) -> None:
        """Release the SDK connection pool at most once."""
        if self._closed:
            return
        self._closed = True
        self.client.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Call the Chat Completions API and return the raw SDK response.

        Args:
            messages: OpenAI-format chat messages.
            tools:    Optional list of OpenAI-format tool schemas. When omitted,
                      the call is made without the ``tools`` kwarg so the model
                      cannot emit tool calls.
            timeout:  Optional per-request SDK deadline in seconds.

        Returns:
            The ``ChatCompletion`` object produced by the SDK. The caller is
            responsible for reading ``.choices[0].message`` etc.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        if tools is not None:
            kwargs["tools"] = list(tools)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return self.client.chat.completions.create(**kwargs)
