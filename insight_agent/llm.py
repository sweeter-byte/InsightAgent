"""Thin wrapper around the OpenAI-compatible Chat Completions client.

`LLMClient` is intentionally minimal. It:
  1. reads model configuration from environment variables;
  2. builds an `openai.OpenAI` client;
  3. exposes a single `chat(...)` method that forwards arguments to the SDK
     and returns the raw SDK response.

No agent loop, no tool execution, no retry / memory / cache logic lives here.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


_ENV_VAR_HINTS: dict[str, str] = {
    "LLM_API_KEY": "API key for the OpenAI-compatible endpoint.",
    "LLM_BASE_URL": "Base URL of the OpenAI-compatible endpoint, e.g. https://api.openai.com/v1",
    "LLM_MODEL": "Model name to send to the endpoint, e.g. gpt-4o-mini.",
}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"{_ENV_VAR_HINTS[name]} "
            f"Please set it in your shell or in a `.env` file "
            f"(see `.env.example` for reference)."
        )
    return value


class LLMClient:
    """A thin OpenAI-compatible Chat Completions client.

    Attributes:
        model:  the model name read from ``LLM_MODEL``.
        client: the underlying ``openai.OpenAI`` instance.
    """

    def __init__(self) -> None:
        api_key = _require_env("LLM_API_KEY")
        base_url = _require_env("LLM_BASE_URL")
        self.model: str = _require_env("LLM_MODEL")
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        """Call the Chat Completions API and return the raw SDK response.

        Args:
            messages: OpenAI-format chat messages.
            tools:    Optional list of OpenAI-format tool schemas. When omitted,
                      the call is made without the ``tools`` kwarg so the model
                      cannot emit tool calls.

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
        return self.client.chat.completions.create(**kwargs)
