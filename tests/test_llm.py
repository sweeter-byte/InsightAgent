"""Smoke tests for `LLMClient` configuration handling.

These do NOT hit any real API. Real-endpoint smoke testing is documented in
README as a manual step.
"""

from __future__ import annotations

import pytest

from insight_agent.llm import LLMClient


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_missing_all_env_vars_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        LLMClient()


def test_missing_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-dummy")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    with pytest.raises(RuntimeError, match="LLM_BASE_URL"):
        LLMClient()


def test_missing_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-dummy")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        LLMClient()
