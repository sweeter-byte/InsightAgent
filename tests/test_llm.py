"""Smoke tests for `LLMClient` configuration handling.

These do NOT hit any real API. Real-endpoint smoke testing is documented in
README as a manual step.
"""

from __future__ import annotations

import pytest

from insight_agent.llm import LLMClient, LLMConfig


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_llm_config_reads_mapping_without_exposing_secret() -> None:
    config = LLMConfig.from_env(
        {
            "LLM_API_KEY": " super-secret ",
            "LLM_BASE_URL": " https://example.invalid/v1 ",
            "LLM_MODEL": " model-a ",
        }
    )

    assert config == LLMConfig(
        api_key="super-secret",
        base_url="https://example.invalid/v1",
        model="model-a",
    )
    assert "super-secret" not in repr(config)


@pytest.mark.parametrize("field", ["api_key", "base_url", "model"])
def test_llm_config_rejects_empty_direct_values(field: str) -> None:
    values = {
        "api_key": "secret",
        "base_url": "https://example.invalid/v1",
        "model": "model-a",
    }
    values[field] = "  "

    with pytest.raises(ValueError, match=field):
        LLMConfig(**values)


def test_llm_config_mapping_error_names_variable_without_secret() -> None:
    secret = "secret-that-must-not-leak"

    with pytest.raises(RuntimeError, match="LLM_BASE_URL") as excinfo:
        LLMConfig.from_env({"LLM_API_KEY": secret, "LLM_MODEL": "model-a"})

    assert secret not in str(excinfo.value)


def test_llm_client_accepts_config_and_closes_sdk_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)

    client = LLMClient(
        LLMConfig(
            api_key="secret",
            base_url="https://example.invalid/v1",
            model="model-a",
        )
    )
    client.close()
    client.close()

    assert client.model == "model-a"
    assert client.client.kwargs == {
        "api_key": "secret",
        "base_url": "https://example.invalid/v1",
    }
    assert client.client.close_calls == 1


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


def test_chat_forwards_optional_timeout_without_changing_legacy_calls() -> None:
    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "response"

    client = object.__new__(LLMClient)
    client.model = "judge-model"
    client.client = type(
        "FakeOpenAI",
        (),
        {"chat": type("FakeChat", (), {"completions": FakeCompletions()})()},
    )()

    assert client.chat([{"role": "user", "content": "first"}]) == "response"
    assert client.chat(
        [{"role": "user", "content": "second"}],
        tools=None,
        timeout=2.5,
    ) == "response"

    assert "timeout" not in calls[0]
    assert calls[1]["timeout"] == 2.5
