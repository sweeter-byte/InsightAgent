"""Smoke tests for `LLMClient` configuration handling.

These do NOT hit any real API. Real-endpoint smoke testing is documented in
README as a manual step.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from insight_agent.llm import LLMClient, LLMConfig, _require_env


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Missing-variable tests must not refill values from a developer's .env.
    monkeypatch.setattr("insight_agent.llm.load_dotenv", lambda: False)
    for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("module", ["insight_agent.llm", "insight_agent.runtime.app"])
def test_import_does_not_load_dotenv_in_fresh_interpreter(
    module: str,
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("INSIGHT_IMPORT_SENTINEL=loaded\n", encoding="utf-8")
    env = dict(os.environ)
    env.pop("INSIGHT_IMPORT_SENTINEL", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import dotenv, importlib, os, sys\n"
            "calls = []\n"
            "original = dotenv.load_dotenv\n"
            "def track_load(*args, **kwargs):\n"
            "    calls.append(True)\n"
            "    return original(os.path.join(os.getcwd(), '.env'), *args, **kwargs)\n"
            "dotenv.load_dotenv = track_load\n"
            "importlib.import_module(sys.argv[1])\n"
            "assert not calls, 'dotenv was loaded during import'\n"
            "assert 'INSIGHT_IMPORT_SENTINEL' not in os.environ\n",
            module,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


def test_llm_config_loads_dotenv_only_when_reading_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dotenv import load_dotenv

    _clear_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "LLM_API_KEY=dotenv-secret\n"
        "LLM_BASE_URL=https://dotenv.example.invalid/v1\n"
        "LLM_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("insight_agent.llm.load_dotenv", lambda: load_dotenv(dotenv_path))

    assert LLMConfig.from_env() == LLMConfig(
        "dotenv-secret", "https://dotenv.example.invalid/v1", "dotenv-model"
    )


def test_llm_config_explicit_mapping_is_isolated_from_dotenv_and_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "process-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://process.example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "process-model")
    monkeypatch.setattr(
        "insight_agent.llm.load_dotenv",
        lambda: pytest.fail("explicit mapping must not load dotenv"),
    )

    config = LLMConfig.from_env(
        {
            "LLM_API_KEY": "mapping-secret",
            "LLM_BASE_URL": "https://mapping.example.invalid/v1",
            "LLM_MODEL": "mapping-model",
        }
    )
    assert config == LLMConfig(
        "mapping-secret", "https://mapping.example.invalid/v1", "mapping-model"
    )
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        LLMConfig.from_env({})


def test_legacy_require_env_loads_dotenv_before_reading_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dotenv import load_dotenv

    monkeypatch.delenv("VISION_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("VISION_API_KEY=legacy-dotenv-secret\n", encoding="utf-8")
    monkeypatch.setattr("insight_agent.llm.load_dotenv", lambda: load_dotenv(dotenv_path))

    assert _require_env("VISION_API_KEY") == "legacy-dotenv-secret"


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
