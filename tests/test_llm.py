"""Smoke tests for `LLMClient` configuration handling.

These do NOT hit any real API. Real-endpoint smoke testing is documented in
README as a manual step.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from insight_agent.llm import (
    ContextBudgetExceeded,
    LLMClient,
    LLMConfig,
    _estimate_chat_input_tokens,
    _require_env,
)

_MODEL_ENV_KEYS = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_CONTEXT_WINDOW_TOKENS",
    "LLM_RESERVED_OUTPUT_TOKENS",
    "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
    "LLM_REQUEST_METRICS_ENABLED",
    "VISION_API_KEY",
    "VISION_BASE_URL",
    "VISION_MODEL",
)


def _snapshot_model_env() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _MODEL_ENV_KEYS}


def _restore_model_env(snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    assert _snapshot_model_env() == snapshot


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Missing-variable tests must not refill values from a developer's .env.
    monkeypatch.setattr("insight_agent.llm.load_dotenv", lambda: False)
    for key in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_CONTEXT_WINDOW_TOKENS",
        "LLM_RESERVED_OUTPUT_TOKENS",
        "LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
        "LLM_REQUEST_METRICS_ENABLED",
    ):
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

    original_env = _snapshot_model_env()
    try:
        _clear_env(monkeypatch)
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "LLM_API_KEY=dotenv-secret\n"
            "LLM_BASE_URL=https://dotenv.example.invalid/v1\n"
            "LLM_MODEL=dotenv-model\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "insight_agent.llm.load_dotenv", lambda: load_dotenv(dotenv_path)
        )

        assert LLMConfig.from_env() == LLMConfig(
            "dotenv-secret", "https://dotenv.example.invalid/v1", "dotenv-model"
        )
    finally:
        _restore_model_env(original_env)


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

    original_env = _snapshot_model_env()
    try:
        monkeypatch.delenv("VISION_API_KEY", raising=False)
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "VISION_API_KEY=legacy-dotenv-secret\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            "insight_agent.llm.load_dotenv", lambda: load_dotenv(dotenv_path)
        )

        assert _require_env("VISION_API_KEY") == "legacy-dotenv-secret"
    finally:
        _restore_model_env(original_env)


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


def test_llm_config_reads_optional_context_budget() -> None:
    config = LLMConfig.from_env(
        {
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "model-a",
            "LLM_CONTEXT_WINDOW_TOKENS": " 128000 ",
            "LLM_RESERVED_OUTPUT_TOKENS": " 4096 ",
            "LLM_CONTEXT_SAFETY_MARGIN_TOKENS": " 1024 ",
        }
    )

    assert config.context_window_tokens == 128000
    assert config.reserved_output_tokens == 4096
    assert config.context_safety_margin_tokens == 1024


def test_llm_config_leaves_context_budget_disabled_by_default() -> None:
    config = LLMConfig("secret", "https://example.invalid/v1", "model-a")

    assert config.context_window_tokens is None
    assert config.reserved_output_tokens == 0
    assert config.context_safety_margin_tokens == 0
    assert config.request_metrics_enabled is False


def test_llm_config_reads_request_metrics_switch() -> None:
    config = LLMConfig.from_env(
        {
            "LLM_API_KEY": "secret",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "model-a",
            "LLM_REQUEST_METRICS_ENABLED": "true",
        }
    )

    assert config.request_metrics_enabled is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("context_window_tokens", 0),
        ("context_window_tokens", True),
        ("reserved_output_tokens", -1),
        ("context_safety_margin_tokens", -1),
    ],
)
def test_llm_config_rejects_invalid_context_budget(field: str, value: object) -> None:
    values: dict[str, object] = {
        "api_key": "secret",
        "base_url": "https://example.invalid/v1",
        "model": "model-a",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        LLMConfig(**values)  # type: ignore[arg-type]


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


def test_input_estimate_covers_all_message_roles_and_tool_definitions() -> None:
    empty_by_role = [
        {"role": role, "content": ""}
        for role in ("system", "user", "assistant", "tool")
    ]
    full_messages = [
        {**message, "content": f"private-{message['role']}-content"}
        for message in empty_by_role
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "private-tool-description",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    empty_estimate = _estimate_chat_input_tokens(empty_by_role, None)
    message_estimate = _estimate_chat_input_tokens(full_messages, None)
    tools_estimate = _estimate_chat_input_tokens(full_messages, tools)

    assert message_estimate > empty_estimate
    assert tools_estimate > message_estimate


def test_chat_allows_complete_request_within_context_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object):
            calls.append(kwargs)
            return SimpleNamespace(usage=None)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "tool request"},
        {"role": "tool", "content": "tool result"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object"},
            },
        }
    ]
    reserve = 8
    margin = 5
    estimated = _estimate_chat_input_tokens(messages, tools)
    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            "secret",
            "https://example.invalid/v1",
            "model",
            context_window_tokens=estimated + reserve + margin,
            reserved_output_tokens=reserve,
            context_safety_margin_tokens=margin,
        )
    )

    client.chat(messages, tools)

    assert len(calls) == 1
    assert calls[0]["messages"] == messages
    assert calls[0]["tools"] == tools


def test_chat_rejects_over_budget_request_before_sdk_without_leaking_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object):
            calls.append(kwargs)
            return SimpleNamespace(usage=None)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    secret = "api-key-that-must-not-leak"
    private_content = "private prompt and evidence body"
    messages = [{"role": "user", "content": private_content}]
    estimated = _estimate_chat_input_tokens(messages, None)
    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            secret,
            "https://example.invalid/v1",
            "model",
            context_window_tokens=estimated + 6,
            reserved_output_tokens=4,
            context_safety_margin_tokens=3,
        )
    )

    with pytest.raises(ContextBudgetExceeded) as excinfo:
        client.chat(messages)

    assert calls == []
    assert excinfo.value.context_window_tokens == estimated + 6
    assert excinfo.value.estimated_input_tokens == estimated
    assert excinfo.value.reserved_output_tokens == 4
    assert excinfo.value.safety_margin_tokens == 3
    assert excinfo.value.input_budget_tokens == estimated - 1
    assert private_content not in str(excinfo.value)
    assert secret not in str(excinfo.value)


def test_llm_client_records_provider_usage_without_inventing_missing_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    responses = [
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3)),
        SimpleNamespace(usage=None),
    ]

    class FakeCompletions:
        def create(self, **kwargs: object):
            return responses.pop(0)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(LLMConfig("secret", "https://example.invalid/v1", "model"))

    client.chat([{"role": "user", "content": "metered"}])
    first = client.usage_snapshot()
    client.chat([{"role": "user", "content": "unmetered"}])
    second = client.usage_snapshot()

    assert first == {
        "calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "unmetered_calls": 0,
    }
    assert second == {
        "calls": 2,
        "input_tokens": 7,
        "output_tokens": 3,
        "unmetered_calls": 1,
    }


def _request_metric_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "insight_agent.llm"
        and record.getMessage().startswith("{")
        and '"event":"llm_request_metrics"' in record.getMessage()
    ]


def test_chat_logs_estimated_and_actual_request_metrics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace

    class FakeCompletions:
        def create(self, **kwargs: object):
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=17, completion_tokens=5)
            )

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    messages = [{"role": "user", "content": "small request"}]
    estimated = _estimate_chat_input_tokens(messages, None)
    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            "secret",
            "https://example.invalid/v1",
            "model-a",
            context_window_tokens=1000,
            reserved_output_tokens=100,
            context_safety_margin_tokens=50,
            request_metrics_enabled=True,
        )
    )
    caplog.set_level(logging.INFO, logger="insight_agent.llm")

    client.chat(messages)

    assert _request_metric_events(caplog) == [
        {
            "actual_input_tokens": 17,
            "actual_output_tokens": 5,
            "budget_usage_ratio": estimated / 850,
            "context_window_tokens": 1000,
            "error_type": None,
            "estimated_input_tokens": estimated,
            "event": "llm_request_metrics",
            "input_budget_tokens": 850,
            "input_token_estimate_method": "model_agnostic_heuristic",
            "model": "model-a",
            "node_name": None,
            "request_sent": True,
            "status": "success",
            "task_id": None,
        }
    ]


def test_chat_logs_null_budget_fields_without_configured_context_window(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace

    class FakeCompletions:
        def create(self, **kwargs: object):
            return SimpleNamespace(usage=None)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    messages = [{"role": "user", "content": "unmetered request"}]
    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            "secret",
            "https://example.invalid/v1",
            "model-a",
            request_metrics_enabled=True,
        )
    )
    caplog.set_level(logging.INFO, logger="insight_agent.llm")

    client.chat(messages)

    event = _request_metric_events(caplog)[0]
    assert event["estimated_input_tokens"] == _estimate_chat_input_tokens(
        messages, None
    )
    assert event["context_window_tokens"] is None
    assert event["input_budget_tokens"] is None
    assert event["budget_usage_ratio"] is None
    assert event["actual_input_tokens"] is None
    assert event["actual_output_tokens"] is None
    assert event["node_name"] is None
    assert event["task_id"] is None


def test_chat_logs_context_rejection_without_sending_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object):
            calls.append(kwargs)
            return SimpleNamespace(usage=None)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    messages = [{"role": "user", "content": "request too large"}]
    estimated = _estimate_chat_input_tokens(messages, None)
    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            "secret",
            "https://example.invalid/v1",
            "model-a",
            context_window_tokens=estimated,
            reserved_output_tokens=1,
            request_metrics_enabled=True,
        )
    )
    caplog.set_level(logging.INFO, logger="insight_agent.llm")

    with pytest.raises(ContextBudgetExceeded):
        client.chat(messages)

    event = _request_metric_events(caplog)[0]
    assert event["status"] == "context_rejected"
    assert event["request_sent"] is False
    assert event["error_type"] == "ContextBudgetExceeded"
    assert event["actual_input_tokens"] is None
    assert event["actual_output_tokens"] is None
    assert calls == []


def test_chat_logs_api_error_without_swallowing_or_leaking_it(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace

    private_prompt = "private prompt and evidence body"
    secret = "api-key-that-must-not-leak"

    class ProviderFailure(RuntimeError):
        pass

    error = ProviderFailure(f"provider rejected {private_prompt} using {secret}")

    class FakeCompletions:
        def create(self, **kwargs: object):
            raise error

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(
        LLMConfig(
            secret,
            "https://example.invalid/v1",
            "model-a",
            request_metrics_enabled=True,
        )
    )
    caplog.set_level(logging.INFO, logger="insight_agent.llm")

    with pytest.raises(ProviderFailure) as excinfo:
        client.chat([{"role": "user", "content": private_prompt}])

    assert excinfo.value is error
    event = _request_metric_events(caplog)[0]
    assert event["status"] == "api_error"
    assert event["request_sent"] is True
    assert event["error_type"] == "ProviderFailure"
    assert event["actual_input_tokens"] is None
    assert event["actual_output_tokens"] is None
    assert private_prompt not in caplog.text
    assert secret not in caplog.text


def test_disabled_request_metrics_do_not_add_token_estimation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace

    class FakeCompletions:
        def create(self, **kwargs: object):
            return SimpleNamespace(usage=None)

    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            pass

    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    monkeypatch.setattr(
        "insight_agent.llm._estimate_chat_input_tokens",
        lambda *args: pytest.fail("disabled metrics must not estimate tokens"),
    )
    client = LLMClient(LLMConfig("secret", "https://example.invalid/v1", "model-a"))
    caplog.set_level(logging.INFO, logger="insight_agent.llm")

    client.chat([{"role": "user", "content": "request"}])

    assert _request_metric_events(caplog) == []
