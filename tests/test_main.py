"""Tests for the CLI / composition layer in ``insight_agent.__main__``.

Two concerns are covered here:

  1. ``_build_app()`` wires the collaborators together correctly and shares a
     single ``LLMClient`` across router / direct path / ResearchAgent.
  2. ``main()`` keeps both CLI modes working (one-shot + REPL) and preserves the
     Chapter 1 error handling (config error, AgentStepsExceeded).

Nothing here contacts a real endpoint: ``openai.OpenAI`` is stubbed out for the
composition test, and ``_build_app`` is replaced by a fake app for the CLI test.
"""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from insight_agent import __main__ as cli
from insight_agent.agent import AgentStepsExceeded, ResearchAgent
from insight_agent.app import InsightAgent
from insight_agent.llm import LLMClient
from insight_agent.router import IntentRouter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeApp:
    """Stand-in for ``InsightAgent`` used to drive ``main()`` without an LLM."""

    def __init__(
        self,
        reply: str = "cli answer",
        *,
        raise_exc: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls: list[str] = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.reply


# ---------------------------------------------------------------------------
# 1. _build_app composition
# ---------------------------------------------------------------------------


def test_build_app_returns_insight_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_build_app`` assembles the full object graph from configured env vars."""

    class DummyOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    # Replace the real SDK client so construction never touches the network.
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("insight_agent.llm.OpenAI", DummyOpenAI)

    app = cli._build_app()

    assert isinstance(app, InsightAgent)
    assert isinstance(app.router, IntentRouter)
    assert isinstance(app.research_agent, ResearchAgent)
    assert isinstance(app.llm, LLMClient)


def test_build_app_shares_single_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Router, direct path and ResearchAgent must reuse ONE ``LLMClient``."""

    class DummyOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr("insight_agent.llm.OpenAI", DummyOpenAI)

    app = cli._build_app()

    assert app.llm is app.router.llm is app.research_agent.llm


# ---------------------------------------------------------------------------
# 2. one-shot mode
# ---------------------------------------------------------------------------


def test_main_one_shot_prints_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    fake = FakeApp(reply="one-shot result")
    monkeypatch.setattr(cli, "_build_app", lambda: fake)

    rc = cli.main(["读取", "examples/result.txt"])

    assert rc == 0
    assert fake.calls == ["读取 examples/result.txt"]
    assert "one-shot result" in capsys.readouterr().out


def test_main_one_shot_config_error_returns_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    def boom() -> InsightAgent:
        raise RuntimeError("Missing required environment variable 'LLM_API_KEY'.")

    monkeypatch.setattr(cli, "_build_app", boom)

    rc = cli.main(["anything"])

    assert rc == 2
    captured = capsys.readouterr()
    assert "[config error]" in captured.err
    assert "LLM_API_KEY" in captured.err


def test_main_one_shot_agent_steps_exceeded_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    fake = FakeApp(raise_exc=AgentStepsExceeded("exceeded 10 steps"))
    monkeypatch.setattr(cli, "_build_app", lambda: fake)

    rc = cli.main(["loop forever"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "[agent stopped]" in captured.err
    assert "exceeded 10 steps" in captured.err


# ---------------------------------------------------------------------------
# 3. REPL mode
# ---------------------------------------------------------------------------


def test_main_repl_runs_query_then_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    fake = FakeApp(reply="repl answer")
    monkeypatch.setattr(cli, "_build_app", lambda: fake)

    # First line is a real query, second exits the loop.
    inputs = iter(["什么是 Agent Loop？", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    rc = cli.main([])

    assert rc == 0
    assert fake.calls == ["什么是 Agent Loop？"]
    out = capsys.readouterr().out
    assert cli._BANNER in out
    assert "repl answer" in out


def test_main_repl_handles_eof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    fake = FakeApp(reply="bye")
    monkeypatch.setattr(cli, "_build_app", lambda: fake)

    inputs = iter(["hello"])

    def _side_effect(_prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", _side_effect)

    rc = cli.main([])

    assert rc == 0
    assert fake.calls == ["hello"]


def test_main_repl_skips_blank_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureManager
) -> None:
    fake = FakeApp(reply="only once")
    monkeypatch.setattr(cli, "_build_app", lambda: fake)

    inputs = iter(["", "   ", "real query", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    rc = cli.main([])

    assert rc == 0
    # blank / whitespace-only lines must not reach the app
    assert fake.calls == ["real query"]
