"""Tests for the CLI adapter in ``insight_agent.__main__``."""

from __future__ import annotations

import builtins

import pytest

from insight_agent import __main__ as cli
from insight_agent.agent import AgentStepsExceeded
from insight_agent.application import AppConfig
from insight_agent.app import InsightAgent
from insight_agent.evidence import EvidenceGradingError
from insight_agent.planning import PlanningError
from insight_agent.reporting import ReportGenerationError
from insight_agent.routing import RoutingError
from insight_agent.self_check import SelfCheckError
from insight_agent.vision_retrieval import VisionRetrievalError
from insight_agent.web_search import WebSearchError


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
        self.close_calls = 0

    def run(self, query: str) -> str:
        self.calls.append(query)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.reply

    def close(self) -> None:
        self.close_calls += 1


def _main_with_app(
    app: FakeApp,
    argv: list[str] | None = None,
    *,
    config: AppConfig | object = object(),
) -> int:
    def config_factory() -> AppConfig:
        return config  # type: ignore[return-value]

    def application_factory(received: AppConfig) -> InsightAgent:
        assert received is config
        return app  # type: ignore[return-value]

    return cli.main(
        argv,
        config_factory=config_factory,
        application_factory=application_factory,
    )


def test_main_one_shot_loads_config_builds_runs_prints_and_closes_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = object()
    app = FakeApp(reply="one-shot result")
    config_calls: list[None] = []
    built_configs: list[AppConfig] = []

    def config_factory() -> AppConfig:
        config_calls.append(None)
        return config  # type: ignore[return-value]

    def application_factory(received: AppConfig) -> InsightAgent:
        built_configs.append(received)
        return app  # type: ignore[return-value]

    rc = cli.main(
        ["读取", "notes.txt"],
        config_factory=config_factory,
        application_factory=application_factory,
    )

    assert rc == 0
    assert len(config_calls) == 1
    assert len(built_configs) == 1
    assert built_configs[0] is config
    assert app.calls == ["读取 notes.txt"]
    assert app.close_calls == 1
    assert "one-shot result" in capsys.readouterr().out


def test_main_config_factory_error_returns_2_without_closing_app(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def config_factory() -> AppConfig:
        raise RuntimeError("Missing required environment variable 'LLM_API_KEY'.")

    def application_factory(_config: AppConfig) -> InsightAgent:
        pytest.fail("application factory must not run when config loading fails")

    rc = cli.main(
        ["anything"],
        config_factory=config_factory,
        application_factory=application_factory,
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "[config error]" in captured.err
    assert "LLM_API_KEY" in captured.err


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_main_application_factory_configuration_error_returns_2(
    capsys: pytest.CaptureFixture[str], error_type: type[Exception]
) -> None:
    config = object()
    app = FakeApp()
    config_calls: list[None] = []
    build_calls: list[AppConfig] = []

    def config_factory() -> AppConfig:
        config_calls.append(None)
        return config  # type: ignore[return-value]

    def application_factory(received: AppConfig) -> InsightAgent:
        build_calls.append(received)
        raise error_type("invalid configuration")

    rc = cli.main(
        ["anything"],
        config_factory=config_factory,
        application_factory=application_factory,
    )

    assert rc == 2
    assert len(config_calls) == 1
    assert len(build_calls) == 1
    assert build_calls[0] is config
    assert app.close_calls == 0
    assert "[config error]" in capsys.readouterr().err


def test_main_application_factory_error_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "highly-sensitive-key"
    monkeypatch.setenv("LLM_API_KEY", secret)

    def application_factory(_config: AppConfig) -> InsightAgent:
        raise RuntimeError(f"invalid configuration containing {secret}")

    rc = cli.main(
        ["anything"],
        config_factory=lambda: object(),  # type: ignore[arg-type,return-value]
        application_factory=application_factory,
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert "[config error]" in captured.err
    assert secret not in captured.err


def test_main_application_factory_unexpected_error_remains_visible() -> None:
    def application_factory(_config: AppConfig) -> InsightAgent:
        raise KeyError("unexpected factory failure")

    with pytest.raises(KeyError, match="unexpected factory failure"):
        cli.main(
            ["anything"],
            config_factory=lambda: object(),  # type: ignore[arg-type,return-value]
            application_factory=application_factory,
        )


@pytest.mark.parametrize(
    ("error", "label", "detail"),
    [
        (
            AgentStepsExceeded("exceeded 10 steps"),
            "[agent stopped]",
            "exceeded 10 steps",
        ),
        (
            PlanningError("planner returned invalid JSON"),
            "[planning error]",
            "invalid JSON",
        ),
        (
            RoutingError("router returned unknown source 'video'"),
            "[routing error]",
            "unknown source",
        ),
        (
            EvidenceGradingError("grader returned unknown Evidence ID"),
            "[evidence grading error]",
            "unknown Evidence ID",
        ),
        (
            ReportGenerationError("Claim references disallowed Evidence"),
            "[report generation error]",
            "disallowed Evidence",
        ),
        (
            SelfCheckError("Repair failed after one round"),
            "[self-check error]",
            "Repair failed",
        ),
        (
            WebSearchError("Tavily unavailable"),
            "[web search error]",
            "Tavily unavailable",
        ),
        (
            VisionRetrievalError("Vision index unavailable"),
            "[vision retrieval error]",
            "Vision index unavailable",
        ),
    ],
)
def test_main_one_shot_domain_error_returns_1(
    error: Exception,
    label: str,
    detail: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeApp(raise_exc=error)

    rc = _main_with_app(app, ["research", "topic"])

    assert rc == 1
    captured = capsys.readouterr()
    assert label in captured.err
    assert detail in captured.err
    assert app.calls == ["research topic"]
    assert app.close_calls == 1


def test_main_repl_runs_query_then_exits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeApp(reply="repl answer")
    inputs = iter(["什么是 Agent Loop？", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    rc = _main_with_app(app, [])

    assert rc == 0
    assert app.calls == ["什么是 Agent Loop？"]
    assert app.close_calls == 1
    out = capsys.readouterr().out
    assert cli._BANNER in out
    assert "repl answer" in out


def test_main_repl_handles_eof(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeApp(reply="bye")
    inputs = iter(["hello"])

    def side_effect(_prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(builtins, "input", side_effect)

    rc = _main_with_app(app, [])

    assert rc == 0
    assert app.calls == ["hello"]
    assert app.close_calls == 1
    assert capsys.readouterr().out.endswith("\n")


def test_main_repl_skips_blank_lines(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = FakeApp(reply="only once")
    inputs = iter(["", "   ", "real query", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    rc = _main_with_app(app, [])

    assert rc == 0
    assert app.calls == ["real query"]
    assert app.close_calls == 1
    capsys.readouterr()
