"""Command-line entry: ``python -m insight_agent``.

The CLI is a thin adapter consuming the shared application composition root.
With no arguments it starts an interactive REPL; with arguments it runs one
query formed from ``sys.argv[1:]`` and exits.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from typing import Optional

from insight_agent.agent import AgentStepsExceeded
from insight_agent.app import InsightAgent
from insight_agent.application import AppConfig, build_application
from insight_agent.evidence import EvidenceGradingError
from insight_agent.planning import PlanningError
from insight_agent.reporting import ReportGenerationError
from insight_agent.routing import RoutingError
from insight_agent.self_check import SelfCheckError
from insight_agent.vision_retrieval import VisionRetrievalError
from insight_agent.web_search import WebSearchError


_BANNER = (
    "InsightAgent — minimal Research Agent. Type a request, or `exit`/Ctrl-D to quit."
)


def _print_answer(answer: str) -> None:
    print(f"\n{answer.strip()}\n" if answer and answer.strip() else "\n[empty response]\n")


def _redact_config_error(message: str, config: AppConfig | None) -> str:
    """Hide startup credentials while preserving readable diagnostics."""
    candidates: set[str] = set()
    for name, value in os.environ.items():
        if any(
            marker in name.upper()
            for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            candidates.update((value, value.strip()))
    if config is not None:
        for component in ("llm", "vision", "web_search"):
            value = getattr(getattr(config, component, None), "api_key", None)
            if isinstance(value, str):
                candidates.update((value, value.strip()))
    for value in sorted(candidates - {""}, key=len, reverse=True):
        token = rf"(?<![^\W_]){re.escape(value)}(?![^\W_])"
        message = re.sub(token, "[redacted]", message)
    return message


def _run_once(app: InsightAgent, query: str) -> Optional[int]:
    """Execute a single query. Returns an exit code, or ``None`` to keep going."""
    try:
        answer = app.run(query)
    except AgentStepsExceeded as exc:
        print(f"[agent stopped] {exc}", file=sys.stderr)
        return 1
    except PlanningError as exc:
        print(f"[planning error] {exc}", file=sys.stderr)
        return 1
    except RoutingError as exc:
        print(f"[routing error] {exc}", file=sys.stderr)
        return 1
    except EvidenceGradingError as exc:
        print(f"[evidence grading error] {exc}", file=sys.stderr)
        return 1
    except ReportGenerationError as exc:
        print(f"[report generation error] {exc}", file=sys.stderr)
        return 1
    except SelfCheckError as exc:
        print(f"[self-check error] {exc}", file=sys.stderr)
        return 1
    except WebSearchError as exc:
        print(f"[web search error] {exc}", file=sys.stderr)
        return 1
    except VisionRetrievalError as exc:
        print(f"[vision retrieval error] {exc}", file=sys.stderr)
        return 1
    _print_answer(answer)
    return None


def main(
    argv: Optional[list[str]] = None,
    *,
    config_factory: Callable[[], AppConfig] = AppConfig.from_env,
    application_factory: Callable[[AppConfig], InsightAgent] = build_application,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    config = None
    try:
        config = config_factory()
        app = application_factory(config)
    except (RuntimeError, ValueError) as exc:
        message = _redact_config_error(str(exc), config)
        print(f"[config error] {message}", file=sys.stderr)
        return 2

    try:
        if args:
            rc = _run_once(app, " ".join(args))
            return rc if rc is not None else 0

        print(_BANNER)
        while True:
            try:
                line = input(">> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n[interrupted]")
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit", ":q"}:
                break
            rc = _run_once(app, line)
            if rc is not None:
                return rc
        return 0
    finally:
        close = getattr(app, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
