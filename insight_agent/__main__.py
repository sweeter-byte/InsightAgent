"""Command-line entry: ``python -m insight_agent``.

Behavior:
  * With no arguments — start an interactive REPL, one query per line.
  * With arguments — run a single query built from ``sys.argv[1:]`` and exit.

Nothing here is required by the agent itself; the CLI is just a thin wrapper
that assembles an ``LLMClient`` + default registry + ``ResearchAgent`` and
reads a line of user input.
"""

from __future__ import annotations

import sys
from typing import Optional

from insight_agent.agent import AgentStepsExceeded, ResearchAgent
from insight_agent.llm import LLMClient
from insight_agent.tools.registry import build_default_registry, default_tool_schemas


_BANNER = (
    "InsightAgent — minimal Research Agent. Type a request, or `exit`/Ctrl-D to quit."
)


def _build_agent() -> ResearchAgent:
    return ResearchAgent(
        llm=LLMClient(),
        registry=build_default_registry(),
        tool_schemas=default_tool_schemas(),
    )


def _print_answer(answer: str) -> None:
    print(f"\n{answer.strip()}\n" if answer and answer.strip() else "\n[empty response]\n")


def _run_once(agent: ResearchAgent, query: str) -> Optional[int]:
    """Execute a single query. Returns an exit code, or ``None`` to keep going."""
    try:
        answer = agent.run(query)
    except AgentStepsExceeded as exc:
        print(f"[agent stopped] {exc}", file=sys.stderr)
        return 1
    _print_answer(answer)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        agent = _build_agent()
    except RuntimeError as exc:  # missing env vars
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    if args:
        rc = _run_once(agent, " ".join(args))
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
        rc = _run_once(agent, line)
        if rc is not None:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
