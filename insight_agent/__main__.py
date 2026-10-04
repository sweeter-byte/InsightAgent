"""Command-line entry: ``python -m insight_agent``.

Behavior:
  * With no arguments — start an interactive REPL, one query per line.
  * With arguments — run a single query built from ``sys.argv[1:]`` and exit.

Nothing here is required by the agent itself; the CLI is just a thin wrapper
that assembles the app — a shared ``LLMClient``, a default registry,
``ResearchAgent``, both routers, the research workflow, and the ``InsightAgent``
façade — and reads a line of user input.
"""

from __future__ import annotations

import sys
from typing import Optional

from insight_agent.agent import AgentStepsExceeded, ResearchAgent
from insight_agent.app import InsightAgent
from insight_agent.llm import LLMClient
from insight_agent.planning import PlanningError, ResearchCoordinator, ResearchPlanner
from insight_agent.research import ResearchRoutingWorkflow
from insight_agent.router import IntentRouter
from insight_agent.routing import RetrievalSource, RoutingError
from insight_agent.routing.router import RetrievalRouter
from insight_agent.tools.registry import build_default_registry, default_tool_schemas


_BANNER = (
    "InsightAgent — minimal Research Agent. Type a request, or `exit`/Ctrl-D to quit."
)


def _build_app() -> InsightAgent:
    """Compose the full application object.

    A single ``LLMClient`` instance is shared by both routers, the planner, the
    direct-answer path, and the ResearchAgent. We deliberately avoid a factory
    or DI framework for this small object graph.
    """
    llm = LLMClient()
    research_agent = ResearchAgent(
        llm=llm,
        registry=build_default_registry(),
        tool_schemas=default_tool_schemas(),
    )
    planner = ResearchPlanner(llm=llm)
    retrieval_router = RetrievalRouter(llm=llm)
    research_workflow = ResearchRoutingWorkflow(router=retrieval_router)
    research_coordinator = ResearchCoordinator(
        planner=planner,
        workflow=research_workflow,
        research_agent=research_agent,
        # All three are valid routing targets in Chapter 8. Web and vision
        # entry nodes record decisions only; they do not execute retrieval.
        available_sources=set(RetrievalSource),
    )
    router = IntentRouter(llm=llm)
    return InsightAgent(
        router=router,
        llm=llm,
        research_agent=research_agent,
        research_coordinator=research_coordinator,
    )


def _print_answer(answer: str) -> None:
    print(f"\n{answer.strip()}\n" if answer and answer.strip() else "\n[empty response]\n")


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
    _print_answer(answer)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        app = _build_app()
    except RuntimeError as exc:  # missing env vars
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

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


if __name__ == "__main__":
    raise SystemExit(main())
