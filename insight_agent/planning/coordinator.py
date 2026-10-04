"""Thin coordination from a research query to the existing agent loop."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from insight_agent.planning.models import ResearchPlan, ResearchState


class Planner(Protocol):
    def plan(self, query: str) -> ResearchPlan:
        """Return a validated plan for one query."""
        ...


class ResearchRunner(Protocol):
    def run(self, query: str) -> str:
        """Execute one research context and return its answer."""
        ...


class ResearchCoordinator:
    """Plan a research query, retain its state, then reuse ``ResearchAgent``."""

    def __init__(
        self,
        planner: Planner,
        research_agent: ResearchRunner,
    ) -> None:
        self.planner = planner
        self.research_agent = research_agent
        self.last_state: ResearchState | None = None

    def run(self, query: str) -> str:
        """Create state, attach a validated plan, and delegate execution."""
        state = ResearchState(query=query)
        state.plan = self.planner.plan(query)
        self.last_state = state
        return self.research_agent.run(format_research_context(state))


def format_research_context(state: ResearchState) -> str:
    """Render a planned state as execution context for the existing agent."""
    if state.plan is None:
        raise ValueError("research state must contain a plan before execution")

    plan_json = json.dumps(
        asdict(state.plan),
        ensure_ascii=False,
        indent=2,
    )
    return (
        "Original user request:\n"
        f"{state.query}\n\n"
        "Research Plan (structured JSON):\n"
        f"{plan_json}\n\n"
        "Execution instructions:\n"
        "- Follow the research plan and address its tasks in dependency order.\n"
        "- Task descriptions are planning instructions, not factual evidence.\n"
        "- When factual support is needed, use the existing retrieval tools, "
        "including search_knowledge_base, and ground the answer in retrieved "
        "evidence.\n"
        "- If retrieved evidence is insufficient, say so rather than treating "
        "the plan as evidence."
    )
