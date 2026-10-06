"""Thin coordination from a research query to its rendered workflow report."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from insight_agent.planning.models import ResearchPlan, ResearchState
from insight_agent.routing.models import RetrievalSource


class Planner(Protocol):
    def plan(self, query: str) -> ResearchPlan:
        """Return a validated plan for one query."""
        ...


class ResearchWorkflow(Protocol):
    def run(self, state: ResearchState) -> ResearchState:
        """Route every planned task and return the final workflow state."""
        ...


class ResearchCoordinator:
    """Plan a research query, retain its state, and return its rendered report."""

    def __init__(
        self,
        planner: Planner,
        workflow: ResearchWorkflow,
        available_sources: set[RetrievalSource],
    ) -> None:
        if not available_sources:
            raise ValueError("available_sources must not be empty")
        if any(not isinstance(source, RetrievalSource) for source in available_sources):
            raise ValueError("available_sources must contain RetrievalSource values")
        self.planner = planner
        self.workflow = workflow
        self.available_sources = set(available_sources)
        self.last_state: ResearchState | None = None

    def run(self, query: str) -> str:
        """Create state, attach a validated plan, and delegate execution."""
        state = ResearchState(
            query=query,
            plan=self.planner.plan(query),
            available_sources=set(self.available_sources),
        )
        final_state = self.workflow.run(state)
        self.last_state = final_state
        if not isinstance(final_state.final_output, str):
            raise ValueError("research workflow did not produce final_output")
        return final_state.final_output


def format_research_context(state: ResearchState) -> str:
    """Render a planned state as execution context for the existing agent."""
    if state.plan is None:
        raise ValueError("research state must contain a plan before execution")

    plan_json = json.dumps(
        asdict(state.plan),
        ensure_ascii=False,
        indent=2,
    )
    task_by_id = {task.id: task for task in state.plan.tasks}
    route_rows: list[dict[str, str]] = []
    for decision in state.route_decisions:
        task = task_by_id.get(decision.task_id)
        if task is None:
            raise ValueError(
                f"route decision references unknown task {decision.task_id!r}"
            )
        route_rows.append(
            {
                "task_id": decision.task_id,
                "question": task.question,
                "source": decision.source.value,
                "reason": decision.reason,
            }
        )
    routes_json = json.dumps(route_rows, ensure_ascii=False, indent=2)
    vision_json = json.dumps(
        {
            task_id: asdict(result)
            for task_id, result in state.vision_results.items()
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "Original user request:\n"
        f"{state.query}\n\n"
        "Research Plan (structured JSON):\n"
        f"{plan_json}\n\n"
        "Route Decisions (control information, NOT Evidence):\n"
        f"{routes_json}\n\n"
        "Vision Retrieval Results (retrieved material, NOT Evidence):\n"
        f"{vision_json}\n\n"
        "Execution instructions:\n"
        "- Follow the research plan and address its tasks in dependency order.\n"
        "- Task descriptions are planning instructions, not factual evidence.\n"
        "- Route Decisions are control information, not factual evidence.\n"
        "- For local tasks, use existing local capabilities such as "
        "search_knowledge_base and ground the answer in retrieved evidence.\n"
        "- Web Search results are retrieved material, not verified Evidence.\n"
        "- Vision Retrieval results are retrieved material, not verified Evidence.\n"
        "- Use each Vision result only for the task ID that owns it.\n"
        "- Do not use model memory or Local RAG to pretend an unavailable "
        "source was executed.\n"
        "- If the required source cannot be executed, state that the evidence "
        "is insufficient or the required capability is unavailable."
    )
