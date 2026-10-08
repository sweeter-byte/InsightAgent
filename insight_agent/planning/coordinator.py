"""Thin coordination from a research query to its rendered workflow report."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot

from insight_agent.planning.models import ResearchPlan, ResearchState
from insight_agent.routing.models import RetrievalSource


class Planner(Protocol):
    def plan(self, query: str) -> ResearchPlan:
        """Return a validated plan for one query."""
        ...


class UnknownResearchThread(LookupError):
    """Raised when no checkpoint exists for a requested research thread."""


def _thread_config(thread_id: str) -> RunnableConfig:
    """Load research infrastructure only after package initialization."""
    from insight_agent.research.persistence import thread_config

    return thread_config(thread_id)


class ResearchWorkflow(Protocol):
    def run(
        self,
        state: ResearchState,
        *,
        config: RunnableConfig | None = None,
    ) -> ResearchState:
        """Route every planned task and return the final workflow state."""
        ...

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        """Return the latest checkpoint for a logical research thread."""
        ...

    def state_from_snapshot(self, snapshot: StateSnapshot) -> ResearchState:
        """Convert persisted graph values into canonical research state."""
        ...

    def resume(self, config: RunnableConfig) -> ResearchState:
        """Continue a pending logical research thread."""
        ...


@dataclass(frozen=True, slots=True)
class ResearchThreadSnapshot:
    """Read-only business view of one persisted research thread."""

    thread_id: str
    state: ResearchState
    next_nodes: tuple[str, ...]

    @property
    def completed(self) -> bool:
        """Whether the graph has no pending nodes."""
        return not self.next_nodes


class ResearchCoordinator:
    """Plan a research query, retain its state, and return its rendered report."""

    def __init__(
        self,
        planner: Planner,
        workflow: ResearchWorkflow,
        available_sources: set[RetrievalSource],
        *,
        thread_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not available_sources:
            raise ValueError("available_sources must not be empty")
        if any(not isinstance(source, RetrievalSource) for source in available_sources):
            raise ValueError("available_sources must contain RetrievalSource values")
        self.planner = planner
        self.workflow = workflow
        self.available_sources = set(available_sources)
        self._thread_id_factory = thread_id_factory or (lambda: str(uuid4()))
        self.last_state: ResearchState | None = None

    def run(self, query: str) -> str:
        """Run a new research thread and preserve the existing string API."""
        snapshot = self.start_research(query)
        if not snapshot.completed:
            raise ValueError("research workflow did not complete")
        if not isinstance(snapshot.state.final_output, str):
            raise ValueError("research workflow did not produce final_output")
        return snapshot.state.final_output

    def start_research(self, query: str) -> ResearchThreadSnapshot:
        """Create and execute a new independently checkpointed research thread."""
        thread_id = self._thread_id_factory()
        config = _thread_config(thread_id)
        state = ResearchState(
            query=query,
            plan=self.planner.plan(query),
            available_sources=set(self.available_sources),
        )
        self.workflow.run(state, config=config)
        snapshot = self._load_thread(thread_id)
        self.last_state = snapshot.state
        return snapshot

    def get_research_state(self, thread_id: str) -> ResearchThreadSnapshot:
        """Read a thread without executing or creating workflow state."""
        return self._load_thread(thread_id)

    def resume_research(self, thread_id: str) -> ResearchThreadSnapshot:
        """Continue pending nodes, or return an already-completed thread."""
        snapshot = self._load_thread(thread_id)
        if snapshot.completed:
            self.last_state = snapshot.state
            return snapshot
        self.workflow.resume(_thread_config(thread_id))
        resumed = self._load_thread(thread_id)
        self.last_state = resumed.state
        return resumed

    def _load_thread(self, thread_id: str) -> ResearchThreadSnapshot:
        config = _thread_config(thread_id)
        snapshot = self.workflow.get_state(config)
        if snapshot.metadata is None:
            raise UnknownResearchThread(
                f"unknown research thread: {thread_id!r}"
            )
        return ResearchThreadSnapshot(
            thread_id=thread_id,
            state=self.workflow.state_from_snapshot(snapshot),
            next_nodes=tuple(snapshot.next),
        )


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
