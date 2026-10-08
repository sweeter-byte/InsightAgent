"""Tests for the thin handoff from planning to the existing agent loop."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot

from insight_agent.planning import (
    ResearchCoordinator,
    ResearchPlan,
    ResearchState,
    ResearchTask,
    ResearchThreadSnapshot,
    format_research_context,
)
from insight_agent.research import UnknownResearchThread
from insight_agent.routing import RetrievalSource, RouteDecision
from insight_agent.vision_retrieval import (
    VisionAnalysis,
    VisionFailure,
    VisionRetrievalResult,
)


class FakePlanner:
    def __init__(self, plan: ResearchPlan) -> None:
        self.result = plan
        self.calls: list[str] = []

    def plan(self, query: str) -> ResearchPlan:
        self.calls.append(query)
        return self.result


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls: list[ResearchState] = []
        self.configs: list[RunnableConfig | None] = []
        self.resume_calls: list[RunnableConfig] = []
        self.snapshots: dict[str, StateSnapshot] = {}
        self.resume_results: dict[str, ResearchState] = {}

    def run(
        self,
        state: ResearchState,
        *,
        config: RunnableConfig | None = None,
    ) -> ResearchState:
        self.calls.append(state)
        self.configs.append(config)
        assert state.plan is not None
        decisions = [
            RouteDecision(
                task_id=task.id,
                source=(
                    RetrievalSource.LOCAL
                    if index == 0
                    else RetrievalSource.WEB
                ),
                reason=f"Route reason for {task.id}",
            )
            for index, task in enumerate(state.plan.tasks)
        ]
        final_state = ResearchState(
            query=state.query,
            plan=state.plan,
            available_sources=set(state.available_sources),
            task_index=len(state.plan.tasks),
            route_decisions=decisions,
            final_output="# 研究结果\n",
        )
        if config is not None:
            thread_id = str(config["configurable"]["thread_id"])
            self.snapshots[thread_id] = self._snapshot(
                final_state,
                config=config,
            )
        return final_state

    def get_state(self, config: RunnableConfig) -> StateSnapshot:
        thread_id = str(config["configurable"]["thread_id"])
        return self.snapshots.get(
            thread_id,
            StateSnapshot({}, (), config, None, None, None, (), ()),
        )

    @staticmethod
    def state_from_snapshot(snapshot: StateSnapshot) -> ResearchState:
        assert isinstance(snapshot.values, ResearchState)
        return snapshot.values

    def resume(self, config: RunnableConfig) -> ResearchState:
        self.resume_calls.append(config)
        thread_id = str(config["configurable"]["thread_id"])
        state = self.resume_results[thread_id]
        self.snapshots[thread_id] = self._snapshot(state, config=config)
        return state

    @staticmethod
    def _snapshot(
        state: ResearchState,
        *,
        config: RunnableConfig,
        next_nodes: tuple[str, ...] = (),
    ) -> StateSnapshot:
        return StateSnapshot(
            state,
            next_nodes,
            config,
            {"source": "loop", "step": 1, "parents": {}},
            datetime.now(UTC).isoformat(),
            None,
            (),
            (),
        )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Compare Agent Memory designs",
        constraints=["Compare mechanisms and trade-offs"],
        tasks=[
            ResearchTask(
                id="T1",
                question="What are the main design families?",
                depends_on=[],
            ),
            ResearchTask(
                id="T2",
                question="How do the design families compare?",
                depends_on=["T1"],
            ),
        ],
    )


def test_coordinator_saves_state_and_returns_workflow_final_output() -> None:
    plan = _plan()
    planner = FakePlanner(plan)
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL, RetrievalSource.WEB},
    )

    answer = coordinator.run("研究 Agent Memory")

    assert answer == "# 研究结果\n"
    assert planner.calls == ["研究 Agent Memory"]
    assert len(workflow.calls) == 1
    initial_state = workflow.calls[0]
    assert initial_state.query == "研究 Agent Memory"
    assert initial_state.plan is plan
    assert initial_state.available_sources == {
        RetrievalSource.LOCAL,
        RetrievalSource.WEB,
    }
    assert coordinator.last_state is not initial_state
    assert coordinator.last_state is not None
    assert coordinator.last_state.task_index == len(plan.tasks)
    assert [decision.task_id for decision in coordinator.last_state.route_decisions] == [
        "T1",
        "T2",
    ]


def test_start_research_returns_isolated_threads_for_same_query() -> None:
    ids: Iterator[str] = iter(["thread-a", "thread-b"])
    planner = FakePlanner(_plan())
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL},
        thread_id_factory=lambda: next(ids),
    )

    first = coordinator.start_research("same query")
    second = coordinator.start_research("same query")

    assert isinstance(first, ResearchThreadSnapshot)
    assert first.thread_id == "thread-a"
    assert second.thread_id == "thread-b"
    assert first.completed is True
    assert second.completed is True
    assert first.state.query == second.state.query == "same query"
    assert planner.calls == ["same query", "same query"]
    assert workflow.configs == [
        {"configurable": {"thread_id": "thread-a"}},
        {"configurable": {"thread_id": "thread-b"}},
    ]


def test_get_research_state_reads_without_executing() -> None:
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=FakePlanner(_plan()),  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL},
        thread_id_factory=lambda: "thread-a",
    )
    started = coordinator.start_research("query")
    call_count = len(workflow.calls)

    loaded = coordinator.get_research_state("thread-a")

    assert loaded == started
    assert len(workflow.calls) == call_count


@pytest.mark.parametrize("operation", ["get", "resume"])
def test_unknown_research_thread_is_explicit(operation: str) -> None:
    coordinator = ResearchCoordinator(
        planner=FakePlanner(_plan()),  # type: ignore[arg-type]
        workflow=FakeWorkflow(),  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL},
    )

    with pytest.raises(UnknownResearchThread, match="does-not-exist"):
        if operation == "get":
            coordinator.get_research_state("does-not-exist")
        else:
            coordinator.resume_research("does-not-exist")


def test_resume_completed_thread_returns_without_graph_execution() -> None:
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=FakePlanner(_plan()),  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL},
        thread_id_factory=lambda: "complete",
    )
    started = coordinator.start_research("query")

    resumed = coordinator.resume_research("complete")

    assert resumed == started
    assert workflow.resume_calls == []


def test_resume_pending_thread_does_not_replan() -> None:
    workflow = FakeWorkflow()
    planner = FakePlanner(_plan())
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL},
    )
    config: RunnableConfig = {"configurable": {"thread_id": "pending"}}
    pending_state = ResearchState(
        query="query",
        plan=_plan(),
        available_sources={RetrievalSource.LOCAL},
        task_index=1,
    )
    completed_state = ResearchState(
        query="query",
        plan=pending_state.plan,
        available_sources={RetrievalSource.LOCAL},
        task_index=2,
        final_output="# restored\n",
    )
    workflow.snapshots["pending"] = workflow._snapshot(
        pending_state,
        config=config,
        next_nodes=("generate_report",),
    )
    workflow.resume_results["pending"] = completed_state

    resumed = coordinator.resume_research("pending")

    assert planner.calls == []
    assert workflow.resume_calls == [config]
    assert resumed.completed is True
    assert resumed.state is completed_state


def test_execution_context_preserves_query_and_structured_plan() -> None:
    state = ResearchState(
        query="研究 Agent Memory",
        plan=_plan(),
        available_sources={RetrievalSource.LOCAL, RetrievalSource.WEB},
    )

    context = format_research_context(FakeWorkflow().run(state))
    assert "研究 Agent Memory" in context
    assert '"objective": "Compare Agent Memory designs"' in context
    assert '"id": "T1"' in context
    assert '"depends_on": [\n        "T1"\n      ]' in context
    assert "Follow the research plan" in context
    assert "Task descriptions are planning instructions, not factual evidence" in context
    assert "search_knowledge_base" in context
    assert "Route Decisions (control information, NOT Evidence)" in context
    assert '"task_id": "T1"' in context
    assert '"question": "What are the main design families?"' in context
    assert '"source": "local"' in context
    assert '"source": "web"' in context
    assert '"reason": "Route reason for T2"' in context
    assert "Web Search results are retrieved material, not verified Evidence" in context
    assert "Vision Retrieval results are retrieved material, not verified Evidence" in context
    assert "model memory or Local RAG" in context
    assert "evidence is insufficient or the required capability is unavailable" in context


def test_context_formatter_requires_a_completed_plan() -> None:
    state = ResearchState(query="query")

    try:
        format_research_context(state)
    except ValueError as exc:
        assert "plan" in str(exc)
    else:
        raise AssertionError("formatting an unplanned state must fail")


def test_execution_context_includes_task_scoped_vision_retrieval_material() -> None:
    state = ResearchState(
        query="inspect diagram",
        plan=_plan(),
        available_sources={RetrievalSource.VISION},
        vision_results={
            "T1": VisionRetrievalResult(
                task_id="T1",
                query="What are the main design families?",
                analyses=[
                    VisionAnalysis(
                        source="workflow.png",
                        content="UNIQUE_VISION_FACT: three arrows converge",
                        metadata={"kind": "diagram"},
                    )
                ],
                failures=[
                    VisionFailure(
                        source="deleted.png",
                        reason="Original image is not accessible",
                    )
                ],
            )
        },
    )

    context = format_research_context(state)

    assert "Vision Retrieval Results (retrieved material, NOT Evidence)" in context
    assert "UNIQUE_VISION_FACT: three arrows converge" in context
    assert '"source": "workflow.png"' in context
    assert '"source": "deleted.png"' in context
    assert '"no_candidates": false' in context


def test_coordinator_returns_only_after_planning_and_workflow() -> None:
    events: list[str] = []

    class OrderedPlanner(FakePlanner):
        def plan(self, query: str) -> ResearchPlan:
            events.append("planner")
            return super().plan(query)

    class OrderedWorkflow(FakeWorkflow):
        def run(
            self,
            state: ResearchState,
            *,
            config: RunnableConfig | None = None,
        ) -> ResearchState:
            events.append("workflow")
            return super().run(state, config=config)

    coordinator = ResearchCoordinator(
        planner=OrderedPlanner(_plan()),  # type: ignore[arg-type]
        workflow=OrderedWorkflow(),  # type: ignore[arg-type]
        available_sources=set(RetrievalSource),
    )

    coordinator.run("query")

    assert events == ["planner", "workflow"]
