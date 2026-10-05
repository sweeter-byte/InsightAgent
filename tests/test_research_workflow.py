"""Tests for the sequential LangGraph research-routing workflow."""

from __future__ import annotations

from typing import Any

import pytest

from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.research.workflow import ResearchRoutingWorkflow
from insight_agent.routing import RetrievalSource, RouteDecision, RoutingError
from insight_agent.ingestion import Document, SourceType
from insight_agent.web_search import (
    WebRetrievalResult,
    WebSearchConfigurationError,
    WebSearchHit,
)


def _plan(task_count: int = 1) -> ResearchPlan:
    return ResearchPlan(
        objective="Compare memory systems",
        constraints=["Prefer primary sources"],
        tasks=[
            ResearchTask(
                id=f"T{index}",
                question=f"Question {index}",
                depends_on=[] if index == 1 else [f"T{index - 1}"],
            )
            for index in range(1, task_count + 1)
        ],
    )


class FakeRouter:
    def __init__(self, sources: list[RetrievalSource]) -> None:
        self.sources = list(sources)
        self.calls: list[dict[str, Any]] = []

    def route(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        available_sources: set[RetrievalSource],
    ) -> RouteDecision:
        self.calls.append(
            {
                "task": task,
                "objective": objective,
                "constraints": list(constraints),
                "available_sources": set(available_sources),
            }
        )
        if not self.sources:
            raise AssertionError("FakeRouter ran out of scripted sources")
        source = self.sources.pop(0)
        return RouteDecision(
            task_id=task.id,
            source=source,
            reason=f"Route {task.id} to {source.value}",
        )


class FakeWebRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str) -> WebRetrievalResult:
        self.calls.append(query)
        rank = len(self.calls)
        url = f"https://example.com/{rank}"
        return WebRetrievalResult(
            query=query,
            hits=[
                WebSearchHit(
                    rank=1,
                    title=f"Result {rank}",
                    url=url,
                    snippet="snippet",
                )
            ],
            documents=[
                Document(
                    content=f"Fetched content {rank}",
                    source=url,
                    source_type=SourceType.URL,
                    metadata={"final_url": url},
                )
            ],
            failures=[],
        )


class RecordingWorkflow(ResearchRoutingWorkflow):
    def __init__(
        self,
        router: FakeRouter,
        web_retriever: FakeWebRetriever | None = None,
    ) -> None:
        self.entries: list[str] = []
        super().__init__(
            router=router,  # type: ignore[arg-type]
            web_retriever=web_retriever,
        )

    def local_entry(self, state: ResearchState) -> dict[str, Any]:
        self.entries.append("local")
        return super().local_entry(state)

    def web_entry(self, state: ResearchState) -> dict[str, Any]:
        self.entries.append("web")
        return super().web_entry(state)

    def vision_entry(self, state: ResearchState) -> dict[str, Any]:
        self.entries.append("vision")
        return super().vision_entry(state)


def _state(task_count: int = 1) -> ResearchState:
    return ResearchState(
        query="research memory systems",
        plan=_plan(task_count),
        available_sources=set(RetrievalSource),
    )


def test_research_state_keeps_compatible_routing_defaults() -> None:
    state = ResearchState(query="query")

    assert state.plan is None
    assert state.available_sources == set()
    assert state.task_index == 0
    assert state.current_task is None
    assert state.current_route is None
    assert state.route_decisions == []
    assert state.web_results == {}


def test_select_task_uses_index_and_clears_previous_route() -> None:
    router = FakeRouter([RetrievalSource.LOCAL])
    workflow = ResearchRoutingWorkflow(router=router)  # type: ignore[arg-type]
    state = _state(task_count=2)
    state.task_index = 1
    state.current_route = RouteDecision(
        task_id="T1",
        source=RetrievalSource.LOCAL,
        reason="old",
    )

    update = workflow.select_task(state)

    assert update == {"current_task": state.plan.tasks[1], "current_route": None}


@pytest.mark.parametrize("source", list(RetrievalSource))
def test_source_route_enters_matching_branch(source: RetrievalSource) -> None:
    router = FakeRouter([source])
    web_retriever = FakeWebRetriever()
    workflow = RecordingWorkflow(router, web_retriever)

    final_state = workflow.run(_state())

    assert workflow.entries == [source.value]
    assert final_state.route_decisions == [
        RouteDecision(task_id="T1", source=source, reason=f"Route T1 to {source.value}")
    ]
    assert web_retriever.calls == (["Question 1"] if source is RetrievalSource.WEB else [])


def test_web_branch_writes_task_scoped_result() -> None:
    web_retriever = FakeWebRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.WEB]),  # type: ignore[arg-type]
        web_retriever=web_retriever,
    )

    final_state = workflow.run(_state())

    assert web_retriever.calls == ["Question 1"]
    assert set(final_state.web_results) == {"T1"}
    assert final_state.web_results["T1"].query == "Question 1"
    assert final_state.web_results["T1"].documents[0].content == "Fetched content 1"


@pytest.mark.parametrize(
    "source",
    [RetrievalSource.LOCAL, RetrievalSource.VISION],
)
def test_non_web_branches_do_not_call_web_retriever(
    source: RetrievalSource,
) -> None:
    web_retriever = FakeWebRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([source]),  # type: ignore[arg-type]
        web_retriever=web_retriever,
    )

    final_state = workflow.run(_state())

    assert web_retriever.calls == []
    assert final_state.web_results == {}


def test_multiple_web_tasks_accumulate_results_and_reach_end() -> None:
    web_retriever = FakeWebRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.WEB, RetrievalSource.WEB]),  # type: ignore[arg-type]
        web_retriever=web_retriever,
    )

    final_state = workflow.run(_state(task_count=2))

    assert web_retriever.calls == ["Question 1", "Question 2"]
    assert list(final_state.web_results) == ["T1", "T2"]
    assert final_state.web_results["T1"].query == "Question 1"
    assert final_state.web_results["T2"].query == "Question 2"
    assert final_state.task_index == 2
    assert final_state.current_task is None


def test_web_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.WEB])
    )

    with pytest.raises(WebSearchConfigurationError, match="not configured"):
        workflow.run(_state())


def test_one_task_produces_exactly_one_decision_and_ends_cleanly() -> None:
    router = FakeRouter([RetrievalSource.LOCAL])
    workflow = ResearchRoutingWorkflow(router=router)  # type: ignore[arg-type]

    final_state = workflow.run(_state())

    assert len(router.calls) == 1
    assert len(final_state.route_decisions) == 1
    assert final_state.task_index == 1
    assert final_state.current_task is None
    assert final_state.current_route is None


def test_multiple_tasks_are_routed_in_plan_order() -> None:
    sources = [
        RetrievalSource.LOCAL,
        RetrievalSource.WEB,
        RetrievalSource.VISION,
    ]
    router = FakeRouter(sources)
    workflow = RecordingWorkflow(router, FakeWebRetriever())
    state = _state(task_count=3)

    final_state = workflow.run(state)

    assert [call["task"].id for call in router.calls] == ["T1", "T2", "T3"]
    assert all(call["objective"] == state.plan.objective for call in router.calls)
    assert all(call["constraints"] == state.plan.constraints for call in router.calls)
    assert all(
        call["available_sources"] == set(RetrievalSource) for call in router.calls
    )
    assert workflow.entries == ["local", "web", "vision"]
    assert [decision.task_id for decision in final_state.route_decisions] == [
        "T1",
        "T2",
        "T3",
    ]
    assert [decision.source for decision in final_state.route_decisions] == sources
    assert final_state.task_index == len(state.plan.tasks)


def test_workflow_rejects_state_without_plan() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.LOCAL])
    )

    with pytest.raises(RoutingError, match="plan"):
        workflow.run(
            ResearchState(
                query="query",
                available_sources={RetrievalSource.LOCAL},
            )
        )


def test_workflow_rejects_empty_available_sources() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.LOCAL])
    )

    with pytest.raises(RoutingError, match="available_sources"):
        workflow.run(ResearchState(query="query", plan=_plan()))


def test_workflow_rejects_router_decision_outside_available_sources() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.WEB])
    )
    state = ResearchState(
        query="query",
        plan=_plan(),
        available_sources={RetrievalSource.LOCAL},
    )

    with pytest.raises(RoutingError, match="not available"):
        workflow.run(state)


def test_select_task_reports_out_of_range_index() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.LOCAL])
    )
    state = _state()
    state.task_index = 1

    with pytest.raises(RoutingError, match="task_index"):
        workflow.select_task(state)
