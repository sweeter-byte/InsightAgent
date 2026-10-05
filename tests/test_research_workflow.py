"""Tests for the sequential LangGraph research-routing workflow."""

from __future__ import annotations

from typing import Any

import pytest

from insight_agent.evidence import EvidenceCollector
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.research.workflow import ResearchRoutingWorkflow
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource, RouteDecision, RoutingError
from insight_agent.ingestion import Document, SourceType
from insight_agent.web_search import (
    WebRetrievalResult,
    WebSearchConfigurationError,
    WebSearchHit,
)
from insight_agent.vision_retrieval import (
    VisionAnalysis,
    VisionRetrievalError,
    VisionRetrievalResult,
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


class FakeLocalRetriever:
    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self.results = (
            [
                RetrievalResult(
                    chunk_id="chunk-local",
                    score=0.8,
                    content="Local content",
                    document_id="document-local",
                    source="notes/local.md",
                    source_type=SourceType.MARKDOWN,
                    chunk_index=0,
                    start_char=0,
                    end_char=13,
                    metadata={"section": "local"},
                )
            ]
            if results is None
            else results
        )
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return self.results


class FakeVisionRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        *,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
    ) -> VisionRetrievalResult:
        self.calls.append(
            {
                "task": task,
                "objective": objective,
                "constraints": list(constraints),
            }
        )
        return VisionRetrievalResult(
            task_id=task.id,
            query=task.question,
            analyses=[
                VisionAnalysis(
                    source=f"{task.id}.png",
                    content=f"Visual analysis for {task.id}",
                    metadata={"task": task.id},
                )
            ],
            failures=[],
        )


class RecordingWorkflow(ResearchRoutingWorkflow):
    def __init__(
        self,
        router: FakeRouter,
        web_retriever: FakeWebRetriever | None = None,
        vision_retriever: FakeVisionRetriever | None = None,
        local_retriever: FakeLocalRetriever | None = None,
    ) -> None:
        self.entries: list[str] = []
        super().__init__(
            router=router,  # type: ignore[arg-type]
            local_retriever=local_retriever or FakeLocalRetriever(),
            web_retriever=web_retriever,
            vision_retriever=vision_retriever,
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


class OrderedWorkflow(ResearchRoutingWorkflow):
    def __init__(
        self,
        router: FakeRouter,
        *,
        local_retriever: FakeLocalRetriever | None = None,
        web_retriever: FakeWebRetriever | None = None,
        vision_retriever: FakeVisionRetriever | None = None,
    ) -> None:
        self.events: list[str] = []
        super().__init__(
            router=router,  # type: ignore[arg-type]
            local_retriever=local_retriever,
            web_retriever=web_retriever,
            vision_retriever=vision_retriever,
        )

    def local_entry(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("local")
        return super().local_entry(state)

    def web_entry(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("web")
        return super().web_entry(state)

    def vision_entry(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("vision")
        return super().vision_entry(state)

    def collect_evidence(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("collect_evidence")
        return super().collect_evidence(state)

    def advance_task(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("advance_task")
        return super().advance_task(state)


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
    assert state.local_results == {}
    assert state.web_results == {}
    assert state.vision_results == {}
    assert state.evidence_pool == {}


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
    vision_retriever = FakeVisionRetriever()
    workflow = RecordingWorkflow(router, web_retriever, vision_retriever)

    final_state = workflow.run(_state())

    assert workflow.entries == [source.value]
    assert final_state.route_decisions == [
        RouteDecision(task_id="T1", source=source, reason=f"Route T1 to {source.value}")
    ]
    assert web_retriever.calls == (["Question 1"] if source is RetrievalSource.WEB else [])
    assert len(vision_retriever.calls) == (
        1 if source is RetrievalSource.VISION else 0
    )


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


def test_local_branch_writes_task_scoped_raw_results() -> None:
    local_retriever = FakeLocalRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.LOCAL]),  # type: ignore[arg-type]
        local_retriever=local_retriever,
    )

    final_state = workflow.run(_state())

    assert local_retriever.calls == [("Question 1", None)]
    assert final_state.local_results == {"T1": local_retriever.results}
    assert final_state.web_results == {}
    assert final_state.vision_results == {}


def test_local_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.LOCAL])
    )

    with pytest.raises(RoutingError, match="Local Retrieval is not configured"):
        workflow.run(_state())


@pytest.mark.parametrize("source", list(RetrievalSource))
def test_each_source_collects_evidence_before_advancing(
    source: RetrievalSource,
) -> None:
    workflow = OrderedWorkflow(
        FakeRouter([source]),
        local_retriever=FakeLocalRetriever(),
        web_retriever=FakeWebRetriever(),
        vision_retriever=FakeVisionRetriever(),
    )

    final_state = workflow.run(_state())

    assert workflow.events == [source.value, "collect_evidence", "advance_task"]
    assert len(final_state.evidence_pool["T1"]) == 1
    evidence = final_state.evidence_pool["T1"][0]
    assert evidence.task_id == "T1"
    assert evidence.retrieval_source is source


def test_collect_evidence_deduplicates_an_existing_exact_id() -> None:
    local_result = FakeLocalRetriever().results[0]
    existing = EvidenceCollector().collect_local("T1", [local_result])[0]
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.current_route = RouteDecision(
        task_id="T1",
        source=RetrievalSource.LOCAL,
        reason="local",
    )
    state.local_results = {"T1": [local_result]}
    state.evidence_pool = {"T1": [existing]}
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([]),
        local_retriever=FakeLocalRetriever(),
    )

    update = workflow.collect_evidence(state)

    assert update["evidence_pool"] == {"T1": [existing]}


@pytest.mark.parametrize("source", list(RetrievalSource))
def test_empty_source_result_creates_an_empty_task_evidence_pool(
    source: RetrievalSource,
) -> None:
    class EmptyWebRetriever(FakeWebRetriever):
        def retrieve(self, query: str) -> WebRetrievalResult:
            self.calls.append(query)
            return WebRetrievalResult(
                query=query,
                hits=[],
                documents=[],
                failures=[],
            )

    class EmptyVisionRetriever(FakeVisionRetriever):
        def retrieve(self, **kwargs: Any) -> VisionRetrievalResult:
            self.calls.append(dict(kwargs))
            task = kwargs["task"]
            return VisionRetrievalResult(
                task_id=task.id,
                query=task.question,
                analyses=[],
                failures=[],
                no_candidates=True,
            )

    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([source]),  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever([]),
        web_retriever=EmptyWebRetriever(),
        vision_retriever=EmptyVisionRetriever(),
    )

    final_state = workflow.run(_state())

    assert final_state.evidence_pool == {"T1": []}


def test_vision_branch_writes_task_scoped_result_with_plan_context() -> None:
    vision_retriever = FakeVisionRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.VISION]),  # type: ignore[arg-type]
        vision_retriever=vision_retriever,
    )

    final_state = workflow.run(_state())

    assert len(vision_retriever.calls) == 1
    call = vision_retriever.calls[0]
    assert call["task"].id == "T1"
    assert call["objective"] == "Compare memory systems"
    assert call["constraints"] == ["Prefer primary sources"]
    assert set(final_state.vision_results) == {"T1"}
    assert final_state.vision_results["T1"].query == "Question 1"


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
        local_retriever=FakeLocalRetriever(),
        web_retriever=web_retriever,
        vision_retriever=(
            FakeVisionRetriever() if source is RetrievalSource.VISION else None
        ),
    )

    final_state = workflow.run(_state())

    assert web_retriever.calls == []
    assert final_state.web_results == {}


@pytest.mark.parametrize("source", [RetrievalSource.LOCAL, RetrievalSource.WEB])
def test_non_vision_branches_do_not_call_vision_retriever(
    source: RetrievalSource,
) -> None:
    vision_retriever = FakeVisionRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([source]),  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        web_retriever=FakeWebRetriever() if source is RetrievalSource.WEB else None,
        vision_retriever=vision_retriever,
    )

    final_state = workflow.run(_state())

    assert vision_retriever.calls == []
    assert final_state.vision_results == {}


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


def test_multiple_vision_tasks_accumulate_results_and_reach_end() -> None:
    vision_retriever = FakeVisionRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.VISION, RetrievalSource.VISION]),  # type: ignore[arg-type]
        vision_retriever=vision_retriever,
    )

    final_state = workflow.run(_state(task_count=2))

    assert list(final_state.vision_results) == ["T1", "T2"]
    assert final_state.vision_results["T1"].query == "Question 1"
    assert final_state.vision_results["T2"].query == "Question 2"
    assert final_state.task_index == 2


def test_web_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.WEB])
    )

    with pytest.raises(WebSearchConfigurationError, match="not configured"):
        workflow.run(_state())


def test_vision_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.VISION])
    )

    with pytest.raises(VisionRetrievalError, match="not configured"):
        workflow.run(_state())


@pytest.mark.parametrize(
    ("task_id", "query", "message"),
    [
        ("T99", "Question 1", "task_id"),
        ("T1", "Different question", "query"),
    ],
)
def test_vision_branch_rejects_result_for_a_different_task(
    task_id: str,
    query: str,
    message: str,
) -> None:
    class MismatchedVisionRetriever(FakeVisionRetriever):
        def retrieve(self, **kwargs: Any) -> VisionRetrievalResult:
            return VisionRetrievalResult(
                task_id=task_id,
                query=query,
                analyses=[],
                failures=[],
                no_candidates=True,
            )

    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.VISION]),  # type: ignore[arg-type]
        vision_retriever=MismatchedVisionRetriever(),
    )

    with pytest.raises(RoutingError, match=message):
        workflow.run(_state())


def test_one_task_produces_exactly_one_decision_and_ends_cleanly() -> None:
    router = FakeRouter([RetrievalSource.LOCAL])
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
    )

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
    workflow = RecordingWorkflow(
        router,
        FakeWebRetriever(),
        FakeVisionRetriever(),
    )
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
