"""Tests for the sequential LangGraph research-routing workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from insight_agent.evidence import (
    Evidence,
    EvidenceAssessment,
    EvidenceCollector,
    EvidenceCoverage,
    EvidenceJudgment,
    EvidenceGrader,
    EvidenceQuality,
    EvidenceRelevance,
)
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import Claim, StructuredReport
from insight_agent.research.workflow import (
    ResearchRoutingWorkflow as BaseResearchRoutingWorkflow,
)
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource, RouteDecision, RoutingError
from insight_agent.self_check import (
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)
from insight_agent.ingestion import Document, SourceType
from insight_agent.web_search import (
    TavilySearchProvider,
    WebRetrievalResult,
    WebRetriever,
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
        retrieval_query: str,
        missing_information: list[str],
    ) -> RouteDecision:
        self.calls.append(
            {
                "task": task,
                "objective": objective,
                "constraints": list(constraints),
                "available_sources": set(available_sources),
                "retrieval_query": retrieval_query,
                "missing_information": list(missing_information),
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


class FakeEvidenceGrader:
    def __init__(
        self,
        assessments: list[EvidenceAssessment] | None = None,
    ) -> None:
        self.assessments = list(assessments or [])
        self.calls: list[dict[str, Any]] = []

    def grade(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        evidence: list[Evidence],
    ) -> EvidenceAssessment:
        self.calls.append(
            {
                "task": task,
                "objective": objective,
                "constraints": list(constraints),
                "evidence": list(evidence),
            }
        )
        if self.assessments:
            assessment = self.assessments.pop(0)
        elif evidence:
            assessment = EvidenceAssessment(
                task_id=task.id,
                evidence_judgments=[],
                coverage=EvidenceCoverage.COMPLETE,
                sufficient=True,
                missing_information=[],
                reason="Enough for this workflow test.",
            )
        else:
            assessment = EvidenceAssessment(
                task_id=task.id,
                evidence_judgments=[],
                coverage=EvidenceCoverage.INSUFFICIENT,
                sufficient=False,
                missing_information=["No Evidence was retrieved."],
                reason="No Evidence is available.",
            )
        if evidence and not assessment.evidence_judgments:
            assessment.evidence_judgments = [
                EvidenceJudgment(
                    evidence_id=item.id,
                    relevance=EvidenceRelevance.RELEVANT,
                    quality=EvidenceQuality.STRONG,
                    reason="Usable workflow fixture.",
                )
                for item in evidence
            ]
        return assessment

    def generate_section(self, **kwargs: Any) -> list[Claim]:
        task = kwargs["task"]
        evidence = kwargs["evidence"]
        return [
            Claim(
                id=f"{task.id}-C1",
                task_id=task.id,
                text=f"Report result for {task.id}.",
                evidence_ids=[item.id for item in evidence],
            )
        ]


class AlwaysPassSelfChecker:
    def check(self, **kwargs: Any) -> SelfCheckResult:
        return SelfCheckResult(
            status=SelfCheckStatus.PASS,
            issues=[],
            summary="Offline workflow fixture passes.",
        )


class NeverReportRepairer:
    def repair(self, **kwargs: Any) -> StructuredReport:
        raise AssertionError("Passing workflow fixture must not repair")


def _assessment(
    *,
    task_id: str = "T1",
    sufficient: bool,
    missing_information: list[str] | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        task_id=task_id,
        evidence_judgments=[],
        coverage=(
            EvidenceCoverage.COMPLETE
            if sufficient
            else EvidenceCoverage.PARTIAL
        ),
        sufficient=sufficient,
        missing_information=list(missing_information or []),
        reason="Sufficient." if sufficient else "More evidence is needed.",
    )


class ResearchRoutingWorkflow(BaseResearchRoutingWorkflow):
    """Test adapter that explicitly injects the offline report fake."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("report_generator", kwargs["grader"])
        kwargs.setdefault("self_checker", AlwaysPassSelfChecker())
        kwargs.setdefault("report_repairer", NeverReportRepairer())
        super().__init__(*args, **kwargs)


class FakeWebRetriever:
    def __init__(self, content: str | None = None) -> None:
        self.calls: list[str] = []
        self.content = content

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
                    content=self.content or f"Fetched content {rank}",
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
        query: str | None = None,
        objective: str,
        constraints: list[str],
    ) -> VisionRetrievalResult:
        self.calls.append(
            {
                "task": task,
                "query": query,
                "objective": objective,
                "constraints": list(constraints),
            }
        )
        return VisionRetrievalResult(
            task_id=task.id,
            query=query or task.question,
            analyses=[
                VisionAnalysis(
                    source=f"{task.id}.png",
                    content=f"Visual analysis for {task.id}",
                    metadata={"task": task.id},
                )
            ],
            failures=[],
        )


def test_graph_state_round_trip_preserves_self_check_fields() -> None:
    state = _state()
    state.self_check_result = SelfCheckResult(
        status=SelfCheckStatus.REVISE,
        issues=[
            SelfCheckIssue(
                code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
                reason="Claim needs repair.",
                task_id="T1",
                claim_id="T1-C1",
            )
        ],
        summary="Repair required.",
    )
    state.self_check_rounds = 1

    restored = BaseResearchRoutingWorkflow._from_graph_state(
        BaseResearchRoutingWorkflow._to_graph_state(state)
    )

    assert restored.self_check_result == state.self_check_result
    assert restored.self_check_rounds == 1


class RecordingWorkflow(ResearchRoutingWorkflow):
    def __init__(
        self,
        router: FakeRouter,
        web_retriever: FakeWebRetriever | None = None,
        vision_retriever: FakeVisionRetriever | None = None,
        local_retriever: FakeLocalRetriever | None = None,
        grader: FakeEvidenceGrader | None = None,
    ) -> None:
        self.entries: list[str] = []
        super().__init__(
            router=router,  # type: ignore[arg-type]
            local_retriever=local_retriever or FakeLocalRetriever(),
            web_retriever=web_retriever,
            vision_retriever=vision_retriever,
            grader=grader or FakeEvidenceGrader(),
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
        grader: FakeEvidenceGrader | None = None,
    ) -> None:
        self.events: list[str] = []
        super().__init__(
            router=router,  # type: ignore[arg-type]
            local_retriever=local_retriever,
            web_retriever=web_retriever,
            vision_retriever=vision_retriever,
            grader=grader or FakeEvidenceGrader(),
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

    def grade_evidence(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("grade_evidence")
        return super().grade_evidence(state)

    def prepare_retry(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("prepare_retry")
        return super().prepare_retry(state)

    def advance_task(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("advance_task")
        return super().advance_task(state)

    def generate_report(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("generate_report")
        return super().generate_report(state)

    def self_check_report(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("self_check_report")
        return super().self_check_report(state)

    def finalize_report(self, state: ResearchState) -> dict[str, Any]:
        self.events.append("finalize_report")
        return super().finalize_report(state)


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
    assert state.evidence_assessments == {}
    assert state.retrieval_query is None


def test_select_task_uses_index_and_clears_previous_route() -> None:
    router = FakeRouter([RetrievalSource.LOCAL])
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )
    state = _state(task_count=2)
    state.task_index = 1
    state.current_route = RouteDecision(
        task_id="T1",
        source=RetrievalSource.LOCAL,
        reason="old",
    )

    update = workflow.select_task(state)

    assert update == {
        "current_task": state.plan.tasks[1],
        "current_route": None,
        "retrieval_query": state.plan.tasks[1].question,
    }


def test_grade_evidence_appends_assessment_and_reads_entire_pool() -> None:
    grader = FakeEvidenceGrader([_assessment(sufficient=True)])
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([]),  # type: ignore[arg-type]
        grader=grader,  # type: ignore[arg-type]
    )
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.evidence_pool = {"T1": [EvidenceCollector().collect_local(
        "T1", FakeLocalRetriever().results
    )[0]]}
    previous = _assessment(
        sufficient=False,
        missing_information=["Earlier gap"],
    )
    state.evidence_assessments = {"T1": [previous]}

    update = workflow.grade_evidence(state)

    assert grader.calls[0]["evidence"] == state.evidence_pool["T1"]
    history = update["evidence_assessments"]["T1"]
    assert history[0] is previous
    assert history[-1].sufficient is True
    assert len(history[-1].evidence_judgments) == 1


@pytest.mark.parametrize(
    ("assessments", "expected"),
    [
        ([_assessment(sufficient=True)], "advance_task"),
        (
            [_assessment(sufficient=False, missing_information=["gap"])],
            "prepare_retry",
        ),
        (
            [
                _assessment(sufficient=False, missing_information=["gap 1"]),
                _assessment(sufficient=False, missing_information=["gap 2"]),
            ],
            "advance_task",
        ),
    ],
)
def test_choose_after_grading_is_sufficient_and_budget_aware(
    assessments: list[EvidenceAssessment],
    expected: str,
) -> None:
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([]),  # type: ignore[arg-type]
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.evidence_assessments = {"T1": assessments}

    assert workflow.choose_after_grading(state) == expected


def test_prepare_retry_builds_query_from_question_and_missing_information() -> None:
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([]),  # type: ignore[arg-type]
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.evidence_assessments = {
        "T1": [
            _assessment(
                sufficient=False,
                missing_information=["Missing limitation", "Missing cost"],
            )
        ]
    }

    update = workflow.prepare_retry(state)

    assert update["retrieval_query"] == (
        "Original task:\nQuestion 1\n\n"
        "Missing information:\n- Missing limitation\n- Missing cost"
    )


def test_advance_task_clears_transient_query() -> None:
    state = _state()
    state.retrieval_query = "focused query"

    assert ResearchRoutingWorkflow.advance_task(state)["retrieval_query"] is None


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
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert web_retriever.calls == ["Question 1"]
    assert set(final_state.web_results) == {"T1"}
    assert final_state.web_results["T1"].query == "Question 1"
    assert final_state.web_results["T1"].documents[0].content == "Fetched content 1"


def test_tavily_http_retry_does_not_add_a_research_round() -> None:
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(
        200,
        request=request,
        json={
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/result",
                    "content": "Search snippet",
                }
            ]
        },
    )
    delays: list[float] = []
    retriever = WebRetriever(
        TavilySearchProvider(api_key="tvly-test", sleeper=delays.append),
        url_loader=lambda url, *, timeout: [
            Document(
                content="Fetched content",
                source=url,
                source_type=SourceType.URL,
                metadata={"final_url": url},
            )
        ],
    )
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.WEB]),  # type: ignore[arg-type]
        web_retriever=retriever,
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        side_effect=[httpx.ConnectTimeout("connect timed out"), response],
    ) as post:
        final_state = workflow.run(_state())

    assert post.call_count == 2
    assert delays == [0.5]
    assert len(final_state.route_decisions) == 1
    assert len(final_state.evidence_assessments["T1"]) == 1


def test_grader_repair_does_not_add_a_research_round() -> None:
    class RepairingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[dict[str, Any]], tools: Any = None) -> Any:
            self.calls += 1
            evidence_ids = [
                item["evidence_id"]
                for item in json.loads(messages[1]["content"])["evidence"]
            ]
            judged_ids = [] if self.calls == 1 else evidence_ids
            payload = {
                "task_id": "T1",
                "evidence_judgments": [
                    {
                        "evidence_id": evidence_id,
                        "relevance": "relevant",
                        "quality": "strong",
                        "reason": "Direct support with provenance.",
                    }
                    for evidence_id in judged_ids
                ],
                "coverage": "complete",
                "sufficient": True,
                "missing_information": [],
                "reason": "The task is covered.",
            }
            message = SimpleNamespace(content=json.dumps(payload), tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = RepairingLLM()
    grader = EvidenceGrader(llm=llm)  # type: ignore[arg-type]
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.LOCAL]),  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=grader,
        report_generator=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert llm.calls == 2
    assert len(final_state.route_decisions) == 1
    assert len(final_state.evidence_assessments["T1"]) == 1


def test_local_branch_writes_task_scoped_raw_results() -> None:
    local_retriever = FakeLocalRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.LOCAL]),  # type: ignore[arg-type]
        local_retriever=local_retriever,
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert local_retriever.calls == [("Question 1", None)]
    assert final_state.local_results == {"T1": local_retriever.results}
    assert final_state.web_results == {}
    assert final_state.vision_results == {}


def test_local_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.LOCAL]),
        grader=FakeEvidenceGrader(),
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

    assert workflow.events == [
        source.value,
        "collect_evidence",
        "grade_evidence",
        "advance_task",
        "generate_report",
        "self_check_report",
        "finalize_report",
    ]
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
        grader=FakeEvidenceGrader(),
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
                query=kwargs["query"],
                analyses=[],
                failures=[],
                no_candidates=True,
            )

    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([source, source]),  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever([]),
        web_retriever=EmptyWebRetriever(),
        vision_retriever=EmptyVisionRetriever(),
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert final_state.evidence_pool == {"T1": []}


def test_vision_branch_writes_task_scoped_result_with_plan_context() -> None:
    vision_retriever = FakeVisionRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.VISION]),  # type: ignore[arg-type]
        vision_retriever=vision_retriever,
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
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
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
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
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert vision_retriever.calls == []
    assert final_state.vision_results == {}


def test_multiple_web_tasks_accumulate_results_and_reach_end() -> None:
    web_retriever = FakeWebRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([RetrievalSource.WEB, RetrievalSource.WEB]),  # type: ignore[arg-type]
        web_retriever=web_retriever,
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
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
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state(task_count=2))

    assert list(final_state.vision_results) == ["T1", "T2"]
    assert final_state.vision_results["T1"].query == "Question 1"
    assert final_state.vision_results["T2"].query == "Question 2"
    assert final_state.task_index == 2


def test_web_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.WEB]),
        grader=FakeEvidenceGrader(),
    )

    with pytest.raises(WebSearchConfigurationError, match="not configured"):
        workflow.run(_state())


def test_vision_branch_requires_configured_retriever() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.VISION]),
        grader=FakeEvidenceGrader(),
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
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    with pytest.raises(RoutingError, match=message):
        workflow.run(_state())


def test_one_task_produces_exactly_one_decision_and_ends_cleanly() -> None:
    router = FakeRouter([RetrievalSource.LOCAL])
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert len(router.calls) == 1
    assert len(final_state.route_decisions) == 1
    assert final_state.task_index == 1
    assert final_state.current_task is None
    assert final_state.current_route is None


def test_insufficient_evidence_retries_through_router_then_accumulates() -> None:
    router = FakeRouter([RetrievalSource.LOCAL, RetrievalSource.WEB])
    task_question = "比较向量记忆与结构化记忆的核心机制、主要优势和局限"
    grader = FakeEvidenceGrader(
        [
            _assessment(
                sufficient=False,
                missing_information=["缺少结构化记忆的主要局限与适用边界"],
            ),
            _assessment(sufficient=True),
        ]
    )
    local_retriever = FakeLocalRetriever(
        [
            RetrievalResult(
                chunk_id=f"chunk-{index}",
                score=0.9 - index / 10,
                content=content,
                document_id="memory-notes",
                source="notes/memory.md",
                source_type=SourceType.MARKDOWN,
                chunk_index=index,
                start_char=index * 100,
                end_char=index * 100 + len(content),
            )
            for index, content in enumerate(
                [
                    "向量记忆核心机制",
                    "结构化记忆核心机制",
                    "向量记忆优势",
                ],
                start=1,
            )
        ]
    )
    web_retriever = FakeWebRetriever("结构化记忆的局限与适用边界")
    workflow = OrderedWorkflow(
        router,
        local_retriever=local_retriever,
        web_retriever=web_retriever,
        grader=grader,
    )
    state = ResearchState(
        query=task_question,
        plan=ResearchPlan(
            objective="比较两类记忆方案",
            constraints=["覆盖机制、优势和局限"],
            tasks=[ResearchTask(id="T1", question=task_question)],
        ),
        available_sources=set(RetrievalSource),
    )

    final_state = workflow.run(state)

    retry_query = (
        f"Original task:\n{task_question}\n\n"
        "Missing information:\n- 缺少结构化记忆的主要局限与适用边界"
    )
    assert workflow.events == [
        "local",
        "collect_evidence",
        "grade_evidence",
        "prepare_retry",
        "web",
        "collect_evidence",
        "grade_evidence",
        "advance_task",
        "generate_report",
        "self_check_report",
        "finalize_report",
    ]
    assert [call["retrieval_query"] for call in router.calls] == [
        task_question,
        retry_query,
    ]
    assert router.calls[0]["missing_information"] == []
    assert router.calls[1]["missing_information"] == [
        "缺少结构化记忆的主要局限与适用边界"
    ]
    assert local_retriever.calls == [(task_question, None)]
    assert web_retriever.calls == [retry_query]
    assert [decision.source for decision in final_state.route_decisions] == [
        RetrievalSource.LOCAL,
        RetrievalSource.WEB,
    ]
    assert [item.content for item in final_state.evidence_pool["T1"]] == [
        "向量记忆核心机制",
        "结构化记忆核心机制",
        "向量记忆优势",
        "结构化记忆的局限与适用边界",
    ]
    assert [len(call["evidence"]) for call in grader.calls] == [3, 4]
    assert len(final_state.evidence_assessments["T1"]) == 2
    assert final_state.evidence_assessments["T1"][-1].sufficient is True


def test_budget_exhaustion_keeps_last_assessment_insufficient() -> None:
    router = FakeRouter([RetrievalSource.LOCAL, RetrievalSource.LOCAL])
    grader = FakeEvidenceGrader(
        [
            _assessment(sufficient=False, missing_information=["gap 1"]),
            _assessment(sufficient=False, missing_information=["gap 2"]),
        ]
    )
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=grader,  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state())

    assert len(router.calls) == 2
    assert len(final_state.route_decisions) == 2
    assert len(final_state.evidence_assessments["T1"]) == 2
    assert final_state.evidence_assessments["T1"][-1].sufficient is False
    assert final_state.evidence_assessments["T1"][-1].missing_information == [
        "gap 2"
    ]
    assert len(final_state.evidence_pool["T1"]) == 1
    assert final_state.task_index == 1


def test_new_task_resets_query_after_previous_task_retry() -> None:
    router = FakeRouter(
        [
            RetrievalSource.LOCAL,
            RetrievalSource.LOCAL,
            RetrievalSource.LOCAL,
        ]
    )
    grader = FakeEvidenceGrader(
        [
            _assessment(sufficient=False, missing_information=["T1 gap"]),
            _assessment(sufficient=True),
            _assessment(task_id="T2", sufficient=True),
        ]
    )
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=grader,  # type: ignore[arg-type]
    )

    final_state = workflow.run(_state(task_count=2))

    assert [call["retrieval_query"] for call in router.calls] == [
        "Question 1",
        "Original task:\nQuestion 1\n\nMissing information:\n- T1 gap",
        "Question 2",
    ]
    assert router.calls[-1]["missing_information"] == []
    assert [decision.task_id for decision in final_state.route_decisions] == [
        "T1",
        "T1",
        "T2",
    ]


@pytest.mark.parametrize("source", list(RetrievalSource))
def test_each_retriever_uses_current_retrieval_query(
    source: RetrievalSource,
) -> None:
    local = FakeLocalRetriever()
    web = FakeWebRetriever()
    vision = FakeVisionRetriever()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([]),  # type: ignore[arg-type]
        local_retriever=local,
        web_retriever=web,
        vision_retriever=vision,
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.current_route = RouteDecision(
        task_id="T1",
        source=source,
        reason="test current query",
    )
    state.retrieval_query = "Focused retry query"

    if source is RetrievalSource.LOCAL:
        workflow.local_entry(state)
        assert local.calls == [("Focused retry query", None)]
    elif source is RetrievalSource.WEB:
        workflow.web_entry(state)
        assert web.calls == ["Focused retry query"]
    else:
        workflow.vision_entry(state)
        assert vision.calls[0]["query"] == "Focused retry query"


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
        router=FakeRouter([RetrievalSource.LOCAL]),
        grader=FakeEvidenceGrader(),
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
        router=FakeRouter([RetrievalSource.LOCAL]),
        grader=FakeEvidenceGrader(),
    )

    with pytest.raises(RoutingError, match="available_sources"):
        workflow.run(ResearchState(query="query", plan=_plan()))


def test_workflow_rejects_router_decision_outside_available_sources() -> None:
    workflow = ResearchRoutingWorkflow(  # type: ignore[arg-type]
        router=FakeRouter([RetrievalSource.WEB]),
        grader=FakeEvidenceGrader(),
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
        router=FakeRouter([RetrievalSource.LOCAL]),
        grader=FakeEvidenceGrader(),
    )
    state = _state()
    state.task_index = 1

    with pytest.raises(RoutingError, match="task_index"):
        workflow.select_task(state)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_workflow_rejects_invalid_retrieval_round_budget(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ResearchRoutingWorkflow(
            router=FakeRouter([]),  # type: ignore[arg-type]
            grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
            max_retrieval_rounds=value,
        )


def test_custom_retrieval_round_budget_allows_a_third_round() -> None:
    router = FakeRouter([RetrievalSource.LOCAL] * 3)
    grader = FakeEvidenceGrader(
        [
            _assessment(sufficient=False, missing_information=["gap 1"]),
            _assessment(sufficient=False, missing_information=["gap 2"]),
            _assessment(sufficient=True),
        ]
    )
    workflow = ResearchRoutingWorkflow(
        router=router,  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=grader,  # type: ignore[arg-type]
        max_retrieval_rounds=3,
    )

    final_state = workflow.run(_state())

    assert len(final_state.route_decisions) == 3
    assert len(final_state.evidence_assessments["T1"]) == 3
    assert final_state.evidence_assessments["T1"][-1].sufficient is True


def test_route_history_must_match_completed_assessment_rounds() -> None:
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter([]),  # type: ignore[arg-type]
        local_retriever=FakeLocalRetriever(),
        grader=FakeEvidenceGrader(),  # type: ignore[arg-type]
    )
    state = _state()
    state.current_task = state.plan.tasks[0]
    state.current_route = RouteDecision(
        task_id="T1",
        source=RetrievalSource.LOCAL,
        reason="new round",
    )
    state.retrieval_query = "Question 1"
    state.route_decisions = [
        RouteDecision(
            task_id="T1",
            source=RetrievalSource.LOCAL,
            reason="ungraded prior round",
        )
    ]

    with pytest.raises(RoutingError, match="assessment history"):
        workflow.local_entry(state)
