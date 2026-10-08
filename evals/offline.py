"""Deterministic fixture composition for Offline Mock Tests."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evals.fixtures import (
    RecordedFixtureError,
    RecordedSearchProvider,
    RecordedUrlLoader,
    resolve_vision_fixture_paths,
)
from evals.judge import EvaluationJudge, JudgeRequest
from evals.models import EvalCase, JudgeLabel, JudgeResult, RuntimeMetrics
from evals.workflow import EvaluationWorkflowRunner, WorkflowCaseOutput
from insight_agent.evidence import EvidenceCollector
from insight_agent.ingestion import Document, SourceType
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import Claim, ReportSection, assemble_report
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource, RouteDecision
from insight_agent.vision_retrieval import VisionAnalysis, VisionRetrievalResult
from insight_agent.web_search import WebRetriever, WebSearchHit


class FixtureRetriever:
    """Retriever whose ordered results are fully recorded by exact query."""

    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = {
            query: list(results) for query, results in results_by_query.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RecordedFixtureError("top_k must be a positive integer")
        if query not in self.results_by_query:
            raise RecordedFixtureError(
                f"retrieval query is not recorded: {query!r}"
            )
        return list(self.results_by_query[query][:top_k])


class RecordedEvaluationJudge(EvaluationJudge):
    """Mock Judge with explicit per-Case results stored in the fixture."""

    def __init__(self, results_by_case: dict[str, JudgeResult]) -> None:
        self.results_by_case = dict(results_by_case)
        self.requests: list[JudgeRequest] = []

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        try:
            return self.results_by_case[request.case_id]
        except KeyError as exc:
            raise RecordedFixtureError(
                f"Judge result is not recorded for Case {request.case_id!r}"
            ) from exc


class FixtureWorkflowRunner(EvaluationWorkflowRunner):
    """Fake workflow boundary built from recorded Web and local Vision material."""

    def __init__(
        self,
        *,
        workflow_by_case: dict[str, dict[str, object]],
        web_retriever: WebRetriever,
        project_root: Path,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.workflow_by_case = workflow_by_case
        self.web_retriever = web_retriever
        self.project_root = project_root.resolve()
        self.clock = clock

    def run(self, case: EvalCase) -> WorkflowCaseOutput:
        try:
            fixture = self.workflow_by_case[case.case_id]
        except KeyError as exc:
            raise RecordedFixtureError(
                f"workflow output is not recorded for Case {case.case_id!r}"
            ) from exc
        source = _fixture_source(fixture)
        started = self.clock()
        task = ResearchTask("T1", case.query)
        plan = ResearchPlan(
            objective=case.query,
            constraints=list(case.constraints),
            tasks=[task],
        )
        state = ResearchState(
            query=case.query,
            plan=plan,
            available_sources={source},
            route_decisions=[RouteDecision("T1", source, "recorded fixture")],
        )
        artifacts: dict[str, object] = {}
        failed_calls = 0

        if source is RetrievalSource.WEB:
            web_result = self.web_retriever.retrieve(case.query)
            state.web_results["T1"] = web_result
            evidence = EvidenceCollector.collect_web("T1", web_result)
            failed_calls = len(web_result.failures)
            artifacts["web_result"] = web_result
        elif source is RetrievalSource.VISION:
            paths = resolve_vision_fixture_paths(
                case,
                fixture_root=self.project_root,
            )
            analysis = _required_text(fixture, "analysis")
            vision_result = VisionRetrievalResult(
                task_id="T1",
                query=case.query,
                analyses=[
                    VisionAnalysis(
                        source=str(path),
                        content=analysis,
                        metadata={"filename": path.name, "fixture": True},
                    )
                    for path in paths
                ],
                failures=[],
                no_candidates=not paths,
            )
            state.vision_results["T1"] = vision_result
            evidence = EvidenceCollector.collect_vision("T1", vision_result)
            artifacts["vision_result"] = vision_result
        else:
            raise RecordedFixtureError(
                "fixture workflow supports only recorded web or fixed vision sources"
            )

        state.evidence_pool["T1"] = evidence
        claims = []
        claim_text = fixture.get("claim_text")
        if claim_text is not None:
            if not isinstance(claim_text, str) or not claim_text.strip():
                raise RecordedFixtureError("claim_text must be a non-empty string")
            if not evidence:
                raise RecordedFixtureError("recorded Claim requires Evidence")
            claims.append(
                Claim(
                    id="T1-C1",
                    task_id="T1",
                    text=claim_text.strip(),
                    evidence_ids=[item.id for item in evidence],
                )
            )
        report = assemble_report(
            plan.objective,
            [
                ReportSection(
                    task_id="T1",
                    title=_optional_text(fixture, "section_title") or "Fixture result",
                    claims=claims,
                    sufficient=bool(evidence),
                )
            ],
            state.evidence_pool,
        )
        state.report = report
        state.final_output = _required_text(fixture, "final_output")
        artifacts.update(
            {
                "report": report,
                "evidence_pool": state.evidence_pool,
                "final_output": state.final_output,
            }
        )
        evidence_by_id = {item.id: item for item in evidence}
        return WorkflowCaseOutput(
            state=state,
            report=report,
            evidence_by_id=evidence_by_id,
            runtime=RuntimeMetrics(
                completed=True,
                latency_ms=max(0, round((self.clock() - started) * 1000)),
                prompt_tokens=None,
                completion_tokens=None,
                retrieval_calls=1,
                failed_calls=failed_calls,
            ),
            artifacts=artifacts,
        )


@dataclass(frozen=True, slots=True)
class OfflineFixture:
    fixture_version: str
    index_version: str
    retriever: FixtureRetriever
    workflow_runner: FixtureWorkflowRunner
    judge: RecordedEvaluationJudge


def load_offline_fixture(
    path: str | Path,
    *,
    project_root: str | Path,
) -> OfflineFixture:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordedFixtureError(f"cannot load fixture {source}: {exc}") from exc
    root = _required_object(raw, "fixture")
    fixture_version = _required_text(root, "fixture_version")
    index_version = _required_text(root, "index_version")
    results_by_query = _parse_retrieval(root.get("retrieval_by_query", {}))
    hits_by_query = _parse_search_hits(root.get("search_hits_by_query", {}))
    documents_by_url = _parse_pages(root.get("pages_by_url", {}))
    workflow_by_case = _object_mapping(root.get("workflow_by_case", {}), "workflow_by_case")
    judge_results = _parse_judge_results(root.get("judge_results_by_case", {}))

    provider = RecordedSearchProvider(hits_by_query)
    loader = RecordedUrlLoader(documents_by_url)
    web_retriever = WebRetriever(
        provider,
        url_loader=loader,
        search_limit=5,
        fetch_limit=3,
    )
    return OfflineFixture(
        fixture_version=fixture_version,
        index_version=index_version,
        retriever=FixtureRetriever(results_by_query),
        workflow_runner=FixtureWorkflowRunner(
            workflow_by_case=workflow_by_case,
            web_retriever=web_retriever,
            project_root=Path(project_root),
        ),
        judge=RecordedEvaluationJudge(judge_results),
    )


def _parse_retrieval(value: object) -> dict[str, list[RetrievalResult]]:
    mapping = _required_object(value, "retrieval_by_query")
    parsed: dict[str, list[RetrievalResult]] = {}
    for query, raw_results in mapping.items():
        if not isinstance(query, str) or not query.strip():
            raise RecordedFixtureError("retrieval queries must be non-empty strings")
        if not isinstance(raw_results, list):
            raise RecordedFixtureError("retrieval results must be arrays")
        results: list[RetrievalResult] = []
        for raw in raw_results:
            item = _required_object(raw, "retrieval result")
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RecordedFixtureError("retrieval score must be numeric")
            try:
                source_type = SourceType(_required_text(item, "source_type"))
            except ValueError as exc:
                raise RecordedFixtureError("retrieval source_type is invalid") from exc
            results.append(
                RetrievalResult(
                    chunk_id=_required_text(item, "chunk_id"),
                    score=float(score),
                    content=_required_text(item, "content"),
                    document_id=_required_text(item, "document_id"),
                    source=_required_text(item, "source"),
                    source_type=source_type,
                    chunk_index=_required_int(item, "chunk_index"),
                    start_char=_required_int(item, "start_char"),
                    end_char=_required_int(item, "end_char"),
                    metadata=_optional_object(item, "metadata"),
                )
            )
        parsed[query] = results
    return parsed


def _parse_search_hits(value: object) -> dict[str, list[WebSearchHit]]:
    mapping = _required_object(value, "search_hits_by_query")
    parsed: dict[str, list[WebSearchHit]] = {}
    for query, raw_hits in mapping.items():
        if not isinstance(raw_hits, list):
            raise RecordedFixtureError("recorded search hits must be arrays")
        hits: list[WebSearchHit] = []
        for rank, raw in enumerate(raw_hits, start=1):
            item = _required_object(raw, "search hit")
            raw_score = item.get("score")
            if raw_score is not None and (
                isinstance(raw_score, bool) or not isinstance(raw_score, (int, float))
            ):
                raise RecordedFixtureError("search hit score must be numeric or null")
            hits.append(
                WebSearchHit(
                    rank=rank,
                    title=_required_text(item, "title"),
                    url=_required_text(item, "url"),
                    snippet=_required_text(item, "snippet"),
                    score=float(raw_score) if raw_score is not None else None,
                )
            )
        parsed[query] = hits
    return parsed


def _parse_pages(value: object) -> dict[str, list[Document]]:
    mapping = _required_object(value, "pages_by_url")
    parsed: dict[str, list[Document]] = {}
    for url, raw in mapping.items():
        item = _required_object(raw, "recorded page")
        metadata = _optional_object(item, "metadata")
        metadata.setdefault("final_url", url)
        parsed[url] = [
            Document(
                content=_required_text(item, "content"),
                source=url,
                source_type=SourceType.URL,
                metadata=metadata,
            )
        ]
    return parsed


def _parse_judge_results(value: object) -> dict[str, JudgeResult]:
    mapping = _required_object(value, "judge_results_by_case")
    parsed: dict[str, JudgeResult] = {}
    for case_id, raw in mapping.items():
        item = _required_object(raw, "Judge result")
        try:
            label = JudgeLabel(_required_text(item, "label"))
        except ValueError as exc:
            raise RecordedFixtureError("recorded Judge label is invalid") from exc
        parsed[case_id] = JudgeResult(label, _required_text(item, "reason"))
    return parsed


def _fixture_source(fixture: dict[str, object]) -> RetrievalSource:
    try:
        return RetrievalSource(_required_text(fixture, "source"))
    except ValueError as exc:
        raise RecordedFixtureError("fixture workflow source is invalid") from exc


def _required_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RecordedFixtureError(f"{name} must be a JSON object")
    return dict(value)


def _object_mapping(
    value: object,
    name: str,
) -> dict[str, dict[str, object]]:
    mapping = _required_object(value, name)
    return {key: _required_object(item, f"{name}.{key}") for key, item in mapping.items()}


def _required_text(mapping: dict[str, object], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RecordedFixtureError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(mapping: dict[str, object], field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RecordedFixtureError(f"{field} must be a non-empty string or null")
    return value.strip()


def _required_int(mapping: dict[str, object], field: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordedFixtureError(f"{field} must be an integer")
    return value


def _optional_object(mapping: dict[str, object], field: str) -> dict[str, object]:
    value = mapping.get(field, {})
    return _required_object(value, field)
