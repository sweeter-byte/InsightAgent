"""Integration tests for the fixed report terminal stage."""

from __future__ import annotations

from typing import Any

import pytest

from insight_agent.evidence import (
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
)
from insight_agent.ingestion import SourceType
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import Claim, ReportGenerationError
from insight_agent.research import ResearchRoutingWorkflow
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource, RouteDecision


def _plan(task_count: int = 1) -> ResearchPlan:
    return ResearchPlan(
        objective="Compare memory systems",
        constraints=["Use only collected evidence"],
        tasks=[
            ResearchTask(id=f"T{index}", question=f"Question {index}")
            for index in range(1, task_count + 1)
        ],
    )


def _evidence(
    evidence_id: str = "E1",
    *,
    task_id: str = "T1",
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id=task_id,
        retrieval_source=RetrievalSource.LOCAL,
        origin_id=f"origin-{evidence_id}",
        content=f"Content {evidence_id}",
        source=f"/private/{evidence_id}.md",
        metadata={"filename": f"{evidence_id}.md"},
    )


def _assessment(
    evidence: list[Evidence],
    *,
    task_id: str = "T1",
    sufficient: bool,
    missing_information: list[str] | None = None,
    weak: bool = False,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        task_id=task_id,
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id=item.id,
                relevance=EvidenceRelevance.RELEVANT,
                quality=(EvidenceQuality.WEAK if weak else EvidenceQuality.STRONG),
                reason="Test judgment.",
            )
            for item in evidence
        ],
        coverage=(
            EvidenceCoverage.COMPLETE
            if sufficient
            else EvidenceCoverage.PARTIAL
        ),
        sufficient=sufficient,
        missing_information=list(missing_information or []),
        reason="Test assessment.",
    )


class FakeRouter:
    def route(self, **kwargs: Any) -> RouteDecision:
        task = kwargs["task"]
        return RouteDecision(task.id, RetrievalSource.LOCAL, "Use local")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.calls.append(query)
        index = len(self.calls)
        return [
            RetrievalResult(
                chunk_id=f"chunk-{index}",
                score=0.9,
                content=f"Evidence from retrieval {index}",
                document_id=f"document-{index}",
                source=f"/private/document-{index}.md",
                source_type=SourceType.MARKDOWN,
                chunk_index=0,
                start_char=0,
                end_char=10,
                metadata={"filename": f"document-{index}.md"},
            )
        ]


class ScriptedGrader:
    def __init__(self, sufficiency: list[tuple[bool, list[str]]]) -> None:
        self.sufficiency = list(sufficiency)

    def grade(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        evidence: list[Evidence],
    ) -> EvidenceAssessment:
        sufficient, gaps = self.sufficiency.pop(0)
        return _assessment(
            evidence,
            task_id=task.id,
            sufficient=sufficient,
            missing_information=gaps,
        )


class FakeReportGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_section(self, **kwargs: Any) -> list[Claim]:
        self.calls.append(dict(kwargs))
        task = kwargs["task"]
        evidence = kwargs["evidence"]
        return [
            Claim(
                id=f"{task.id}-C1",
                task_id=task.id,
                text=f"Partial result for {task.id}.",
                evidence_ids=[item.id for item in evidence],
            )
        ]


def _workflow(report_generator: FakeReportGenerator) -> ResearchRoutingWorkflow:
    return ResearchRoutingWorkflow(
        router=FakeRouter(),  # type: ignore[arg-type]
        local_retriever=FakeRetriever(),
        grader=ScriptedGrader([]),  # type: ignore[arg-type]
        report_generator=report_generator,  # type: ignore[arg-type]
    )


def test_research_state_defaults_report_outputs_to_none() -> None:
    state = ResearchState(query="query")

    assert state.report is None
    assert state.final_output is None


def test_generate_report_skips_llm_for_no_allowed_evidence_and_keeps_gap() -> None:
    evidence = [_evidence()]
    assessment = _assessment(
        evidence,
        sufficient=False,
        missing_information=["Missing limitation."],
        weak=True,
    )
    state = ResearchState(
        query="query",
        plan=_plan(),
        available_sources={RetrievalSource.LOCAL},
        task_index=1,
        evidence_pool={"T1": evidence},
        evidence_assessments={"T1": [assessment]},
    )
    report_generator = FakeReportGenerator()

    update = _workflow(report_generator).generate_report(state)

    assert report_generator.calls == []
    assert update["report"].sections[0].claims == []
    assert update["report"].sections[0].sufficient is False
    assert update["report"].sections[0].missing_information == [
        "Missing limitation."
    ]
    assert "当前没有可靠结论" in update["final_output"]
    assert "Missing limitation." in update["final_output"]


def test_generate_report_rejects_sufficient_assessment_without_candidates() -> None:
    evidence = [_evidence()]
    state = ResearchState(
        query="query",
        plan=_plan(),
        available_sources={RetrievalSource.LOCAL},
        task_index=1,
        evidence_pool={"T1": evidence},
        evidence_assessments={
            "T1": [_assessment(evidence, sufficient=True, weak=True)]
        },
    )

    with pytest.raises(ReportGenerationError, match="sufficient.*no report candidates"):
        _workflow(FakeReportGenerator()).generate_report(state)


def test_complete_workflow_generates_report_once_after_all_tasks() -> None:
    report_generator = FakeReportGenerator()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter(),  # type: ignore[arg-type]
        local_retriever=FakeRetriever(),
        grader=ScriptedGrader(  # type: ignore[arg-type]
            [
                (True, []),
                (False, ["T2 first gap"]),
                (True, []),
                (False, ["T3 first gap"]),
                (False, ["T3 final gap"]),
            ]
        ),
        report_generator=report_generator,  # type: ignore[arg-type]
    )
    state = ResearchState(
        query="query",
        plan=_plan(3),
        available_sources={RetrievalSource.LOCAL},
    )

    final_state = workflow.run(state)

    assert [call["task"].id for call in report_generator.calls] == ["T1", "T2", "T3"]
    assert len(report_generator.calls[1]["evidence"]) == 2
    assert report_generator.calls[2]["assessment"].sufficient is False
    assert len(final_state.evidence_assessments["T2"]) == 2
    assert len(final_state.evidence_assessments["T3"]) == 2
    assert final_state.evidence_assessments["T3"][-1].sufficient is False
    assert final_state.report is not None
    assert len(final_state.report.sections) == 3
    assert final_state.report.sections[2].missing_information == ["T3 final gap"]
    assert final_state.final_output is not None
    assert "Partial result for T3." in final_state.final_output
    assert "T3 final gap" in final_state.final_output
    assert final_state.final_output.count("# 研究结果") == 1
