"""Integration tests for the bounded report self-check terminal workflow."""

from __future__ import annotations

from typing import Any

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
from insight_agent.reporting import Claim, StructuredReport
from insight_agent.research import ResearchRoutingWorkflow
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource, RouteDecision
from insight_agent.self_check import SelfCheckResult, SelfCheckStatus


class FakeRouter:
    def route(self, **kwargs: Any) -> RouteDecision:
        task = kwargs["task"]
        return RouteDecision(task.id, RetrievalSource.LOCAL, "Use local Evidence")


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                content="Method A reduced tokens in two experiments.",
                document_id="paper",
                source="paper.md",
                source_type=SourceType.MARKDOWN,
                chunk_index=0,
                start_char=0,
                end_char=43,
                metadata={"filename": "paper.md"},
            )
        ]


class InsufficientGrader:
    def grade(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        evidence: list[Evidence],
    ) -> EvidenceAssessment:
        return EvidenceAssessment(
            task_id=task.id,
            evidence_judgments=[
                EvidenceJudgment(
                    evidence_id=item.id,
                    relevance=EvidenceRelevance.RELEVANT,
                    quality=EvidenceQuality.STRONG,
                    reason="Direct experiment result.",
                )
                for item in evidence
            ],
            coverage=EvidenceCoverage.PARTIAL,
            sufficient=False,
            missing_information=["Long-term maintenance cost is unknown."],
            reason="The recorded gap remains.",
        )


class FakeReportGenerator:
    def generate_section(self, **kwargs: Any) -> list[Claim]:
        task = kwargs["task"]
        evidence = kwargs["evidence"]
        return [
            Claim(
                id=f"{task.id}-C1",
                task_id=task.id,
                text="Method A reduced tokens in two experiments.",
                evidence_ids=[evidence[0].id],
            )
        ]


class AlwaysPassChecker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def check(self, **kwargs: Any) -> SelfCheckResult:
        self.calls.append(dict(kwargs))
        return SelfCheckResult(
            status=SelfCheckStatus.PASS,
            issues=[],
            summary="The report is faithful to current research state.",
        )


class NeverRepairer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def repair(self, **kwargs: Any) -> StructuredReport:
        self.calls.append(dict(kwargs))
        raise AssertionError("Repair must not run for a passing report")


def _state() -> ResearchState:
    return ResearchState(
        query="Assess method A",
        plan=ResearchPlan(
            objective="Assess method A",
            constraints=["Use only collected Evidence"],
            tasks=[ResearchTask(id="T1", question="How does method A perform?")],
        ),
        available_sources={RetrievalSource.LOCAL},
    )


def test_pass_finalizes_without_repair() -> None:
    checker = AlwaysPassChecker()
    repairer = NeverRepairer()
    workflow = ResearchRoutingWorkflow(
        router=FakeRouter(),  # type: ignore[arg-type]
        local_retriever=FakeRetriever(),
        grader=InsufficientGrader(),  # type: ignore[arg-type]
        report_generator=FakeReportGenerator(),  # type: ignore[arg-type]
        self_checker=checker,  # type: ignore[arg-type]
        report_repairer=repairer,  # type: ignore[arg-type]
        max_retrieval_rounds=1,
    )

    final_state = workflow.run(_state())

    assert len(checker.calls) == 1
    assert repairer.calls == []
    assert final_state.self_check_result is not None
    assert final_state.self_check_result.status is SelfCheckStatus.PASS
    assert final_state.self_check_rounds == 0
    assert final_state.final_output is not None
    assert final_state.final_output.startswith("# 研究结果\n")
    assert "Long-term maintenance cost is unknown." in final_state.final_output
