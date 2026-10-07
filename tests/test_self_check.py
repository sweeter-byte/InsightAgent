"""Unit tests for report self-check and controlled repair."""

from __future__ import annotations

import json
from types import SimpleNamespace
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
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import Claim, ReportSection, StructuredReport, assemble_report
from insight_agent.routing import RetrievalSource
from insight_agent.self_check import (
    ReportSelfChecker,
    SelfCheckError,
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)


class FakeLLM:
    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": messages, "tools": tools})
        content = self.payloads.pop(0)
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _research_inputs() -> tuple[
    ResearchPlan,
    StructuredReport,
    dict[str, list[Evidence]],
    dict[str, list[EvidenceAssessment]],
]:
    task = ResearchTask(id="T1", question="How does method A perform?")
    plan = ResearchPlan(
        objective="Assess method A",
        constraints=["Use only collected Evidence"],
        tasks=[task],
    )
    evidence = Evidence(
        id="E1",
        task_id="T1",
        retrieval_source=RetrievalSource.LOCAL,
        origin_id="origin-E1",
        content="Method A reduced tokens in two experiments.",
        source="paper.md",
        metadata={"filename": "paper.md"},
    )
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id="E1",
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Direct experimental result.",
            )
        ],
        coverage=EvidenceCoverage.PARTIAL,
        sufficient=False,
        missing_information=["Long-term maintenance cost is unknown."],
        reason="One required comparison remains unavailable.",
    )
    section = ReportSection(
        task_id="T1",
        title=task.question,
        claims=[
            Claim(
                id="T1-C1",
                task_id="T1",
                text="Method A reduced tokens in two experiments.",
                evidence_ids=["E1"],
            )
        ],
        sufficient=False,
        missing_information=list(assessment.missing_information),
    )
    pool = {"T1": [evidence]}
    report = assemble_report(plan.objective, [section], pool)
    return plan, report, pool, {"T1": [assessment]}


def _two_task_research_inputs() -> tuple[
    ResearchPlan,
    StructuredReport,
    dict[str, list[Evidence]],
    dict[str, list[EvidenceAssessment]],
]:
    plan, report, pool, assessments = _research_inputs()
    task = ResearchTask(id="T2", question="What remains unknown?")
    plan.tasks.append(task)
    assessment = EvidenceAssessment(
        task_id="T2",
        evidence_judgments=[],
        coverage=EvidenceCoverage.INSUFFICIENT,
        sufficient=False,
        missing_information=["No maintenance data is available."],
        reason="No usable Evidence was collected.",
    )
    pool["T2"] = []
    assessments["T2"] = [assessment]
    report = assemble_report(
        plan.objective,
        [
            *report.sections,
            ReportSection(
                task_id="T2",
                title=task.question,
                claims=[],
                sufficient=False,
                missing_information=list(assessment.missing_information),
            ),
        ],
        pool,
    )
    return plan, report, pool, assessments


def test_self_check_models_use_finite_vocabulary() -> None:
    issue = SelfCheckIssue(
        code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
        reason="Claim is broader than Evidence.",
        task_id="T1",
        claim_id="T1-C1",
    )
    result = SelfCheckResult(
        status=SelfCheckStatus.REVISE,
        issues=[issue],
        summary="Repair required.",
    )

    assert {item.value for item in SelfCheckStatus} == {"pass", "revise"}
    assert {item.value for item in SelfCheckIssueCode} == {
        "unsupported_claim",
        "citation_mismatch",
        "missing_gap_disclosure",
        "constraint_violation",
        "inconsistent_claims",
    }
    assert result.issues == [issue]


def test_research_state_defaults_self_check_fields() -> None:
    state = ResearchState(query="query")

    assert state.self_check_result is None
    assert state.self_check_rounds == 0


def test_checker_parses_pass_and_builds_claim_centered_context() -> None:
    plan, report, pool, assessments = _research_inputs()
    llm = FakeLLM(
        json.dumps(
            {
                "status": "pass",
                "issues": [],
                "summary": (
                    "Claims are supported and recorded gaps are disclosed."
                ),
            }
        )
    )

    result = ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool=pool,
        assessments=assessments,
    )

    assert result == SelfCheckResult(
        status=SelfCheckStatus.PASS,
        issues=[],
        summary="Claims are supported and recorded gaps are disclosed.",
    )
    assert llm.calls[0]["tools"] is None
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["claim_contexts"] == [
        {
            "task": {"id": "T1", "question": "How does method A perform?"},
            "claim": {
                "id": "T1-C1",
                "text": "Method A reduced tokens in two experiments.",
                "evidence_ids": ["E1"],
            },
            "evidence": [
                {
                    "evidence_id": "E1",
                    "content": "Method A reduced tokens in two experiments.",
                    "retrieval_source": "local",
                }
            ],
            "latest_assessment": {
                "sufficient": False,
                "missing_information": [
                    "Long-term maintenance cost is unknown."
                ],
                "reason": "One required comparison remains unavailable.",
            },
        }
    ]


@pytest.mark.parametrize("code", list(SelfCheckIssueCode))
def test_checker_accepts_each_bounded_issue_code(code: SelfCheckIssueCode) -> None:
    plan, report, pool, assessments = _research_inputs()
    llm = FakeLLM(
        json.dumps(
            {
                "status": "revise",
                "issues": [
                    {
                        "code": code.value,
                        "reason": "The report requires a bounded correction.",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Repair required.",
            }
        )
    )

    result = ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool=pool,
        assessments=assessments,
    )

    assert result.issues[0].code is code


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": "Unknown task.",
                        "task_id": "T99",
                        "claim_id": None,
                    }
                ],
                "summary": "Invalid identity.",
            },
            "unknown task",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "citation_mismatch",
                        "reason": "Unknown Claim.",
                        "task_id": "T1",
                        "claim_id": "T1-C99",
                    }
                ],
                "summary": "Invalid identity.",
            },
            "unknown Claim",
        ),
        (
            {
                "status": "pass",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": "Issue cannot accompany pass.",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Contradictory result.",
            },
            "pass.*issues",
        ),
        (
            {"status": "revise", "issues": [], "summary": "No issue supplied."},
            "revise.*at least one",
        ),
        (
            {"status": "maybe", "issues": [], "summary": "Ambiguous."},
            "status",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "style_problem",
                        "reason": "Invalid vocabulary.",
                        "task_id": None,
                        "claim_id": None,
                    }
                ],
                "summary": "Invalid issue.",
            },
            "code",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": " ",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Missing reason.",
            },
            "reason",
        ),
        (
            {"status": "pass", "issues": [], "summary": " "},
            "summary",
        ),
    ],
)
def test_checker_rejects_invalid_result_contract(
    payload: dict[str, Any],
    message: str,
) -> None:
    plan, report, pool, assessments = _research_inputs()

    with pytest.raises(SelfCheckError, match=message):
        ReportSelfChecker(llm=FakeLLM(json.dumps(payload))).check(
            plan=plan,
            report=report,
            evidence_pool=pool,
            assessments=assessments,
        )


def test_checker_rejects_claim_and_task_mismatch() -> None:
    plan, report, pool, assessments = _two_task_research_inputs()
    payload = {
        "status": "revise",
        "issues": [
            {
                "code": "inconsistent_claims",
                "reason": "Claim is assigned to the wrong task.",
                "task_id": "T2",
                "claim_id": "T1-C1",
            }
        ],
        "summary": "Invalid identity relationship.",
    }

    with pytest.raises(SelfCheckError, match="does not belong"):
        ReportSelfChecker(llm=FakeLLM(json.dumps(payload))).check(
            plan=plan,
            report=report,
            evidence_pool=pool,
            assessments=assessments,
        )
