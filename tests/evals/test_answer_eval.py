from __future__ import annotations
import json

import pytest

from evals import EvalCase, JudgeExecutionStatus, JudgeLabel, JudgeResult
from evals.answer_eval import AnswerEvaluator
from evals.judge import JudgeTask
from insight_agent.reporting import (
    Claim,
    ReportSection,
    StructuredReport,
    build_citation,
)
from tests.evals.fakes import ScriptedJudge, fixed_evidence


def _report(*claims: Claim) -> StructuredReport:
    return StructuredReport(
        objective="Explain chunk overlap.",
        sections=[
            ReportSection(
                task_id="T1",
                title="Chunking",
                claims=list(claims),
                sufficient=True,
            )
        ],
    )


def test_answer_evaluation_skips_only_structurally_affected_claim() -> None:
    evidence = fixed_evidence()
    report = _report(
        Claim(
            id="T1-C1",
            task_id="T1",
            text="Overlap preserves boundary context.",
            evidence_ids=["E1"],
        ),
        Claim(
            id="T1-C2",
            task_id="T1",
            text="Unsupported structure.",
            evidence_ids=["missing"],
        ),
    )
    report.citations = [build_citation(1, evidence)]
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.PASS, "The bound Evidence supports the Claim."),
        JudgeResult(JudgeLabel.PARTIAL, "One required point is missing."),
        JudgeResult(JudgeLabel.PASS, "The explicit constraint is satisfied."),
    )
    case = EvalCase(
        "case-1",
        "Why does overlap matter?",
        required_points=("preserves boundary context", "prevents truncation"),
        constraints=("be concise",),
    )

    result = AnswerEvaluator(judge).evaluate(case, report, {"E1": evidence})

    assert "missing_evidence:T1-C2:missing" in result.structural_errors
    assert [request.task for request in judge.requests] == [
        JudgeTask.CLAIM_GROUNDEDNESS,
        JudgeTask.ANSWER_COVERAGE,
        JudgeTask.ANSWER_CONSTRAINTS,
    ]
    assert result.claim_results[0].groundedness.result is not None
    assert result.claim_results[0].groundedness.result.label is JudgeLabel.PASS
    assert result.claim_results[1].groundedness.status is (
        JudgeExecutionStatus.NOT_COMPUTABLE
    )
    assert result.claim_results[1].groundedness.result is None
    assert result.grounded_claims == 1


def test_answer_evaluation_distinguishes_semantic_fail_from_structure_error() -> None:
    evidence = fixed_evidence(content="Overlap preserves context.")
    report = _report(
        Claim(
            id="T1-C1",
            task_id="T1",
            text="Overlap guarantees perfect retrieval.",
            evidence_ids=["E1"],
        )
    )
    report.citations = [build_citation(1, evidence)]
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.FAIL, "The Evidence does not support a guarantee."),
    )

    result = AnswerEvaluator(judge).evaluate(
        EvalCase("case-1", "query"),
        report,
        {"E1": evidence},
    )

    assert result.structural_errors == ()
    assert result.claim_results[0].groundedness.status is (
        JudgeExecutionStatus.COMPLETED
    )
    assert result.claim_results[0].groundedness.result is not None
    assert result.claim_results[0].groundedness.result.label is JudgeLabel.FAIL
    assert result.grounded_claims == 0
    assert result.coverage.status is JudgeExecutionStatus.NOT_COMPUTABLE
    assert result.constraints.status is JudgeExecutionStatus.NOT_COMPUTABLE


def test_missing_citation_prevents_affected_claim_judge() -> None:
    evidence = fixed_evidence()
    report = _report(
        Claim("T1-C1", "T1", "Supported fact.", ["E1"]),
    )
    judge = ScriptedJudge()

    result = AnswerEvaluator(judge).evaluate(
        EvalCase("case-1", "query"), report, {"E1": evidence}
    )

    assert result.structural_errors == ("missing_citation:E1",)
    assert result.claim_results[0].groundedness.status is (
        JudgeExecutionStatus.NOT_COMPUTABLE
    )
    assert result.grounded_claims is None
    assert judge.requests == []


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"label": "pass"}, JudgeExecutionStatus.INVALID_OUTPUT),
        (RuntimeError("endpoint down"), JudgeExecutionStatus.BACKEND_ERROR),
        (TimeoutError("deadline"), JudgeExecutionStatus.TIMEOUT),
    ],
)
def test_answer_judge_infrastructure_error_does_not_erase_structure(
    response: object,
    expected: JudgeExecutionStatus,
) -> None:
    evidence = fixed_evidence()
    report = _report(
        Claim("T1-C1", "T1", "Supported fact.", ["E1"]),
        Claim("T1-C2", "T1", "Broken binding.", ["missing"]),
    )
    report.citations = [build_citation(1, evidence)]
    judge = ScriptedJudge(response)

    result = AnswerEvaluator(judge).evaluate(
        EvalCase("case-1", "query"), report, {"E1": evidence}
    )

    assert "missing_evidence:T1-C2:missing" in result.structural_errors
    assert result.claim_results[0].groundedness.status is expected
    assert result.claim_results[0].groundedness.result is None
    assert result.claim_results[1].groundedness.status is (
        JudgeExecutionStatus.NOT_COMPUTABLE
    )


def test_answer_judge_payload_uses_claim_evidence_case_and_rubric_only() -> None:
    evidence = fixed_evidence()
    report = _report(Claim("T1-C1", "T1", "Supported fact.", ["E1"]))
    report.citations = [build_citation(1, evidence)]
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.PASS, "Supported."),
        JudgeResult(JudgeLabel.PASS, "Covered."),
        JudgeResult(JudgeLabel.PARTIAL, "Slightly verbose."),
    )
    case = EvalCase(
        "case-1",
        "query",
        required_points=("fact",),
        constraints=("concise",),
    )

    AnswerEvaluator(judge).evaluate(case, report, {"E1": evidence})

    claim_request = judge.requests[0]
    assert claim_request.payload["claim"]["id"] == "T1-C1"  # type: ignore[index]
    assert claim_request.payload["evidence"][0]["id"] == "E1"  # type: ignore[index]
    serialized = json.dumps(
        [request.payload for request in judge.requests], ensure_ascii=False
    ).lower()
    assert "assessment" not in serialized
    assert "self_check" not in serialized
