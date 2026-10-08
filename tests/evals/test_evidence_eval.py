from __future__ import annotations
import json

import pytest

from evals import (
    EvalCase,
    JudgeExecutionStatus,
    JudgeLabel,
    JudgeResult,
)
from evals.evidence_eval import EvidenceEvaluator
from evals.judge import JudgeTask
from insight_agent.routing import RetrievalSource
from tests.evals.fakes import ScriptedJudge, fixed_evidence


def _case(*, required_points: tuple[str, ...] = ("preserves context",)) -> EvalCase:
    return EvalCase(
        case_id="case-1",
        query="Why does chunk overlap matter?",
        required_points=required_points,
    )


def test_evidence_evaluation_keeps_semantic_failure_separate_from_provenance() -> None:
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.FAIL, "The Evidence is unrelated."),
        JudgeResult(JudgeLabel.PARTIAL, "Only one required point is covered."),
    )

    result = EvidenceEvaluator(judge).evaluate(_case(), [fixed_evidence()])

    assert result.provenance_valid_rate == 1.0
    assert result.provenance_errors == ()
    assert result.relevance.result is not None
    assert result.relevance.result.label is JudgeLabel.FAIL
    assert result.coverage.result is not None
    assert result.coverage.result.label is JudgeLabel.PARTIAL


def test_evidence_evaluation_retains_provenance_error_when_judge_passes() -> None:
    invalid = fixed_evidence(
        "W1",
        source="not-a-url",
        retrieval_source=RetrievalSource.WEB,
    )
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.PASS, "Relevant to the query."),
        JudgeResult(JudgeLabel.PASS, "All required points are covered."),
    )

    result = EvidenceEvaluator(judge).evaluate(_case(), [invalid])

    assert result.provenance_valid_rate == 0.0
    assert result.provenance_errors == ("invalid_provenance:W1:web_url",)
    assert result.relevance.result is not None
    assert result.relevance.result.label is JudgeLabel.PASS


def test_evidence_coverage_without_gold_is_not_computable_or_judged() -> None:
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.PASS, "Relevant to the query."),
    )

    result = EvidenceEvaluator(judge).evaluate(
        _case(required_points=()),
        [fixed_evidence()],
    )

    assert [request.task for request in judge.requests] == [
        JudgeTask.EVIDENCE_RELEVANCE
    ]
    assert result.coverage.status is JudgeExecutionStatus.NOT_COMPUTABLE
    assert result.coverage.result is None
    assert "required_points" in result.coverage.reason


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"label": "pass"}, JudgeExecutionStatus.INVALID_OUTPUT),
        (RuntimeError("endpoint down"), JudgeExecutionStatus.BACKEND_ERROR),
        (TimeoutError("deadline"), JudgeExecutionStatus.TIMEOUT),
    ],
)
def test_evidence_judge_infrastructure_state_is_not_semantic_fail(
    response: object,
    expected: JudgeExecutionStatus,
) -> None:
    judge = ScriptedJudge(response)

    result = EvidenceEvaluator(judge).evaluate(
        _case(required_points=()),
        [fixed_evidence()],
    )

    assert result.relevance.status is expected
    assert result.relevance.result is None


def test_evidence_judge_requests_are_independent_and_minimal() -> None:
    judge = ScriptedJudge(
        JudgeResult(JudgeLabel.PASS, "Relevant."),
        JudgeResult(JudgeLabel.PASS, "Covered."),
    )

    EvidenceEvaluator(judge).evaluate(_case(), [fixed_evidence()])

    assert [request.task for request in judge.requests] == [
        JudgeTask.EVIDENCE_RELEVANCE,
        JudgeTask.EVIDENCE_COVERAGE,
    ]
    relevance, coverage = judge.requests
    assert relevance.query == _case().query
    assert relevance.payload["evidence"][0]["id"] == "E1"  # type: ignore[index]
    assert coverage.payload["required_points"] == ["preserves context"]
    serialized = json.dumps(
        [request.payload for request in judge.requests], ensure_ascii=False
    ).lower()
    assert "assessment" not in serialized
    assert "self_check" not in serialized
