"""Unit tests for report self-check and controlled repair."""

from __future__ import annotations

from insight_agent.planning import ResearchState
from insight_agent.self_check import (
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)


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
