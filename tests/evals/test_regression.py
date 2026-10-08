from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from evals import (
    BaselineKind,
    BaselineRecord,
    PersistedRun,
    PolicyAction,
    RegressionPolicy,
    RegressionStatus,
    compare_regression,
    load_regression_policy,
)


def _case(case_id: str, *, label: str = "pass", critical: bool = False) -> dict:
    judgment = {
        "status": "completed",
        "result": {"label": label, "reason": label},
        "reason": "",
    }
    return {
        "metadata": {},
        "case": {
            "case_id": case_id,
            "query": "query",
            "tags": ["retrieval", "research"],
            "gold_retrieval_ids": [f"gold-{case_id}"],
            "required_points": ["point"],
            "constraints": ["constraint"],
            "fixture_set": "fixture-v1",
            "metadata": {"critical": critical},
        },
        "retrieval": {
            "case_id": case_id,
            "retrieved_ids": [f"gold-{case_id}"],
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "metric_status": "computed",
            "metric_reason": "",
            "pre_rerank_ids": [],
            "post_rerank_ids": [],
            "pre_rerank_reciprocal_rank": None,
            "post_rerank_reciprocal_rank": None,
        },
        "evidence": {
            "case_id": case_id,
            "provenance_valid_rate": 1.0,
            "evidence_count": 1,
            "relevance": judgment,
            "coverage": judgment,
            "provenance_errors": [],
        },
        "answer": {
            "case_id": case_id,
            "total_claims": 1,
            "claim_results": [{"claim_id": "C1", "groundedness": judgment}],
            "coverage": judgment,
            "constraints": judgment,
            "grounded_claims": 1 if label == "pass" else 0,
            "structural_errors": [],
        },
        "runtime": {
            "completed": True,
            "latency_ms": 100,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "retrieval_calls": 1,
            "failed_calls": 0,
        },
        "execution_status": "completed",
        "error": None,
        "artifact_paths": {"result": f"cases/{case_id}/result.json"},
    }


def _run(*cases: dict, run_id: str = "run-a") -> PersistedRun:
    return PersistedRun(
        format_version="evaluation_run.v1",
        metadata={
            "eval_run_id": run_id,
            "dataset_version": "dataset-v1",
            "dataset_sha256": "a" * 64,
            "git_commit": "abc123",
            "model_name": "model-a",
            "prompt_version": "prompt-v1",
            "index_version": "index-v1",
            "retriever_configuration": {
                "top_k": 5,
                "reranker": "reranker-a",
                "fixture_version": "fixture-v1",
            },
            "mode": "offline",
            "evaluation_type": "mock_test",
            "config_fingerprint": "fingerprint-a",
        },
        aggregates={
            "retrieval": {"mean_recall_at_k": 1.0, "mrr": 1.0},
            "answer": {"grounded_claim_pass_rate": 1.0},
            "runtime": {
                "p95_latency_ms": 100.0,
                "total_prompt_tokens": 10 * len(cases),
                "total_completion_tokens": 5 * len(cases),
            },
        },
        results=tuple(cases),
        bad_cases=(),
    )


def _baseline(run: PersistedRun) -> BaselineRecord:
    return BaselineRecord(
        "evaluation_baseline.v1", BaselineKind.MOCK_ONLY, True, run
    )


def _policy(**changes: object) -> RegressionPolicy:
    values = {
        "format_version": "evaluation_regression_policy.v1",
        "aggregate_max_drops": {
            "retrieval.mean_recall_at_k": 0.02,
            "answer.grounded_claim_pass_rate": 0.02,
        },
        "runtime_max_increase_ratios": {
            "runtime.p95_latency_ms": 0.2,
            "runtime.total_prompt_tokens": 0.2,
        },
        "critical_case_ids": (),
        "fail_on_any_pass_to_fail": True,
        "missing_metric_action": PolicyAction.INCONCLUSIVE,
        "infrastructure_error_action": PolicyAction.INCONCLUSIVE,
    }
    values.update(changes)
    return RegressionPolicy(**values)


def test_same_evaluation_run_has_no_regression() -> None:
    run = _run(_case("case-1"))

    comparison = compare_regression(_baseline(run), run, _policy())

    assert comparison.status is RegressionStatus.PASS
    assert comparison.issues == ()
    assert comparison.baseline_computable_metrics == (
        comparison.candidate_computable_metrics
    )


def test_pass_to_fail_triggers_regression_even_when_aggregates_are_unchanged() -> None:
    baseline_run = _run(
        _case("critical", label="pass", critical=True),
        _case("other", label="fail"),
    )
    candidate_cases = (
        _case("critical", label="fail", critical=True),
        _case("other", label="pass"),
    )
    candidate = _run(*candidate_cases, run_id="run-b")
    candidate.aggregates["answer"]["grounded_claim_pass_rate"] = 0.5
    baseline_run.aggregates["answer"]["grounded_claim_pass_rate"] = 0.5

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.REGRESSION
    assert any(
        issue.case_id == "critical" and "pass to fail" in issue.reason.lower()
        for issue in comparison.issues
    )


def test_aggregate_drop_and_runtime_increase_use_policy_thresholds() -> None:
    baseline_run = _run(_case("case-1"))
    candidate = deepcopy(_run(_case("case-1"), run_id="run-b"))
    candidate.results[0]["retrieval"]["recall_at_k"] = 0.9
    candidate.results[0]["runtime"]["latency_ms"] = 125
    candidate.aggregates["retrieval"]["mean_recall_at_k"] = 0.9
    candidate.aggregates["runtime"]["p95_latency_ms"] = 125.0

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.REGRESSION
    assert {issue.metric for issue in comparison.issues} >= {
        "retrieval.mean_recall_at_k",
        "runtime.p95_latency_ms",
    }


def test_missing_metric_is_not_coerced_to_zero() -> None:
    baseline_run = _run(_case("case-1"))
    candidate_case = _case("case-1")
    candidate_case["answer"]["claim_results"][0]["groundedness"] = {
        "status": "not_computable",
        "result": None,
        "reason": "missing Evidence",
    }
    candidate = _run(candidate_case, run_id="run-b")
    del candidate.aggregates["answer"]["grounded_claim_pass_rate"]

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.INCONCLUSIVE
    change = next(
        item
        for item in comparison.aggregate_changes
        if item.metric == "answer.grounded_claim_pass_rate"
    )
    assert change.baseline == 1.0
    assert change.candidate is None
    assert change.delta is None


def test_stored_policy_metric_must_match_persisted_case_results() -> None:
    baseline_run = _run(_case("case-1"))
    candidate_case = _case("case-1", label="partial")
    candidate = _run(candidate_case, run_id="run-b")
    candidate.aggregates["answer"]["grounded_claim_pass_rate"] = 1.0

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.INCOMPATIBLE
    assert any(
        issue.kind == "compatibility"
        and issue.metric
        == "candidate_aggregate_integrity.answer.grounded_claim_pass_rate"
        for issue in comparison.issues
    )


@pytest.mark.parametrize(
    ("mutation", "check_name"),
    [
        (lambda run: run.metadata.__setitem__("dataset_sha256", "b" * 64), "dataset_sha256"),
        (
            lambda run: run.results[0]["case"].__setitem__(
                "gold_retrieval_ids", ["different-gold"]
            ),
            "gold_retrieval_ids",
        ),
        (lambda run: run.metadata.__setitem__("index_version", "index-v2"), "index_version"),
        (lambda run: run.metadata.__setitem__("evaluation_type", "live_test"), "evaluation_type"),
        (lambda run: run.metadata.__setitem__("mode", "live"), "mode"),
        (
            lambda run: run.results[0]["case"].__setitem__(
                "fixture_set", "fixture-v2"
            ),
            "fixture_set",
        ),
    ],
)
def test_incompatible_inputs_refuse_quality_conclusions(mutation, check_name) -> None:
    baseline_run = _run(_case("case-1"))
    candidate = deepcopy(_run(_case("case-1"), run_id="run-b"))
    mutation(candidate)

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.INCOMPATIBLE
    assert comparison.aggregate_changes == ()
    assert any(
        check.name == check_name and not check.compatible
        for check in comparison.compatibility_checks
    )


def test_prompt_model_and_reranker_changes_remain_comparable() -> None:
    baseline_run = _run(_case("case-1"))
    candidate = deepcopy(_run(_case("case-1"), run_id="run-b"))
    candidate.metadata["model_name"] = "model-b"
    candidate.metadata["prompt_version"] = "prompt-v2"
    candidate.metadata["retriever_configuration"]["reranker"] = "reranker-b"
    candidate.metadata["config_fingerprint"] = "fingerprint-b"

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.PASS
    assert set(comparison.observed_system_changes) >= {
        "model_name",
        "prompt_version",
        "retriever_configuration.reranker",
        "config_fingerprint",
    }


def test_judge_timeout_follows_infrastructure_policy_not_quality_failure() -> None:
    baseline_run = _run(_case("case-1"))
    candidate_case = _case("case-1")
    candidate_case["answer"]["coverage"] = {
        "status": "timeout",
        "result": None,
        "reason": "judge timeout",
    }
    candidate = _run(candidate_case, run_id="run-b")

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.INCONCLUSIVE
    assert any(issue.kind == "infrastructure_error" for issue in comparison.issues)
    assert not any(issue.kind == "case_regression" for issue in comparison.issues)


def test_candidate_case_execution_failure_follows_infrastructure_policy() -> None:
    baseline_run = _run(_case("case-1"))
    failed_case = _case("case-1")
    failed_case["execution_status"] = "failed"
    failed_case["runtime"]["completed"] = False
    failed_case["error"] = "RuntimeError: workflow failed"
    candidate = _run(failed_case, run_id="run-b")
    candidate.aggregates["retrieval"]["mean_recall_at_k"] = None
    candidate.aggregates["answer"]["grounded_claim_pass_rate"] = None

    comparison = compare_regression(
        _baseline(baseline_run), candidate, _policy()
    )

    assert comparison.status is RegressionStatus.INCONCLUSIVE
    assert any(
        issue.kind == "infrastructure_error"
        and issue.metric == "runtime.execution"
        for issue in comparison.issues
    )
    assert not any(issue.kind == "case_regression" for issue in comparison.issues)


def test_versioned_policy_loads_without_hard_coded_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "evaluation_regression_policy.v1",
                "aggregate_max_drops": {"retrieval.mrr": 0.03},
                "runtime_max_increase_ratios": {
                    "runtime.p95_latency_ms": 0.25
                },
                "critical_case_ids": ["critical"],
                "fail_on_any_pass_to_fail": False,
                "missing_metric_action": "inconclusive",
                "infrastructure_error_action": "regression",
            }
        ),
        encoding="utf-8",
    )

    policy = load_regression_policy(path)

    assert policy.aggregate_max_drops == {"retrieval.mrr": 0.03}
    assert policy.critical_case_ids == ("critical",)
    assert policy.infrastructure_error_action is PolicyAction.REGRESSION


@pytest.mark.parametrize("threshold", ["NaN", "Infinity", "1e999"])
def test_policy_rejects_non_finite_thresholds(
    tmp_path: Path,
    threshold: str,
) -> None:
    path = tmp_path / "policy.json"
    path.write_text(
        "{"
        '"format_version":"evaluation_regression_policy.v1",'
        f'"aggregate_max_drops":{{"retrieval.mrr":{threshold}}},'
        '"runtime_max_increase_ratios":{},'
        '"critical_case_ids":[],'
        '"fail_on_any_pass_to_fail":true,'
        '"missing_metric_action":"inconclusive",'
        '"infrastructure_error_action":"inconclusive"'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite non-negative"):
        load_regression_policy(path)
