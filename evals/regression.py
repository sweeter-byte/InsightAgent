"""Compatibility-aware comparison of persisted Baseline and Candidate Runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from evals.baseline import BaselineKind, BaselineRecord
from evals.results import PersistedRun


POLICY_FORMAT_VERSION = "evaluation_regression_policy.v1"


class PolicyAction(str, Enum):
    INCONCLUSIVE = "inconclusive"
    REGRESSION = "regression"
    IGNORE = "ignore"


class RegressionStatus(str, Enum):
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    format_version: str
    aggregate_max_drops: dict[str, float]
    runtime_max_increase_ratios: dict[str, float]
    critical_case_ids: tuple[str, ...]
    fail_on_any_pass_to_fail: bool
    missing_metric_action: PolicyAction
    infrastructure_error_action: PolicyAction


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    name: str
    compatible: bool
    baseline: object
    candidate: object
    reason: str


@dataclass(frozen=True, slots=True)
class MetricChange:
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    ratio: float | None = None


@dataclass(frozen=True, slots=True)
class CaseChange:
    case_id: str
    dimension: str
    baseline: str
    candidate: str


@dataclass(frozen=True, slots=True)
class RegressionIssue:
    kind: str
    metric: str
    reason: str
    case_id: str | None = None
    baseline: object = None
    candidate: object = None


@dataclass(frozen=True, slots=True)
class RegressionComparison:
    format_version: str
    baseline_run_id: str
    candidate_run_id: str
    policy_version: str
    status: RegressionStatus
    compatibility_checks: tuple[CompatibilityCheck, ...]
    aggregate_changes: tuple[MetricChange, ...]
    case_changes: tuple[CaseChange, ...]
    issues: tuple[RegressionIssue, ...]
    observed_system_changes: tuple[str, ...]
    baseline_computable_metrics: tuple[str, ...]
    candidate_computable_metrics: tuple[str, ...]
    baseline_bad_cases: tuple[dict[str, Any], ...]
    candidate_bad_cases: tuple[dict[str, Any], ...]


def load_regression_policy(path: str | Path) -> RegressionPolicy:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Regression Policy does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Regression Policy is invalid JSON: {source}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Regression Policy must be a JSON object")
    version = payload.get("format_version")
    if version != POLICY_FORMAT_VERSION:
        raise ValueError(f"unsupported Policy format_version: {version!r}")
    aggregate = _thresholds(payload.get("aggregate_max_drops"), "aggregate_max_drops")
    runtime = _thresholds(
        payload.get("runtime_max_increase_ratios"),
        "runtime_max_increase_ratios",
    )
    critical = payload.get("critical_case_ids", [])
    if not isinstance(critical, list) or any(
        not isinstance(item, str) or not item for item in critical
    ):
        raise ValueError("critical_case_ids must be an array of non-empty strings")
    fail_any = payload.get("fail_on_any_pass_to_fail", True)
    if not isinstance(fail_any, bool):
        raise ValueError("fail_on_any_pass_to_fail must be boolean")
    try:
        missing = PolicyAction(payload.get("missing_metric_action", "inconclusive"))
        infrastructure = PolicyAction(
            payload.get("infrastructure_error_action", "inconclusive")
        )
    except ValueError as exc:
        raise ValueError("Policy action must be inconclusive, regression, or ignore") from exc
    return RegressionPolicy(
        format_version=version,
        aggregate_max_drops=aggregate,
        runtime_max_increase_ratios=runtime,
        critical_case_ids=tuple(critical),
        fail_on_any_pass_to_fail=fail_any,
        missing_metric_action=missing,
        infrastructure_error_action=infrastructure,
    )


def compare_regression(
    baseline: BaselineRecord,
    candidate: PersistedRun,
    policy: RegressionPolicy,
) -> RegressionComparison:
    if not isinstance(baseline, BaselineRecord):
        raise TypeError("baseline must be a BaselineRecord")
    if not isinstance(candidate, PersistedRun):
        raise TypeError("candidate must be a PersistedRun")
    if not isinstance(policy, RegressionPolicy):
        raise TypeError("policy must be a RegressionPolicy")
    baseline_run = baseline.source_run
    checks_list = list(_compatibility_checks(baseline, candidate))
    policy_metrics = sorted(
        set(policy.aggregate_max_drops) | set(policy.runtime_max_increase_ratios)
    )
    for metric in policy_metrics:
        for role, run in (("baseline", baseline_run), ("candidate", candidate)):
            _check(
                checks_list,
                f"{role}_aggregate_integrity.{metric}",
                _derived_policy_metric(run, metric),
                _numeric_path(run.aggregates, metric),
                reason_prefix=f"{role.capitalize()} Run: ",
            )
    checks = tuple(checks_list)
    observed = _observed_system_changes(baseline_run, candidate)
    baseline_id = str(baseline_run.metadata.get("eval_run_id", ""))
    candidate_id = str(candidate.metadata.get("eval_run_id", ""))
    baseline_metrics = _numeric_leaf_paths(baseline_run.aggregates)
    candidate_metrics = _numeric_leaf_paths(candidate.aggregates)
    if any(not item.compatible for item in checks):
        issues = tuple(
            RegressionIssue(
                kind="compatibility",
                metric=item.name,
                baseline=item.baseline,
                candidate=item.candidate,
                reason=item.reason,
            )
            for item in checks
            if not item.compatible
        )
        return RegressionComparison(
            format_version="evaluation_regression.v1",
            baseline_run_id=baseline_id,
            candidate_run_id=candidate_id,
            policy_version=policy.format_version,
            status=RegressionStatus.INCOMPATIBLE,
            compatibility_checks=checks,
            aggregate_changes=(),
            case_changes=(),
            issues=issues,
            observed_system_changes=observed,
            baseline_computable_metrics=baseline_metrics,
            candidate_computable_metrics=candidate_metrics,
            baseline_bad_cases=baseline_run.bad_cases,
            candidate_bad_cases=candidate.bad_cases,
        )

    issues: list[RegressionIssue] = []
    changes: list[MetricChange] = []
    has_regression = False
    has_inconclusive = False
    for metric, maximum_drop in policy.aggregate_max_drops.items():
        old = _comparable_policy_metric(baseline_run, metric)
        new = _comparable_policy_metric(candidate, metric)
        delta = new - old if old is not None and new is not None else None
        changes.append(MetricChange(metric, old, new, delta))
        if old is None and new is None:
            continue
        if new is None:
            state = _apply_action(
                policy.missing_metric_action,
                issues,
                RegressionIssue(
                    kind="missing_metric",
                    metric=metric,
                    baseline=old,
                    candidate=new,
                    reason="configured aggregate metric is unavailable",
                ),
            )
            has_regression |= state is RegressionStatus.REGRESSION
            has_inconclusive |= state is RegressionStatus.INCONCLUSIVE
        elif old - new > maximum_drop:
            has_regression = True
            issues.append(
                RegressionIssue(
                    kind="aggregate_regression",
                    metric=metric,
                    baseline=old,
                    candidate=new,
                    reason=f"drop {old - new:.6g} exceeds maximum {maximum_drop:.6g}",
                )
            )

    for metric, maximum_ratio in policy.runtime_max_increase_ratios.items():
        old = _comparable_policy_metric(baseline_run, metric)
        new = _comparable_policy_metric(candidate, metric)
        delta = new - old if old is not None and new is not None else None
        ratio = None if old in {None, 0.0} or new is None else (new - old) / old
        changes.append(MetricChange(metric, old, new, delta, ratio))
        if old is None and new is None:
            continue
        if new is None:
            state = _apply_action(
                policy.missing_metric_action,
                issues,
                RegressionIssue(
                    kind="missing_metric",
                    metric=metric,
                    baseline=old,
                    candidate=new,
                    reason="configured runtime metric is unavailable",
                ),
            )
            has_regression |= state is RegressionStatus.REGRESSION
            has_inconclusive |= state is RegressionStatus.INCONCLUSIVE
        elif (old == 0 and new > 0) or (
            ratio is not None and ratio > maximum_ratio
        ):
            has_regression = True
            issues.append(
                RegressionIssue(
                    kind="runtime_regression",
                    metric=metric,
                    baseline=old,
                    candidate=new,
                    reason=(
                        "runtime cost increased from zero"
                        if old == 0
                        else f"increase ratio {ratio:.6g} exceeds maximum {maximum_ratio:.6g}"
                    ),
                )
            )

    case_changes: list[CaseChange] = []
    old_cases = _case_map(baseline_run)
    new_cases = _case_map(candidate)
    critical_ids = set(policy.critical_case_ids)
    for case_id in sorted(old_cases):
        old_result = old_cases[case_id]
        new_result = new_cases[case_id]
        old_outcomes = _case_outcomes(old_result)
        new_outcomes = _case_outcomes(new_result)
        dimensions = sorted(set(old_outcomes) | set(new_outcomes))
        case_metadata = old_result.get("case", {}).get("metadata", {})
        is_critical = case_id in critical_ids or (
            isinstance(case_metadata, dict) and case_metadata.get("critical") is True
        )
        for dimension in dimensions:
            old_status = old_outcomes.get(dimension, "not_executed")
            new_status = new_outcomes.get(dimension, "not_executed")
            if old_status == new_status:
                continue
            case_changes.append(
                CaseChange(case_id, dimension, old_status, new_status)
            )
            if new_status == "infrastructure_error" and old_status in {
                "pass",
                "partial",
                "fail",
            }:
                state = _apply_action(
                    policy.infrastructure_error_action,
                    issues,
                    RegressionIssue(
                        kind="infrastructure_error",
                        metric=dimension,
                        case_id=case_id,
                        baseline=old_status,
                        candidate=new_status,
                        reason="Candidate evaluation infrastructure did not produce a quality result",
                    ),
                )
                has_regression |= state is RegressionStatus.REGRESSION
                has_inconclusive |= state is RegressionStatus.INCONCLUSIVE
            elif new_status in {"not_computable", "not_executed"} and old_status in {
                "pass",
                "partial",
                "fail",
            }:
                state = _apply_action(
                    policy.missing_metric_action,
                    issues,
                    RegressionIssue(
                        kind="missing_case_metric",
                        metric=dimension,
                        case_id=case_id,
                        baseline=old_status,
                        candidate=new_status,
                        reason="Candidate Case dimension is unavailable",
                    ),
                )
                has_regression |= state is RegressionStatus.REGRESSION
                has_inconclusive |= state is RegressionStatus.INCONCLUSIVE
            elif old_status == "pass" and new_status == "fail" and (
                policy.fail_on_any_pass_to_fail or is_critical
            ):
                has_regression = True
                issues.append(
                    RegressionIssue(
                        kind="case_regression",
                        metric=dimension,
                        case_id=case_id,
                        baseline=old_status,
                        candidate=new_status,
                        reason="Case dimension regressed from pass to fail",
                    )
                )

    status = (
        RegressionStatus.REGRESSION
        if has_regression
        else RegressionStatus.INCONCLUSIVE
        if has_inconclusive
        else RegressionStatus.PASS
    )
    return RegressionComparison(
        format_version="evaluation_regression.v1",
        baseline_run_id=baseline_id,
        candidate_run_id=candidate_id,
        policy_version=policy.format_version,
        status=status,
        compatibility_checks=checks,
        aggregate_changes=tuple(changes),
        case_changes=tuple(case_changes),
        issues=tuple(issues),
        observed_system_changes=observed,
        baseline_computable_metrics=baseline_metrics,
        candidate_computable_metrics=candidate_metrics,
        baseline_bad_cases=baseline_run.bad_cases,
        candidate_bad_cases=candidate.bad_cases,
    )


def _compatibility_checks(
    baseline: BaselineRecord, candidate: PersistedRun
) -> tuple[CompatibilityCheck, ...]:
    old = baseline.source_run
    checks: list[CompatibilityCheck] = []
    for name in ("dataset_version", "dataset_sha256", "mode", "evaluation_type"):
        _check(checks, name, old.metadata.get(name), candidate.metadata.get(name))
    expected_kind = (
        "mock_test" if baseline.kind is BaselineKind.MOCK_ONLY else "live_test"
    )
    _check(
        checks,
        "baseline_kind",
        expected_kind,
        candidate.metadata.get("evaluation_type"),
    )
    old_cases = _case_map(old)
    new_cases = _case_map(candidate)
    _check(checks, "case_ids", sorted(old_cases), sorted(new_cases))
    if set(old_cases) == set(new_cases):
        for case_id in sorted(old_cases):
            old_case = old_cases[case_id].get("case", {})
            new_case = new_cases[case_id].get("case", {})
            for name in ("gold_retrieval_ids", "fixture_set"):
                _check(
                    checks,
                    name,
                    old_case.get(name),
                    new_case.get(name),
                    reason_prefix=f"Case {case_id}: ",
                )
    has_gold = any(
        bool(result.get("case", {}).get("gold_retrieval_ids"))
        for result in old.results
        if isinstance(result.get("case"), dict)
    )
    if has_gold:
        _check(
            checks,
            "index_version",
            old.metadata.get("index_version"),
            candidate.metadata.get("index_version"),
        )
    old_config = old.metadata.get("retriever_configuration", {})
    new_config = candidate.metadata.get("retriever_configuration", {})
    old_fixture = old_config.get("fixture_version") if isinstance(old_config, dict) else None
    new_fixture = new_config.get("fixture_version") if isinstance(new_config, dict) else None
    _check(checks, "fixture_version", old_fixture, new_fixture)
    return tuple(checks)


def _check(
    output: list[CompatibilityCheck],
    name: str,
    baseline: object,
    candidate: object,
    *,
    reason_prefix: str = "",
) -> None:
    same = baseline == candidate
    output.append(
        CompatibilityCheck(
            name=name,
            compatible=same,
            baseline=baseline,
            candidate=candidate,
            reason=(
                reason_prefix + "values match"
                if same
                else reason_prefix + "fixed experimental inputs differ"
            ),
        )
    )


def _observed_system_changes(old: PersistedRun, new: PersistedRun) -> tuple[str, ...]:
    changes: list[str] = []
    for name in ("git_commit", "model_name", "prompt_version", "config_fingerprint"):
        if old.metadata.get(name) != new.metadata.get(name):
            changes.append(name)
    old_config = old.metadata.get("retriever_configuration", {})
    new_config = new.metadata.get("retriever_configuration", {})
    if isinstance(old_config, dict) and isinstance(new_config, dict):
        for name in sorted(set(old_config) | set(new_config)):
            if name == "fixture_version":
                continue
            if old_config.get(name) != new_config.get(name):
                changes.append(f"retriever_configuration.{name}")
    return tuple(changes)


def _case_map(run: PersistedRun) -> dict[str, dict[str, Any]]:
    return {
        str(result["case"]["case_id"]): result
        for result in run.results
        if isinstance(result.get("case"), dict)
    }


def _case_outcomes(result: dict[str, Any]) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    runtime = result.get("runtime")
    if result.get("execution_status") == "failed" or (
        isinstance(runtime, dict) and runtime.get("completed") is False
    ):
        outcomes["runtime.execution"] = "infrastructure_error"
    elif result.get("execution_status") == "completed" and (
        isinstance(runtime, dict) and runtime.get("completed") is True
    ):
        outcomes["runtime.execution"] = "pass"
    retrieval = result.get("retrieval")
    if isinstance(retrieval, dict):
        if retrieval.get("metric_status") == "computed":
            recall = retrieval.get("recall_at_k")
            outcomes["retrieval.recall_at_k"] = (
                "pass" if isinstance(recall, (int, float)) and recall >= 1.0 else "fail"
            )
        else:
            outcomes["retrieval.recall_at_k"] = "not_computable"
        pre = retrieval.get("pre_rerank_reciprocal_rank")
        post = retrieval.get("post_rerank_reciprocal_rank")
        if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
            outcomes["retrieval.rerank"] = "pass" if post >= pre else "fail"
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        outcomes["evidence.relevance"] = _judgment_status(evidence.get("relevance"))
        outcomes["evidence.coverage"] = _judgment_status(evidence.get("coverage"))
        rate = evidence.get("provenance_valid_rate")
        errors = evidence.get("provenance_errors")
        if isinstance(rate, (int, float)):
            outcomes["evidence.provenance"] = (
                "pass" if rate >= 1.0 and not errors else "fail"
            )
    answer = result.get("answer")
    if isinstance(answer, dict):
        outcomes["answer.structure"] = (
            "pass" if not answer.get("structural_errors") else "fail"
        )
        outcomes["answer.coverage"] = _judgment_status(answer.get("coverage"))
        outcomes["answer.constraints"] = _judgment_status(answer.get("constraints"))
        claims = answer.get("claim_results")
        if isinstance(claims, list) and claims:
            statuses = [
                _judgment_status(item.get("groundedness"))
                for item in claims
                if isinstance(item, dict)
            ]
            outcomes["answer.groundedness"] = _worst_status(statuses)
    return outcomes


def _judgment_status(value: object) -> str:
    if not isinstance(value, dict):
        return "not_executed"
    status = value.get("status")
    if status == "not_computable":
        return "not_computable"
    if status in {"timeout", "backend_error", "invalid_output"}:
        return "infrastructure_error"
    result = value.get("result")
    if status == "completed" and isinstance(result, dict):
        label = result.get("label")
        if label in {"pass", "partial", "fail"}:
            return str(label)
    return "infrastructure_error"


def _worst_status(statuses: list[str]) -> str:
    order = {
        "fail": 5,
        "infrastructure_error": 4,
        "partial": 3,
        "not_computable": 2,
        "not_executed": 1,
        "pass": 0,
    }
    return max(statuses, key=lambda item: order[item]) if statuses else "not_executed"


def _numeric_path(root: dict[str, Any], path: str) -> float | None:
    value: object = root
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numeric_leaf_paths(root: dict[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(value: object, prefix: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{prefix}.{key}" if prefix else key)
        elif not isinstance(value, bool) and isinstance(value, (int, float)):
            paths.append(prefix)

    visit(root, "")
    return tuple(paths)


def _derived_policy_metric(run: PersistedRun, metric: str) -> float | None:
    completed = [
        result for result in run.results if result.get("execution_status") == "completed"
    ]
    if metric == "retrieval.mean_recall_at_k":
        return _mean_values(
            _computed_retrieval_values(completed, "recall_at_k", metric)
        )
    if metric == "retrieval.mrr":
        return _mean_values(
            _computed_retrieval_values(completed, "reciprocal_rank", metric)
        )
    if metric == "evidence.provenance_valid_rate":
        return _mean_values(
            _nested_numbers(completed, ("evidence", "provenance_valid_rate"), metric)
        )
    judgment_paths = {
        "evidence.relevance_pass_rate": ("evidence", "relevance"),
        "evidence.coverage_pass_rate": ("evidence", "coverage"),
        "answer.coverage_pass_rate": ("answer", "coverage"),
        "answer.constraint_pass_rate": ("answer", "constraints"),
    }
    if metric in judgment_paths:
        judgments = [
            value
            for result in completed
            if (value := _nested_value(result, judgment_paths[metric])) is not None
        ]
        return _judgment_pass_rate(judgments, metric)
    if metric == "answer.grounded_claim_pass_rate":
        judgments: list[object] = []
        for result in completed:
            answer = result.get("answer")
            if not isinstance(answer, dict):
                continue
            claims = answer.get("claim_results")
            if not isinstance(claims, list):
                continue
            judgments.extend(
                claim.get("groundedness")
                for claim in claims
                if isinstance(claim, dict)
            )
        return _judgment_pass_rate(judgments, metric)

    runtimes = [
        runtime
        for result in run.results
        if isinstance((runtime := result.get("runtime")), dict)
    ]
    if metric == "runtime.completion_rate":
        if not run.results:
            return None
        completed_flags = [runtime.get("completed") for runtime in runtimes]
        if any(not isinstance(value, bool) for value in completed_flags):
            raise ValueError("runtime.completed must be boolean")
        return sum(completed_flags) / len(run.results)
    if metric in {"runtime.mean_latency_ms", "runtime.p95_latency_ms"}:
        values = [
            _required_number(runtime.get("latency_ms"), metric) for runtime in runtimes
        ]
        if metric == "runtime.mean_latency_ms":
            return _mean_values(values)
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    token_fields = {
        "runtime.total_prompt_tokens": "prompt_tokens",
        "runtime.total_completion_tokens": "completion_tokens",
    }
    if metric in token_fields:
        values = [
            _required_number(value, metric)
            for runtime in runtimes
            if (value := runtime.get(token_fields[metric])) is not None
        ]
        return sum(values) if values else None
    raise ValueError(f"unsupported Regression Policy metric: {metric}")


def _comparable_policy_metric(run: PersistedRun, metric: str) -> float | None:
    value = _derived_policy_metric(run, metric)
    token_fields = {
        "runtime.total_prompt_tokens": "prompt_tokens",
        "runtime.total_completion_tokens": "completion_tokens",
    }
    field = token_fields.get(metric)
    if field is None:
        return value
    runtimes = [
        runtime
        for result in run.results
        if isinstance((runtime := result.get("runtime")), dict)
    ]
    if not runtimes or any(runtime.get(field) is None for runtime in runtimes):
        return None
    return value


def _computed_retrieval_values(
    results: list[dict[str, Any]], field: str, metric: str
) -> list[float]:
    values: list[float] = []
    for result in results:
        retrieval = result.get("retrieval")
        if not isinstance(retrieval, dict) or retrieval.get("metric_status") != "computed":
            continue
        value = retrieval.get(field)
        if value is not None:
            values.append(_required_number(value, metric))
    return values


def _nested_numbers(
    results: list[dict[str, Any]], path: tuple[str, ...], metric: str
) -> list[float]:
    values: list[float] = []
    for result in results:
        value = _nested_value(result, path)
        if value is not None:
            values.append(_required_number(value, metric))
    return values


def _nested_value(root: dict[str, Any], path: tuple[str, ...]) -> object | None:
    value: object = root
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


def _judgment_pass_rate(judgments: list[object], metric: str) -> float | None:
    labels: list[str] = []
    for judgment in judgments:
        if not isinstance(judgment, dict) or judgment.get("status") != "completed":
            continue
        result = judgment.get("result")
        label = result.get("label") if isinstance(result, dict) else None
        if label not in {"pass", "partial", "fail"}:
            raise ValueError(f"{metric} contains an invalid completed Judge result")
        labels.append(str(label))
    if not labels:
        return None
    return labels.count("pass") / len(labels)


def _required_number(value: object, metric: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{metric} contains a non-finite or non-numeric value")
    return float(value)


def _mean_values(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _apply_action(
    action: PolicyAction,
    issues: list[RegressionIssue],
    issue: RegressionIssue,
) -> RegressionStatus | None:
    if action is PolicyAction.IGNORE:
        return None
    issues.append(issue)
    if action is PolicyAction.REGRESSION:
        return RegressionStatus.REGRESSION
    return RegressionStatus.INCONCLUSIVE


def _thresholds(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"{name}.{key} must be a finite non-negative number")
        result[key] = float(item)
    return result
