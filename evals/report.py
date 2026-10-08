"""Stable machine-readable and Markdown Regression reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.baseline import BaselineRecord
from evals.regression import RegressionComparison
from evals.results import PersistedRun
from evals.serialization import atomic_write_json_artifact, json_safe_value


REPORT_FORMAT_VERSION = "evaluation_regression_report.v1"


def regression_report_payload(
    comparison: RegressionComparison,
    baseline: BaselineRecord,
    candidate: PersistedRun,
) -> dict[str, Any]:
    return {
        "format_version": REPORT_FORMAT_VERSION,
        "baseline": {
            **baseline.source_run.metadata,
            "baseline_kind": baseline.kind.value,
            "aggregates": baseline.source_run.aggregates,
        },
        "candidate": {
            **candidate.metadata,
            "aggregates": candidate.aggregates,
        },
        "comparison": json_safe_value(comparison),
        "baseline_bad_cases": list(baseline.source_run.bad_cases),
        "candidate_bad_cases": list(candidate.bad_cases),
        "evaluation_anomalies": _evaluation_anomalies(candidate),
    }


def render_regression_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["baseline"]
    candidate = payload["candidate"]
    comparison = payload["comparison"]
    lines = [
        "# Evaluation Regression Report",
        "",
        f"Baseline Run: `{baseline['eval_run_id']}`  ",
        f"Candidate Run: `{candidate['eval_run_id']}`  ",
        f"Status: **{str(comparison['status']).upper()}**",
        "",
        "## Run Configuration",
        "",
        "| Field | Baseline | Candidate |",
        "| --- | --- | --- |",
    ]
    for field in (
        "dataset_version",
        "dataset_sha256",
        "mode",
        "evaluation_type",
        "model_name",
        "prompt_version",
        "index_version",
        "retriever_configuration",
    ):
        lines.append(
            f"| {field} | {_display(baseline.get(field))} | "
            f"{_display(candidate.get(field))} |"
        )
    lines.extend(["", "## Comparability", ""])
    for check in comparison["compatibility_checks"]:
        marker = "PASS" if check["compatible"] else "FAIL"
        lines.append(f"- [{marker}] `{check['name']}`: {check['reason']}")
    changes = comparison["observed_system_changes"]
    lines.append("")
    lines.append(
        "Observed system changes: "
        + (", ".join(f"`{item}`" for item in changes) if changes else "none")
    )
    lines.append("")
    lines.append(
        "Baseline computable metrics: "
        + _metric_list(comparison["baseline_computable_metrics"])
    )
    lines.append(
        "Candidate computable metrics: "
        + _metric_list(comparison["candidate_computable_metrics"])
    )
    lines.extend(
        [
            "",
            "## Aggregate Metric Changes",
            "",
            "| Metric | Baseline | Candidate | Delta | Ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for change in comparison["aggregate_changes"]:
        lines.append(
            f"| {change['metric']} | {_display(change['baseline'])} | "
            f"{_display(change['candidate'])} | {_display(change['delta'])} | "
            f"{_display(change['ratio'])} |"
        )
    if not comparison["aggregate_changes"]:
        lines.append("| none | — | — | — | — |")
    lines.extend(["", "## Per-Case Changes", ""])
    if comparison["case_changes"]:
        for change in comparison["case_changes"]:
            lines.append(
                f"- `{change['case_id']}` / `{change['dimension']}`: "
                f"{change['baseline']} → {change['candidate']}"
            )
    else:
        lines.append("No per-Case status changes.")
    lines.extend(["", "## Bad Cases", ""])
    for label, key in (
        ("Baseline", "baseline_bad_cases"),
        ("Candidate", "candidate_bad_cases"),
    ):
        items = payload[key]
        lines.append(f"### {label}")
        lines.append("")
        if items:
            for item in items:
                lines.append(
                    f"- `{item.get('case_id')}` / `{item.get('category')}`: "
                    f"{item.get('summary')} (`{item.get('artifact_path')}`)"
                )
        else:
            lines.append("None.")
        lines.append("")
    lines.extend(["## Evaluation Anomalies", ""])
    anomalies = payload["evaluation_anomalies"]
    if anomalies:
        for item in anomalies:
            lines.append(
                f"- `{item['case_id']}` / `{item['dimension']}` / "
                f"`{item['status']}`: {item['reason']}"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Regression Issues", ""])
    if comparison["issues"]:
        for issue in comparison["issues"]:
            prefix = f"`{issue['case_id']}` / " if issue.get("case_id") else ""
            lines.append(
                f"- {prefix}`{issue['metric']}` ({issue['kind']}): {issue['reason']}"
            )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Final Regression Status",
            "",
            f"**{str(comparison['status']).upper()}**",
            "",
        ]
    )
    return "\n".join(lines)


def write_regression_report(
    comparison: RegressionComparison,
    baseline: BaselineRecord,
    candidate: PersistedRun,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    payload = regression_report_payload(comparison, baseline, candidate)
    json_path = output / "regression.json"
    markdown_path = output / "regression.md"
    atomic_write_json_artifact(json_path, payload)
    markdown_path.write_text(
        render_regression_markdown(payload), encoding="utf-8"
    )
    return json_path, markdown_path


def _evaluation_anomalies(run: PersistedRun) -> list[dict[str, str]]:
    anomalies: list[dict[str, str]] = []
    for result in run.results:
        case = result.get("case")
        case_id = str(case.get("case_id", "")) if isinstance(case, dict) else ""
        tags = set(case.get("tags", [])) if isinstance(case, dict) else set()
        if result.get("execution_status") == "failed":
            anomalies.append(
                {
                    "case_id": case_id,
                    "dimension": "runtime.execution",
                    "status": "failed",
                    "reason": str(result.get("error") or "Case execution failed"),
                }
            )
        retrieval = result.get("retrieval")
        if "retrieval" in tags and not isinstance(retrieval, dict):
            anomalies.append(
                {
                    "case_id": case_id,
                    "dimension": "retrieval",
                    "status": "not_executed",
                    "reason": "applicable retrieval dimension was not executed",
                }
            )
        if isinstance(retrieval, dict) and retrieval.get("metric_status") != "computed":
            anomalies.append(
                {
                    "case_id": case_id,
                    "dimension": "retrieval",
                    "status": str(retrieval.get("metric_status")),
                    "reason": str(retrieval.get("metric_reason") or "metric unavailable"),
                }
            )
        evidence = result.get("evidence")
        if "research" in tags and not isinstance(evidence, dict):
            anomalies.append(
                {
                    "case_id": case_id,
                    "dimension": "evidence",
                    "status": "not_executed",
                    "reason": "applicable evidence dimension was not executed",
                }
            )
        if isinstance(evidence, dict):
            _append_judgment_anomaly(
                anomalies, case_id, "evidence.relevance", evidence.get("relevance")
            )
            _append_judgment_anomaly(
                anomalies, case_id, "evidence.coverage", evidence.get("coverage")
            )
        answer = result.get("answer")
        if "research" in tags and not isinstance(answer, dict):
            anomalies.append(
                {
                    "case_id": case_id,
                    "dimension": "answer",
                    "status": "not_executed",
                    "reason": "applicable answer dimension was not executed",
                }
            )
        if isinstance(answer, dict):
            for claim in answer.get("claim_results", []):
                if isinstance(claim, dict):
                    _append_judgment_anomaly(
                        anomalies,
                        case_id,
                        f"answer.groundedness.{claim.get('claim_id')}",
                        claim.get("groundedness"),
                    )
            _append_judgment_anomaly(
                anomalies, case_id, "answer.coverage", answer.get("coverage")
            )
            _append_judgment_anomaly(
                anomalies, case_id, "answer.constraints", answer.get("constraints")
            )
    return anomalies


def _append_judgment_anomaly(
    output: list[dict[str, str]],
    case_id: str,
    dimension: str,
    value: object,
) -> None:
    if not isinstance(value, dict) or value.get("status") == "completed":
        return
    result = value.get("result")
    result_reason = result.get("reason") if isinstance(result, dict) else None
    output.append(
        {
            "case_id": case_id,
            "dimension": dimension,
            "status": str(value.get("status", "not_executed")),
            "reason": str(value.get("reason") or result_reason or "unavailable"),
        }
    )


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _metric_list(values: list[str]) -> str:
    return ", ".join(f"`{item}`" for item in values) if values else "none"
