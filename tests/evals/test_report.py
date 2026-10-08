from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from evals import compare_regression, write_regression_report
from tests.evals.test_regression import _baseline, _case, _policy, _run


def test_json_and_markdown_regression_reports_are_stable_and_complete(
    tmp_path: Path,
) -> None:
    baseline_run = _run(_case("case-1"))
    candidate = replace(
        _run(_case("case-1"), run_id="run-b"),
        bad_cases=(
            {
                "case_id": "case-1",
                "category": "citation_error",
                "summary": "missing citation",
                "artifact_path": "cases/case-1/result.json",
                "failure_kind": "quality_failure",
            },
        ),
    )
    baseline = _baseline(baseline_run)
    comparison = compare_regression(baseline, candidate, _policy())

    first_json, first_markdown = write_regression_report(
        comparison, baseline, candidate, tmp_path
    )
    first_json_bytes = first_json.read_bytes()
    first_markdown_bytes = first_markdown.read_bytes()
    second_json, second_markdown = write_regression_report(
        comparison, baseline, candidate, tmp_path
    )

    assert second_json.read_bytes() == first_json_bytes
    assert second_markdown.read_bytes() == first_markdown_bytes
    payload = json.loads(first_json.read_text(encoding="utf-8"))
    assert payload["format_version"] == "evaluation_regression_report.v1"
    assert payload["baseline"]["eval_run_id"] == "run-a"
    assert payload["candidate"]["eval_run_id"] == "run-b"
    assert payload["comparison"]["status"] == "pass"
    assert payload["comparison"]["compatibility_checks"]
    assert payload["comparison"]["aggregate_changes"]
    assert payload["candidate_bad_cases"][0]["category"] == "citation_error"
    markdown = first_markdown.read_text(encoding="utf-8")
    for heading in (
        "# Evaluation Regression Report",
        "## Run Configuration",
        "## Comparability",
        "## Aggregate Metric Changes",
        "## Per-Case Changes",
        "## Bad Cases",
        "## Evaluation Anomalies",
        "## Final Regression Status",
    ):
        assert heading in markdown


def test_report_lists_not_computable_and_infrastructure_outcomes(
    tmp_path: Path,
) -> None:
    baseline_run = _run(_case("case-1"))
    candidate_case = _case("case-1")
    candidate_case["answer"]["coverage"] = {
        "status": "not_computable",
        "result": None,
        "reason": "missing Gold Label",
    }
    candidate_case["answer"]["constraints"] = {
        "status": "timeout",
        "result": None,
        "reason": "judge timeout",
    }
    candidate = _run(candidate_case, run_id="run-b")
    baseline = _baseline(baseline_run)
    comparison = compare_regression(baseline, candidate, _policy())

    json_path, _ = write_regression_report(
        comparison, baseline, candidate, tmp_path
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert {item["status"] for item in payload["evaluation_anomalies"]} == {
        "not_computable",
        "timeout",
    }
