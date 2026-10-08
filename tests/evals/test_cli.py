from __future__ import annotations

import json
from pathlib import Path

from evals.__main__ import main
from tests.evals.test_baseline import _write_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "eval_data" / "research_eval_smoke_v1.jsonl"
FIXTURE = PROJECT_ROOT / "evals" / "fixtures" / "smoke_v1.json"
POLICY = PROJECT_ROOT / "evals" / "policies" / "default_v1.json"


def test_cli_runs_one_selected_offline_mock_case(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--dataset",
            str(DATASET),
            "--fixture",
            str(FIXTURE),
            "--mode",
            "offline",
            "--output",
            str(tmp_path),
            "--case",
            "chunk_overlap_retrieval",
            "--dataset-version",
            "research-eval-smoke-v1",
            "--prompt-version",
            "fixture-judge-v1",
            "--run-id",
            "cli-test",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "MOCK TEST" in output
    manifest = json.loads(
        (tmp_path / "cli-test" / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["mode"] == "offline"
    assert manifest["metadata"]["evaluation_type"] == "mock_test"
    assert [item["case_id"] for item in manifest["cases"]] == [
        "chunk_overlap_retrieval"
    ]


def test_cli_rejects_live_mode_without_explicit_real_components(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--dataset",
            str(DATASET),
            "--mode",
            "live",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert "real components" in capsys.readouterr().err


def test_cli_reports_unknown_case_without_creating_false_results(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--dataset",
            str(DATASET),
            "--fixture",
            str(FIXTURE),
            "--output",
            str(tmp_path),
            "--case",
            "missing-case",
            "--run-id",
            "unknown-test",
        ]
    )

    assert exit_code == 2
    assert "unknown case_id" in capsys.readouterr().err


def test_cli_explicitly_registers_and_compares_a_mock_baseline(
    tmp_path: Path,
    capsys,
) -> None:
    run_exit = main(
        [
            "--dataset",
            str(DATASET),
            "--fixture",
            str(FIXTURE),
            "--output",
            str(tmp_path / "runs"),
            "--run-id",
            "baseline-source",
        ]
    )
    assert run_exit == 0
    capsys.readouterr()
    run_dir = tmp_path / "runs" / "baseline-source"
    baseline_path = tmp_path / "baselines" / "smoke.json"

    assert main(
        [
            "--register-baseline",
            str(run_dir),
            "--baseline-output",
            str(baseline_path),
            "--baseline-kind",
            "mock_only",
            "--confirm-baseline",
        ]
    ) == 0
    assert baseline_path.is_file()
    capsys.readouterr()

    report_dir = tmp_path / "reports"
    compare_exit = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(run_dir),
            "--policy",
            str(POLICY),
            "--report-output",
            str(report_dir),
        ]
    )

    assert compare_exit == 0
    assert (report_dir / "regression.json").is_file()
    assert (report_dir / "regression.md").is_file()
    assert '"status": "pass"' in capsys.readouterr().out


def test_cli_refuses_unconfirmed_baseline_registration(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--register-baseline",
            str(tmp_path / "missing-run"),
            "--baseline-output",
            str(tmp_path / "baseline.json"),
            "--baseline-kind",
            "mock_only",
        ]
    )

    assert exit_code == 2
    assert "--confirm-baseline" in capsys.readouterr().err


def test_cli_returns_distinct_regression_and_incompatible_exit_codes(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = _write_run(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    assert main(
        [
            "--register-baseline",
            str(run_dir),
            "--baseline-output",
            str(baseline_path),
            "--baseline-kind",
            "mock_only",
            "--confirm-baseline",
        ]
    ) == 0
    capsys.readouterr()
    result_path = run_dir / "cases" / "case-1" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["retrieval"]["recall_at_k"] = 0.0
    result["retrieval"]["reciprocal_rank"] = 0.0
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregates"]["retrieval"]["mean_recall_at_k"] = 0.0
    manifest["aggregates"]["retrieval"]["mrr"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    regression_exit = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(run_dir),
            "--policy",
            str(POLICY),
            "--report-output",
            str(tmp_path / "regression-report"),
        ]
    )
    assert regression_exit == 1
    assert '"status": "regression"' in capsys.readouterr().out

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["dataset_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metadata"]["dataset_sha256"] = "b" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")

    incompatible_exit = main(
        [
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(run_dir),
            "--policy",
            str(POLICY),
            "--report-output",
            str(tmp_path / "incompatible-report"),
        ]
    )
    assert incompatible_exit == 2
    assert '"status": "incompatible"' in capsys.readouterr().out
