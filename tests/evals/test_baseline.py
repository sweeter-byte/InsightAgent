from __future__ import annotations

import json
from pathlib import Path

import pytest

import evals.baseline as baseline_module
from evals import (
    BaselineKind,
    BaselineRegistrationError,
    atomic_write_json_artifact,
    create_baseline,
    load_baseline,
    load_evaluation_run,
    write_json_artifact,
)


def _write_run(
    root: Path,
    *,
    evaluation_type: str = "mock_test",
    case_status: str = "completed",
) -> Path:
    run_dir = root / "run-1"
    result_path = run_dir / "cases" / "case-1" / "result.json"
    result = {
        "metadata": {
            "eval_run_id": "run-1",
            "dataset_version": "dataset-v1",
            "dataset_sha256": "a" * 64,
            "git_commit": "abc123",
            "model_name": "model-a",
            "prompt_version": "prompt-v1",
            "index_version": "index-v1",
            "retriever_configuration": {
                "top_k": 5,
                "fixture_version": "fixture-v1",
            },
            "mode": "offline" if evaluation_type == "mock_test" else "live",
            "evaluation_type": evaluation_type,
            "config_fingerprint": "fingerprint-a",
        },
        "case": {
            "case_id": "case-1",
            "query": "query",
            "tags": ["retrieval"],
            "gold_retrieval_ids": ["gold-1"],
            "required_points": [],
            "constraints": [],
            "fixture_set": "fixture-v1",
            "metadata": {"critical": True},
        },
        "retrieval": {
            "case_id": "case-1",
            "retrieved_ids": ["gold-1"],
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "metric_status": "computed",
            "metric_reason": "",
            "pre_rerank_ids": [],
            "post_rerank_ids": [],
            "pre_rerank_reciprocal_rank": None,
            "post_rerank_reciprocal_rank": None,
        },
        "evidence": None,
        "answer": None,
        "runtime": {
            "completed": case_status == "completed",
            "latency_ms": 10,
            "prompt_tokens": None,
            "completion_tokens": None,
            "retrieval_calls": 1,
            "failed_calls": 0,
        },
        "execution_status": case_status,
        "error": None if case_status == "completed" else "RuntimeError: failed",
        "artifact_paths": {
            "result": "cases/case-1/result.json",
            "intermediates": "cases/case-1/artifacts.json",
        },
    }
    write_json_artifact(result_path, result)
    write_json_artifact(run_dir / "cases" / "case-1" / "artifacts.json", {})
    write_json_artifact(
        run_dir / "bad_cases.json",
        {
            "format_version": "evaluation_bad_cases.v1",
            "eval_run_id": "run-1",
            "bad_cases": [],
        },
    )
    write_json_artifact(
        run_dir / "run.json",
        {
            "format_version": "evaluation_run.v1",
            "metadata": result["metadata"],
            "aggregates": {
                "cases": {"total": 1, "completed": 1, "failed": 0},
                "retrieval": {
                    "computed_cases": 1,
                    "mean_recall_at_k": 1.0,
                    "mrr": 1.0,
                },
                "runtime": {
                    "p95_latency_ms": 10.0,
                    "total_prompt_tokens": None,
                    "total_completion_tokens": None,
                },
            },
            "bad_cases": "bad_cases.json",
            "cases": [
                {
                    "case_id": "case-1",
                    "execution_status": case_status,
                    "error": result["error"],
                    "result": "cases/case-1/result.json",
                    "intermediates": "cases/case-1/artifacts.json",
                }
            ],
        },
    )
    return run_dir


def test_baseline_registration_requires_explicit_confirmation(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    with pytest.raises(BaselineRegistrationError, match="confirmed"):
        create_baseline(
            run_dir,
            tmp_path / "baseline.json",
            confirmed=False,
            kind=BaselineKind.MOCK_ONLY,
        )


def test_mock_run_cannot_be_registered_as_real_quality_baseline(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)

    with pytest.raises(BaselineRegistrationError, match="mock_test"):
        create_baseline(
            run_dir,
            tmp_path / "baseline.json",
            confirmed=True,
            kind=BaselineKind.REAL_QUALITY,
        )


def test_manifest_cannot_relabel_mock_case_artifacts_as_real_quality(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["mode"] = "live"
    manifest["metadata"]["evaluation_type"] = "live_test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched run metadata"):
        create_baseline(
            run_dir,
            tmp_path / "baseline.json",
            confirmed=True,
            kind=BaselineKind.REAL_QUALITY,
        )


def test_baseline_preserves_metadata_aggregates_and_full_case_results(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    destination = tmp_path / "baselines" / "dataset-v1.json"

    record = create_baseline(
        run_dir,
        destination,
        confirmed=True,
        kind=BaselineKind.MOCK_ONLY,
    )
    loaded = load_baseline(destination)

    assert record == loaded
    assert loaded.format_version == "evaluation_baseline.v1"
    assert loaded.kind is BaselineKind.MOCK_ONLY
    assert loaded.source_run.metadata["eval_run_id"] == "run-1"
    assert loaded.source_run.aggregates["retrieval"]["mrr"] == 1.0
    assert loaded.source_run.results[0]["case"]["case_id"] == "case-1"
    assert loaded.source_run.results[0]["retrieval"]["retrieved_ids"] == [
        "gold-1"
    ]


def test_load_baseline_rejects_kind_that_does_not_match_source_run(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    destination = tmp_path / "baseline.json"
    create_baseline(
        run_dir,
        destination,
        confirmed=True,
        kind=BaselineKind.MOCK_ONLY,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["kind"] = "real_quality"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_baseline(destination)


def test_load_baseline_rejects_relabelled_embedded_run_metadata(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    destination = tmp_path / "baseline.json"
    create_baseline(
        run_dir,
        destination,
        confirmed=True,
        kind=BaselineKind.MOCK_ONLY,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["kind"] = "real_quality"
    payload["source_run"]["metadata"]["mode"] = "live"
    payload["source_run"]["metadata"]["evaluation_type"] = "live_test"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched run metadata"):
        load_baseline(destination)


def test_baseline_does_not_overwrite_by_default(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    destination = tmp_path / "baseline.json"
    create_baseline(
        run_dir,
        destination,
        confirmed=True,
        kind=BaselineKind.MOCK_ONLY,
    )
    original = destination.read_bytes()

    with pytest.raises(FileExistsError):
        create_baseline(
            run_dir,
            destination,
            confirmed=True,
            kind=BaselineKind.MOCK_ONLY,
        )

    assert destination.read_bytes() == original


def test_baseline_does_not_overwrite_file_created_during_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _write_run(tmp_path)
    destination = tmp_path / "baseline.json"
    original_load = baseline_module.load_evaluation_run

    def load_after_competing_writer(path: str | Path):
        destination.write_text('{"owner": "other-process"}\n', encoding="utf-8")
        return original_load(path)

    monkeypatch.setattr(
        baseline_module,
        "load_evaluation_run",
        load_after_competing_writer,
    )

    with pytest.raises(FileExistsError):
        create_baseline(
            run_dir,
            destination,
            confirmed=True,
            kind=BaselineKind.MOCK_ONLY,
        )

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "owner": "other-process"
    }


def test_incomplete_persisted_run_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "cases" / "case-1" / "result.json").unlink()

    with pytest.raises(ValueError, match="result artifact"):
        load_evaluation_run(run_dir)


def test_missing_referenced_intermediate_artifact_is_rejected(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "cases" / "case-1" / "artifacts.json").unlink()

    with pytest.raises(ValueError, match="intermediate artifact"):
        load_evaluation_run(run_dir)


def test_run_manifest_case_count_must_match_complete_results(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregates"]["cases"]["total"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="case count"):
        load_evaluation_run(run_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [("completed", 0), ("failed", 1)],
)
def test_run_manifest_status_counts_must_match_results(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregates"]["cases"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="case status counts"):
        load_evaluation_run(run_dir)


def test_run_manifest_rejects_out_of_range_quality_aggregate(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregates"]["retrieval"]["mean_recall_at_k"] = 123.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_evaluation_run(run_dir)


def test_atomic_json_write_keeps_existing_destination_if_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "baseline.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        atomic_write_json_artifact(destination, {"new": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob(".baseline.json.*.tmp")) == []
