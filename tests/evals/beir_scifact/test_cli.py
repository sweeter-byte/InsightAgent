from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from evals.beir_scifact import __main__ as cli
from evals.beir_scifact.config import BenchmarkConfig, BenchmarkPaths
from evals.beir_scifact.dataset import SciFactDataset
from evals.beir_scifact.runtime import runtime_environment
from evals.beir_scifact.runtime import safe_run_directory
from evals.beir_scifact.runtime import run_benchmark
from evals.beir_scifact.runtime import _require_prepared


def test_parser_exposes_all_required_subcommands() -> None:
    parser = cli._parser()

    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["index"]).command == "index"
    assert parser.parse_args(["smoke"]).command == "smoke"
    assert parser.parse_args(["full"]).command == "full"
    assert parser.parse_args(["report", "/tmp/run"]).command == "report"


def test_config_defaults_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCIFACT_DENSE_K", "77")
    monkeypatch.setenv("SCIFACT_ROOT", "/tmp/custom-scifact")

    args = cli._parser().parse_args(["full"])
    config, paths = cli._configuration(args)

    assert config.dense_k == 77
    assert paths.root == Path("/tmp/custom-scifact")


def test_command_line_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCIFACT_DENSE_K", "77")

    args = cli._parser().parse_args(["full", "--dense-k", "88"])
    config, _paths = cli._configuration(args)

    assert config.dense_k == 88


def test_report_subcommand_regenerates_without_loading_models(tmp_path: Path) -> None:
    metrics = {
        name: {
            "ndcg_at_10": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
        }
        for name in "ABCD"
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "complete", "run_id": "r", "smoke": True})
    )
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "bad_cases.json").write_text(
        json.dumps({"improvements": [], "regressions": []})
    )

    assert cli.main(["report", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").exists()


def test_real_run_failure_returns_nonzero_and_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("real model failed")

    monkeypatch.setattr(cli, "run_benchmark", fail)

    assert cli.main(["smoke"]) == 2
    assert "real model failed" in capsys.readouterr().err


def test_runtime_environment_records_reproducibility_versions() -> None:
    environment = runtime_environment()

    assert environment["python"]
    assert environment["platform"]
    assert "torch" in environment["packages"]
    assert "sentence-transformers" in environment["packages"]
    assert "cuda_available" in environment


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/absolute", ".", "a/b"])
def test_run_id_must_be_a_safe_single_directory_name(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="run ID"):
        safe_run_directory(tmp_path, run_id)


def test_safe_run_directory_stays_under_results_root(tmp_path: Path) -> None:
    assert safe_run_directory(tmp_path, "valid-run") == tmp_path / "valid-run"


def test_qdrant_constructor_failure_marks_initialized_run_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SciFactDataset(
        corpus={"d1": {"title": "", "text": "text"}},
        queries={"q1": "query"},
        qrels={"q1": {"d1": 1}},
    )
    monkeypatch.setattr("evals.beir_scifact.runtime._require_prepared", lambda paths: None)
    monkeypatch.setattr("evals.beir_scifact.runtime.load_scifact", lambda path: dataset)
    monkeypatch.setattr(
        "evals.beir_scifact.runtime._load_json",
        lambda path: {
            "status": "complete",
            "fingerprint": "ignored",
            "chunk_count": 1,
            "chunk_id_digest": "ignored",
        },
    )
    monkeypatch.setattr(
        "evals.beir_scifact.runtime.index_fingerprint",
        lambda *args, **kwargs: "ignored",
    )
    monkeypatch.setattr(
        "evals.beir_scifact.runtime.expected_chunk_id_digest",
        lambda *args, **kwargs: "ignored",
    )

    class BrokenStore:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("qdrant lock failure")

    monkeypatch.setattr("evals.beir_scifact.runtime.QdrantVectorStore", BrokenStore)

    with pytest.raises(RuntimeError, match="qdrant lock failure"):
        run_benchmark(
            BenchmarkPaths(root=tmp_path / "data"),
            BenchmarkConfig(),
            smoke=True,
            results_root=tmp_path / "results",
            run_id="constructor-failure",
            query_limit=1,
        )

    manifest = json.loads(
        (tmp_path / "results" / "constructor-failure" / "manifest.json").read_text()
    )
    assert manifest["status"] == "failed"


def test_runtime_rejects_modified_extracted_benchmark_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = BenchmarkPaths(root=tmp_path / "data")
    paths.archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(paths.archive_path, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", "corpus\n")
        archive.writestr("scifact/queries.jsonl", "queries\n")
        archive.writestr("scifact/qrels/test.tsv", "header\n")
    paths.dataset_path.mkdir(parents=True)
    (paths.dataset_path / "qrels").mkdir()
    (paths.dataset_path / "corpus.jsonl").write_text("corpus\n")
    (paths.dataset_path / "queries.jsonl").write_text("modified\n")
    (paths.dataset_path / "qrels" / "test.tsv").write_text("header\n")
    monkeypatch.setattr("evals.beir_scifact.runtime.verify_archive", lambda path: "ok")

    with pytest.raises(ValueError, match="differs from verified archive"):
        _require_prepared(paths)
