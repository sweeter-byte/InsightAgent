from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.deepresearch_bench.__main__ import build_parser, main


@pytest.mark.parametrize(
    "command",
    ["prepare", "doctor", "smoke", "pilot", "full", "export", "score", "report"],
)
def test_every_subcommand_has_help(command: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "evals.deepresearch_bench", command, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_generation_defaults_are_bounded_and_sequential() -> None:
    args = build_parser().parse_args(["full"])
    assert args.concurrency == 1
    assert args.timeout_seconds > 0
    assert 0 <= args.max_retries <= 3


def test_score_requires_explicit_metric_switch() -> None:
    args = build_parser().parse_args(["score", "--run-dir", "run"])
    assert args.race is False
    assert args.fact is False


def test_doctor_reports_absolute_per_run_checkpoint_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = SimpleNamespace(
        llm=SimpleNamespace(model="research-model", base_url="https://llm.example/v1"),
        embedding_model="embedding-model",
        retrieval=SimpleNamespace(reranker_model="reranker-model"),
        web_search=SimpleNamespace(search_limit=5, fetch_limit=3, timeout=10.0),
    )
    monkeypatch.setattr(
        "evals.deepresearch_bench.__main__.validate_official_checkout",
        lambda path: {"root": str(path)},
    )
    monkeypatch.setattr(
        "evals.deepresearch_bench.__main__.resolve_generation_config",
        lambda: config,
    )

    results_root = tmp_path / "results"
    assert main(
        [
            "doctor",
            "--official-root",
            str(tmp_path / "official"),
            "--results-root",
            str(results_root),
        ]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    generation = output["generation"]
    assert generation["checkpoint_path"] == str(
        results_root.resolve()
        / "<run-id>"
        / "checkpoints"
        / "checkpoints.sqlite"
    )
    assert generation["checkpoint_scope"] == "per-run"
