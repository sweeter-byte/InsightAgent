from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.deepresearch_bench.official import OfficialEvaluator


def _official_tree(root: Path) -> None:
    (root / "data" / "prompt_data").mkdir(parents=True)
    (root / "data" / "criteria_data").mkdir(parents=True)
    (root / "data" / "test_data" / "cleaned_data").mkdir(parents=True)
    for relative in (
        "data/prompt_data/query.jsonl",
        "data/criteria_data/criteria.jsonl",
        "data/test_data/cleaned_data/reference.jsonl",
    ):
        (root / relative).write_text("{}\n", encoding="utf-8")
    for relative in (
        "deepresearch_bench_race.py",
        "utils/extract.py",
        "utils/deduplicate.py",
        "utils/scrape.py",
        "utils/validate.py",
        "utils/stat.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _write_fake_outputs(command: list[str]) -> None:
    if "deepresearch_bench_race.py" in command:
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "raw_results.jsonl").write_text('{"id":1,"score":1}\n')
        (output / "race_result.txt").write_text("overall score: 1\n")
    if "--output_path" in command:
        output = Path(command[command.index("--output_path") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '{"id":1,"citations":[]}\n'
            if output.suffix == ".jsonl"
            else "valid_rate: 1\n"
        )


def test_score_requires_explicit_paid_confirmation_and_keys(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    evaluator = OfficialEvaluator(official, tmp_path / "scores", export)

    with pytest.raises(ValueError, match="confirm-paid"):
        evaluator.score(race=True, fact=False, confirm_paid=False, environ={})
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        evaluator.score(race=True, fact=False, confirm_paid=True, environ={})
    with pytest.raises(RuntimeError, match="JINA_API_KEY"):
        evaluator.score(
            race=False,
            fact=True,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret"},
        )


def test_score_invokes_only_official_evaluation_commands(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        _write_fake_outputs(command)
        return type("Completed", (), {"returncode": 0})()

    evaluator = OfficialEvaluator(
        official, tmp_path / "scores", export, command_runner=run
    )
    evaluator.score(
        race=True,
        fact=True,
        confirm_paid=True,
        environ={"OPENROUTER_API_KEY": "secret", "JINA_API_KEY": "jina"},
    )

    flattened = "\n".join(" ".join(call) for call in calls)
    assert "deepresearch_bench_race.py" in flattened
    assert "utils.extract" in flattened
    assert "utils.stat" in flattened
    assert "start_research" not in flattened
    assert all("secret" not in part and "jina" not in part for call in calls for part in call)
    metadata = json.loads(
        (tmp_path / "scores" / "evaluation.json").read_text(encoding="utf-8")
    )
    assert metadata["race_status"] == "succeeded"
    assert metadata["fact_status"] == "succeeded"
    assert metadata["token_usage"] is None


def test_score_persists_failure_without_fabricating_scores(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")

    def fail(command, **kwargs):
        return type("Completed", (), {"returncode": 9})()

    evaluator = OfficialEvaluator(
        official, tmp_path / "scores", export, command_runner=fail
    )
    with pytest.raises(RuntimeError, match="exit code 9"):
        evaluator.score(
            race=True,
            fact=False,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret"},
        )

    metadata = json.loads(
        (tmp_path / "scores" / "evaluation.json").read_text(encoding="utf-8")
    )
    assert metadata["race_status"] == "failed"
    assert metadata["fact_status"] == "not_requested"


def test_score_rejects_empty_export_before_invoking_judge(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text("", encoding="utf-8")
    calls: list[list[str]] = []
    evaluator = OfficialEvaluator(
        official,
        tmp_path / "scores",
        export,
        command_runner=lambda command, **kwargs: calls.append(command),
    )

    with pytest.raises(RuntimeError, match="export.*empty"):
        evaluator.score(
            race=True,
            fact=False,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret"},
        )
    assert calls == []


def test_zero_exit_without_score_artifacts_is_failure(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    evaluator = OfficialEvaluator(
        official,
        tmp_path / "scores",
        export,
        command_runner=lambda command, **kwargs: type(
            "Completed", (), {"returncode": 0}
        )(),
    )

    with pytest.raises(RuntimeError, match="RACE raw results"):
        evaluator.score(
            race=True,
            fact=False,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret"},
        )
    metadata = json.loads((tmp_path / "scores" / "evaluation.json").read_text())
    assert metadata["race_status"] == "failed"


def test_keyboard_interrupt_stops_before_fact(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def interrupt(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt

    evaluator = OfficialEvaluator(
        official, tmp_path / "scores", export, command_runner=interrupt
    )
    with pytest.raises(KeyboardInterrupt):
        evaluator.score(
            race=True,
            fact=True,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret", "JINA_API_KEY": "jina"},
        )

    assert len(calls) == 1
    metadata = json.loads((tmp_path / "scores" / "evaluation.json").read_text())
    assert metadata["race_status"] == "failed"
    assert metadata["fact_status"] == "pending"


def test_combined_score_attempts_fact_after_race_failure(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if "deepresearch_bench_race.py" not in command:
            _write_fake_outputs(command)
        return type(
            "Completed",
            (),
            {"returncode": 7 if "deepresearch_bench_race.py" in command else 0},
        )()

    evaluator = OfficialEvaluator(official, tmp_path / "scores", export, command_runner=run)
    with pytest.raises(RuntimeError, match="exit code 7"):
        evaluator.score(
            race=True,
            fact=True,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret", "JINA_API_KEY": "jina"},
        )

    metadata = json.loads((tmp_path / "scores" / "evaluation.json").read_text())
    assert metadata["race_status"] == "failed"
    assert metadata["fact_status"] == "succeeded"
    assert any("utils.stat" in call for call in calls)


def test_separate_fact_run_preserves_completed_race_status(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")

    def run(command, **kwargs):
        _write_fake_outputs(command)
        return type("Completed", (), {"returncode": 0})()

    evaluator = OfficialEvaluator(official, tmp_path / "scores", export, command_runner=run)
    evaluator.score(
        race=True,
        fact=False,
        confirm_paid=True,
        environ={"OPENROUTER_API_KEY": "secret"},
    )
    evaluator.score(
        race=False,
        fact=True,
        confirm_paid=True,
        environ={"OPENROUTER_API_KEY": "secret", "JINA_API_KEY": "jina"},
    )

    metadata = json.loads((tmp_path / "scores" / "evaluation.json").read_text())
    assert metadata["race_status"] == "succeeded"
    assert metadata["fact_status"] == "succeeded"


def test_changed_export_requires_force_before_reusing_scores(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        _write_fake_outputs(command)
        return type("Completed", (), {"returncode": 0})()

    evaluator = OfficialEvaluator(official, tmp_path / "scores", export, command_runner=run)
    evaluator.score(
        race=True,
        fact=False,
        confirm_paid=True,
        environ={"OPENROUTER_API_KEY": "secret"},
    )
    prior_calls = len(calls)
    export.write_text('{"id":1,"prompt":"p","article":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="export.*changed"):
        evaluator.score(
            race=True,
            fact=False,
            confirm_paid=True,
            environ={"OPENROUTER_API_KEY": "secret"},
        )
    assert len(calls) == prior_calls


def test_changed_judge_configuration_requires_force(tmp_path: Path) -> None:
    official = tmp_path / "official"
    _official_tree(official)
    export = tmp_path / "insightagent.jsonl"
    export.write_text('{"id":1,"prompt":"p","article":"a"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        _write_fake_outputs(command)
        return type("Completed", (), {"returncode": 0})()

    evaluator = OfficialEvaluator(official, tmp_path / "scores", export, command_runner=run)
    evaluator.score(
        race=True,
        fact=False,
        confirm_paid=True,
        environ={"OPENROUTER_API_KEY": "secret"},
    )
    prior_calls = len(calls)

    with pytest.raises(ValueError, match="configuration has changed"):
        evaluator.score(
            race=True,
            fact=False,
            confirm_paid=True,
            environ={
                "OPENROUTER_API_KEY": "secret",
                "RACE_MODEL": "custom/race-model",
            },
        )
    assert len(calls) == prior_calls


def test_unscored_metrics_are_null_not_zero(tmp_path: Path) -> None:
    from evals.deepresearch_bench.reporting import aggregate_metrics

    run_dir = tmp_path / "run"
    case = run_dir / "cases" / "1"
    case.mkdir(parents=True)
    (case / "execution.json").write_text(
        json.dumps(
            {
                "task_id": 1,
                "status": "succeeded",
                "report_generated": True,
                "elapsed_seconds": 2.0,
                "input_tokens": None,
                "output_tokens": None,
            }
        ),
        encoding="utf-8",
    )

    metrics = aggregate_metrics(run_dir, total_tasks=1)

    assert metrics["race"] is None
    assert metrics["fact"] is None
    assert metrics["race_unscored_tasks"] == 1
    assert metrics["fact_unscored_tasks"] == 1
    assert metrics["tokens"]["input_total"] is None


def test_metrics_reject_scores_for_a_changed_export(tmp_path: Path) -> None:
    from evals.deepresearch_bench.dataset import sha256_file
    from evals.deepresearch_bench.reporting import aggregate_metrics

    run_dir = tmp_path / "run"
    (run_dir / "cases").mkdir(parents=True)
    export = run_dir / "exports" / "insightagent.jsonl"
    export.parent.mkdir()
    export.write_text('{"id":1,"prompt":"p","article":"old"}\n', encoding="utf-8")
    scores = run_dir / "scores"
    (scores / "race").mkdir(parents=True)
    (scores / "race" / "race_result.txt").write_text("overall score: 1\n")
    (scores / "evaluation.json").write_text(
        json.dumps(
            {
                "protocol": "official",
                "race_status": "succeeded",
                "fact_status": "not_requested",
                "export_sha256": sha256_file(export),
            }
        )
    )
    export.write_text('{"id":1,"prompt":"p","article":"new"}\n', encoding="utf-8")

    metrics = aggregate_metrics(run_dir, total_tasks=1)

    assert metrics["scoring_stale"] is True
    assert metrics["race"] is None
