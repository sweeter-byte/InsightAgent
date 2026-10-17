from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.deepresearch_bench.dataset import (
    PILOT_TASK_IDS,
    SMOKE_TASK_IDS,
    load_tasks,
    select_tasks,
)


def _write_tasks(path: Path, count: int = 100) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task_id in range(1, count + 1):
            language = "zh" if task_id <= count // 2 else "en"
            handle.write(
                json.dumps(
                    {
                        "id": task_id,
                        "topic": f"topic-{task_id}",
                        "language": language,
                        "prompt": f"prompt-{task_id}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_load_tasks_preserves_official_fields_and_distribution(tmp_path: Path) -> None:
    source = tmp_path / "query.jsonl"
    _write_tasks(source)

    tasks = load_tasks(source)

    assert len(tasks) == 100
    assert tasks[0].id == 1
    assert tasks[0].topic == "topic-1"
    assert tasks[0].language == "zh"
    assert tasks[0].prompt == "prompt-1"
    assert sum(task.language == "zh" for task in tasks) == 50
    assert sum(task.language == "en" for task in tasks) == 50


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "id": 1}), "duplicate"),
        (lambda rows: rows[0].pop("topic"), "topic"),
        (lambda rows: rows[0].update(language="fr"), "language"),
    ],
)
def test_load_tasks_rejects_invalid_official_data(tmp_path: Path, mutate, message: str) -> None:
    source = tmp_path / "query.jsonl"
    _write_tasks(source)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_tasks(source)


def test_fixed_smoke_and_pilot_selection_is_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "query.jsonl"
    _write_tasks(source)
    tasks = load_tasks(source)

    assert [task.id for task in select_tasks(tasks, mode="smoke")] == list(
        SMOKE_TASK_IDS
    )
    assert [task.id for task in select_tasks(tasks, mode="pilot")] == list(
        PILOT_TASK_IDS
    )
    assert 3 <= len(SMOKE_TASK_IDS) <= 5
    assert 10 <= len(PILOT_TASK_IDS) <= 20
    assert {task.language for task in select_tasks(tasks, mode="smoke")} == {
        "zh",
        "en",
    }


def test_selection_supports_ids_and_language_without_randomness(tmp_path: Path) -> None:
    source = tmp_path / "query.jsonl"
    _write_tasks(source)
    tasks = load_tasks(source)

    selected = select_tasks(tasks, mode="full", task_ids=[52, 3, 51], language="en")

    assert [task.id for task in selected] == [51, 52]


def test_generation_loader_does_not_read_reference_or_criteria(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "data" / "prompt_data"
    prompt_dir.mkdir(parents=True)
    source = prompt_dir / "query.jsonl"
    _write_tasks(source)
    criteria = tmp_path / "data" / "criteria_data" / "criteria.jsonl"
    reference = tmp_path / "data" / "test_data" / "cleaned_data" / "reference.jsonl"
    criteria.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    criteria.write_text("forbidden", encoding="utf-8")
    reference.write_text("forbidden", encoding="utf-8")
    criteria.chmod(0)
    reference.chmod(0)

    assert len(load_tasks(source)) == 100
