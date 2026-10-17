from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.deepresearch_bench.prepare import OFFICIAL_REQUIRED_FILES, validate_official_checkout


def _required_tree(root: Path) -> None:
    for relative in OFFICIAL_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_validate_official_checkout_records_commit_and_query_hash(
    tmp_path: Path,
) -> None:
    query = tmp_path / "data" / "prompt_data" / "query.jsonl"
    _required_tree(tmp_path)
    query_data = "".join(
        json.dumps(
            {
                "id": task_id,
                "topic": "topic",
                "language": "zh" if task_id <= 50 else "en",
                "prompt": f"prompt {task_id}",
            }
        )
        + "\n"
        for task_id in range(1, 101)
    )
    query.write_text(query_data, encoding="utf-8")
    commit = "a" * 40

    def run(command, **kwargs):
        stdout = "" if command[1] == "status" else commit + "\n"
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": stdout, "stderr": ""},
        )()

    metadata = validate_official_checkout(
        tmp_path,
        expected_commit=commit,
        expected_query_sha256=hashlib.sha256(query_data.encode()).hexdigest(),
        command_runner=run,
    )

    assert metadata["commit"] == commit
    assert metadata["query_sha256"] == hashlib.sha256(query_data.encode()).hexdigest()
    assert metadata["files_sha256"]["deepresearch_bench_race.py"] == hashlib.sha256(b"").hexdigest()


def test_validate_official_checkout_rejects_wrong_revision(tmp_path: Path) -> None:
    query = tmp_path / "data" / "prompt_data" / "query.jsonl"
    _required_tree(tmp_path)
    query.write_text("query-data", encoding="utf-8")

    def run(command, **kwargs):
        return type(
            "Completed", (), {"returncode": 0, "stdout": "b" * 40 + "\n", "stderr": ""}
        )()

    with pytest.raises(ValueError, match="commit"):
        validate_official_checkout(
            tmp_path,
            expected_commit="a" * 40,
            expected_query_sha256=hashlib.sha256(b"query-data").hexdigest(),
            command_runner=run,
        )


def test_validate_official_checkout_rejects_dirty_tree(tmp_path: Path) -> None:
    query = tmp_path / "data" / "prompt_data" / "query.jsonl"
    _required_tree(tmp_path)
    query_data = "".join(
        json.dumps(
            {
                "id": task_id,
                "topic": "topic",
                "language": "zh" if task_id <= 50 else "en",
                "prompt": f"prompt {task_id}",
            }
        )
        + "\n"
        for task_id in range(1, 101)
    )
    query.write_text(query_data, encoding="utf-8")
    commit = "a" * 40

    def run(command, **kwargs):
        stdout = " M utils/api.py\n" if command[1] == "status" else commit + "\n"
        return type(
            "Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()

    with pytest.raises(ValueError, match="local modifications"):
        validate_official_checkout(
            tmp_path,
            expected_commit=commit,
            expected_query_sha256=hashlib.sha256(query_data.encode()).hexdigest(),
            command_runner=run,
        )


def test_generation_validation_does_not_read_evaluation_files(tmp_path: Path) -> None:
    _required_tree(tmp_path)
    query = tmp_path / "data" / "prompt_data" / "query.jsonl"
    query_data = "".join(
        json.dumps(
            {
                "id": task_id,
                "topic": "topic",
                "language": "zh" if task_id <= 50 else "en",
                "prompt": f"prompt {task_id}",
            }
        )
        + "\n"
        for task_id in range(1, 101)
    )
    query.write_text(query_data, encoding="utf-8")
    commit = "a" * 40

    def run(command, **kwargs):
        stdout = "" if command[1] == "status" else commit + "\n"
        return type(
            "Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""}
        )()

    metadata = validate_official_checkout(
        tmp_path,
        expected_commit=commit,
        expected_query_sha256=hashlib.sha256(query_data.encode()).hexdigest(),
        command_runner=run,
        include_evaluation_files=False,
    )

    assert set(metadata["files_sha256"]) == {"data/prompt_data/query.jsonl"}
