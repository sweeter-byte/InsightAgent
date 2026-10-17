"""Pin and validate the external official DeepResearch Bench checkout."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.deepresearch_bench.dataset import (
    OFFICIAL_COMMIT,
    OFFICIAL_QUERY_SHA256,
    OFFICIAL_REPOSITORY,
    load_tasks,
    sha256_file,
)


CommandRunner = Callable[..., Any]
GENERATION_REQUIRED_FILES = (
    "data/prompt_data/query.jsonl",
)
OFFICIAL_REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "deepresearch_bench_race.py",
    "run_benchmark.sh",
    "data/prompt_data/query.jsonl",
    "data/criteria_data/criteria.jsonl",
    "data/test_data/cleaned_data/reference.jsonl",
    "utils/api.py",
    "utils/__init__.py",
    "utils/io_utils.py",
    "utils/json_extractor.py",
    "utils/score_calculator.py",
    "utils/clean_article.py",
    "utils/extract.py",
    "utils/deduplicate.py",
    "utils/scrape.py",
    "utils/validate.py",
    "utils/stat.py",
    "prompt/__init__.py",
    "prompt/clean_prompt.py",
    "prompt/score_prompt_en.py",
    "prompt/score_prompt_zh.py",
)


def validate_official_checkout(
    root: str | Path,
    *,
    expected_commit: str = OFFICIAL_COMMIT,
    expected_query_sha256: str = OFFICIAL_QUERY_SHA256,
    command_runner: CommandRunner = subprocess.run,
    include_evaluation_files: bool = True,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    query = directory / "data" / "prompt_data" / "query.jsonl"
    required_files = (
        OFFICIAL_REQUIRED_FILES if include_evaluation_files else GENERATION_REQUIRED_FILES
    )
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"official checkout is incomplete: {missing}")
    completed = command_runner(
        ["git", "rev-parse", "HEAD"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve official checkout commit: {completed.stderr.strip()}")
    commit = completed.stdout.strip()
    if commit != expected_commit:
        raise ValueError(
            f"official checkout commit mismatch: expected {expected_commit}, found {commit}"
        )
    if include_evaluation_files:
        status = command_runner(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0:
            raise RuntimeError(
                f"cannot inspect official checkout status: {status.stderr.strip()}"
            )
        dirty = [
            line
            for line in status.stdout.splitlines()
            if line.strip()
            and "__pycache__/" not in line
            and not line.rstrip().endswith(".pyc")
        ]
        if dirty:
            raise ValueError("official checkout has local modifications or untracked files")
    query_sha = sha256_file(query)
    if query_sha != expected_query_sha256:
        raise ValueError(
            "official query SHA256 mismatch: "
            f"expected {expected_query_sha256}, found {query_sha}"
        )
    tasks = load_tasks(query)
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "query_sha256": query_sha,
        "files_sha256": {
            name: sha256_file(directory / name) for name in required_files
        },
        "task_count": len(tasks),
        "language_counts": {
            "zh": sum(task.language == "zh" for task in tasks),
            "en": sum(task.language == "en" for task in tasks),
        },
    }


def prepare_official_checkout(
    root: str | Path,
    *,
    revision: str = OFFICIAL_COMMIT,
    command_runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists() and not (directory / ".git").is_dir():
        raise FileExistsError(f"prepare target exists and is not a Git checkout: {directory}")
    if not directory.exists():
        _run(
            command_runner,
            ["git", "clone", "--no-checkout", OFFICIAL_REPOSITORY, str(directory)],
        )
    completed = command_runner(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        _run(command_runner, ["git", "fetch", "origin", revision], cwd=directory)
    _run(command_runner, ["git", "checkout", "--detach", revision], cwd=directory)
    metadata = validate_official_checkout(
        directory,
        expected_commit=revision,
        expected_query_sha256=OFFICIAL_QUERY_SHA256,
        command_runner=command_runner,
    )
    manifest = directory.parent / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _run(
    runner: CommandRunner,
    command: list[str],
    *,
    cwd: Path | None = None,
) -> None:
    completed = runner(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command[:3])}: "
            f"{completed.stderr.strip()}"
        )
