"""Strict, generation-only loading of official DeepResearch Bench prompts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


OFFICIAL_REPOSITORY = "https://github.com/Ayanami0730/deep_research_bench.git"
OFFICIAL_COMMIT = "f2735b5c3636759c22f9e936c0232de8bf545b67"
OFFICIAL_QUERY_SHA256 = (
    "62b197a264a027fd4e2e795ff322a82dd6d566f5e350596ccc7572539ec987d1"
)
SMOKE_TASK_IDS = (1, 8, 51, 58)
PILOT_TASK_IDS = (1, 8, 15, 22, 29, 36, 43, 50, 51, 58, 65, 72, 79, 86, 93, 100)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    id: int
    topic: str
    language: str
    prompt: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(
    path: str | Path,
    *,
    expected_count: int = 100,
    expected_languages: dict[str, int] | None = None,
) -> list[BenchmarkTask]:
    """Read only query.jsonl and validate the official task contract."""
    source = Path(path)
    rows: list[BenchmarkTask] = []
    seen: set[int] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"blank line at {source}:{line_number}")
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"task at line {line_number} must be an object")
            missing = {"id", "topic", "language", "prompt"} - value.keys()
            if missing:
                raise ValueError(
                    f"task at line {line_number} missing fields: {sorted(missing)}"
                )
            task_id = value["id"]
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
                raise ValueError(f"task id at line {line_number} must be a positive integer")
            if task_id in seen:
                raise ValueError(f"duplicate task id: {task_id}")
            strings: dict[str, str] = {}
            for name in ("topic", "language", "prompt"):
                field = value[name]
                if not isinstance(field, str) or not field.strip():
                    raise ValueError(f"task {task_id} {name} must be a non-empty string")
                strings[name] = field
            if strings["language"] not in {"zh", "en"}:
                raise ValueError(
                    f"task {task_id} language must be 'zh' or 'en'; "
                    f"got {strings['language']!r}"
                )
            seen.add(task_id)
            rows.append(
                BenchmarkTask(
                    id=task_id,
                    topic=strings["topic"],
                    language=strings["language"],
                    prompt=strings["prompt"],
                )
            )
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} tasks, found {len(rows)}")
    expected = expected_languages or {"zh": 50, "en": 50}
    actual = Counter(task.language for task in rows)
    if dict(actual) != expected:
        raise ValueError(
            f"invalid language distribution: expected {expected}, found {dict(actual)}"
        )
    return rows


def select_tasks(
    tasks: Sequence[BenchmarkTask],
    *,
    mode: str,
    task_ids: Iterable[int] | None = None,
    language: str | None = None,
    max_tasks: int | None = None,
) -> list[BenchmarkTask]:
    if mode not in {"smoke", "pilot", "full"}:
        raise ValueError(f"unknown execution mode: {mode!r}")
    if language is not None and language not in {"zh", "en"}:
        raise ValueError("language must be 'zh' or 'en'")
    requested = set(task_ids) if task_ids is not None else None
    if requested is None:
        requested = {
            "smoke": set(SMOKE_TASK_IDS),
            "pilot": set(PILOT_TASK_IDS),
            "full": {task.id for task in tasks},
        }[mode]
    known = {task.id for task in tasks}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown task IDs: {sorted(unknown)}")
    selected = sorted(
        (
            task
            for task in tasks
            if task.id in requested and (language is None or task.language == language)
        ),
        key=lambda task: task.id,
    )
    if max_tasks is not None:
        if isinstance(max_tasks, bool) or max_tasks <= 0:
            raise ValueError("max_tasks must be a positive integer")
        selected = selected[:max_tasks]
    if not selected:
        raise ValueError("task selection is empty")
    return selected
