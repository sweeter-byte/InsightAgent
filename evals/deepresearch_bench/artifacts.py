"""Atomic, resumable run storage for DeepResearch Bench generation."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
BENCHMARK_CHECKPOINT_RELATIVE_PATH = Path("checkpoints/checkpoints.sqlite")


def benchmark_checkpoint_path(run_directory: str | Path) -> Path:
    """Return the absolute SQLite checkpoint path owned by one benchmark run."""
    return Path(run_directory).resolve() / BENCHMARK_CHECKPOINT_RELATIVE_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


class RunStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @classmethod
    def create_or_resume(
        cls,
        results_root: str | Path,
        run_id: str,
        *,
        manifest: Mapping[str, Any],
        config_fingerprint: str,
    ) -> RunStore:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id contains unsafe characters")
        root = Path(results_root).resolve()
        directory = (root / run_id).resolve()
        if directory.parent != root:
            raise ValueError("run_id escapes results root")
        manifest_path = directory / "manifest.json"
        state_path = directory / "state.json"
        if manifest_path.exists():
            existing = _read_json(manifest_path)
            if existing.get("config_fingerprint") != config_fingerprint:
                raise ValueError(
                    "run configuration does not match the existing run; use a new run_id"
                )
            if not state_path.exists():
                raise ValueError("existing run is missing state.json")
            state = _read_json(state_path)
            state["status"] = "running"
            state["updated_at"] = utc_now()
            state["resume_count"] = int(state.get("resume_count", 0)) + 1
            _write_json(state_path, state)
            existing["status"] = "running"
            existing["last_resumed_at"] = utc_now()
            _write_json(manifest_path, existing)
            return cls(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"run directory is not empty: {directory}")
        for relative in ("cases", "exports", "scores/race", "scores/fact", "checkpoints"):
            (directory / relative).mkdir(parents=True, exist_ok=True)
        payload = dict(manifest)
        payload.update(
            {
                "schema_version": 1,
                "config_fingerprint": config_fingerprint,
                "started_at": utc_now(),
                "status": "running",
            }
        )
        _write_json(manifest_path, payload)
        _write_json(
            state_path,
            {
                "schema_version": 1,
                "status": "running",
                "updated_at": utc_now(),
                "resume_count": 0,
                "tasks": {},
            },
        )
        (directory / "benchmark.log").touch()
        return cls(directory)

    def task_status(self, task_id: int) -> str | None:
        state = _read_json(self.directory / "state.json")
        task = state.get("tasks", {}).get(str(task_id))
        return task.get("status") if isinstance(task, dict) else None

    def has_complete_report(self, task_id: int) -> bool:
        case = self.directory / "cases" / str(task_id)
        paths = (case / "metadata.json", case / "execution.json", case / "report.raw.md", case / "report.md")
        if not all(path.is_file() for path in paths):
            return False
        try:
            execution = _read_json(case / "execution.json")
            report = (case / "report.md").read_text(encoding="utf-8")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return (
            execution.get("status") == "succeeded"
            and execution.get("report_generated") is True
            and bool(report.strip())
        )

    def write_case(
        self,
        task: Any,
        execution: Mapping[str, Any],
        *,
        report: str | None = None,
    ) -> None:
        case = self.directory / "cases" / str(task.id)
        case.mkdir(parents=True, exist_ok=True)
        _write_json(
            case / "metadata.json",
            {
                "task_id": task.id,
                "prompt": task.prompt,
                "language": task.language,
                "topic": task.topic,
            },
        )
        if report is not None:
            _write_text(case / "report.raw.md", report)
            _write_text(case / "report.md", report)
        _write_json(case / "execution.json", execution)
        state = _read_json(self.directory / "state.json")
        tasks = state.setdefault("tasks", {})
        tasks[str(task.id)] = {
            "status": execution["status"],
            "report_generated": bool(execution.get("report_generated")),
            "updated_at": utc_now(),
        }
        state["updated_at"] = utc_now()
        _write_json(self.directory / "state.json", state)

    def read_execution(self, task_id: int) -> dict[str, Any]:
        return _read_json(self.directory / "cases" / str(task_id) / "execution.json")

    def export_jsonl(self, output: str | Path | None = None) -> Path:
        destination = Path(output) if output is not None else self.directory / "exports" / "insightagent.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        cases = self.directory / "cases"
        for case in sorted(cases.iterdir(), key=lambda path: int(path.name)):
            execution = _read_json(case / "execution.json")
            if execution.get("status") != "succeeded" or not execution.get("report_generated"):
                continue
            metadata = _read_json(case / "metadata.json")
            article = (case / "report.md").read_text(encoding="utf-8")
            task_id = metadata["task_id"]
            if not isinstance(task_id, int) or task_id in seen:
                raise ValueError("export task IDs must be unique integers")
            if not isinstance(metadata.get("prompt"), str) or not isinstance(article, str) or not article.strip():
                raise ValueError(f"invalid export fields for task {task_id}")
            seen.add(task_id)
            rows.append({"id": task_id, "prompt": metadata["prompt"], "article": article})
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(destination)
        return destination

    def finish(self) -> None:
        state = _read_json(self.directory / "state.json")
        task_states = state.get("tasks", {}).values()
        all_succeeded = all(
            task.get("status") == "succeeded"
            for task in task_states
            if isinstance(task, dict)
        )
        state["status"] = "complete" if all_succeeded else "terminal_with_failures"
        state["updated_at"] = utc_now()
        _write_json(self.directory / "state.json", state)
        manifest = _read_json(self.directory / "manifest.json")
        manifest["status"] = state["status"]
        manifest["ended_at"] = utc_now()
        _write_json(self.directory / "manifest.json", manifest)

    def log(self, message: str) -> None:
        with (self.directory / "benchmark.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")
            handle.flush()
