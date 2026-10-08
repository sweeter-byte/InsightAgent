"""Strict loading of complete, already-persisted Evaluation Runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_FORMAT_VERSION = "evaluation_run.v1"

_RATE_AGGREGATE_PATHS = (
    "retrieval.mean_recall_at_k",
    "retrieval.mrr",
    "evidence.provenance_valid_rate",
    "evidence.relevance_pass_rate",
    "evidence.coverage_pass_rate",
    "answer.grounded_claim_pass_rate",
    "answer.coverage_pass_rate",
    "answer.constraint_pass_rate",
    "runtime.completion_rate",
)


@dataclass(frozen=True, slots=True)
class PersistedRun:
    format_version: str
    metadata: dict[str, Any]
    aggregates: dict[str, Any]
    results: tuple[dict[str, Any], ...]
    bad_cases: tuple[dict[str, Any], ...]
    run_directory: Path | None = None


def load_evaluation_run(path: str | Path) -> PersistedRun:
    """Load one complete Run manifest and every referenced Case result."""
    source = Path(path)
    manifest_path = source / "run.json" if source.is_dir() else source
    run_directory = manifest_path.parent
    manifest = _object(manifest_path, "run manifest")
    version = manifest.get("format_version")
    if version != RUN_FORMAT_VERSION:
        raise ValueError(
            f"unsupported Evaluation Run format_version: {version!r}"
        )
    metadata = _mapping(manifest.get("metadata"), "run metadata")
    aggregates = _mapping(manifest.get("aggregates"), "run aggregates")
    run_id = _validate_run_metadata(metadata, label="run metadata")
    case_entries = manifest.get("cases")
    if not isinstance(case_entries, list):
        raise ValueError("run cases must be an array")

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry_value in enumerate(case_entries):
        entry = _mapping(entry_value, f"run case {index}")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"run case {index} case_id must be non-empty")
        if case_id in seen:
            raise ValueError(f"duplicate persisted case_id: {case_id}")
        seen.add(case_id)
        result_path = _artifact_path(
            run_directory,
            entry.get("result"),
            label=f"case {case_id} result artifact",
        )
        intermediate_path = _artifact_path(
            run_directory,
            entry.get("intermediates"),
            label=f"case {case_id} intermediate artifact",
        )
        result = _object(result_path, f"case {case_id} result artifact")
        _object(intermediate_path, f"case {case_id} intermediate artifact")
        stored_case = result.get("case")
        if not isinstance(stored_case, dict) or stored_case.get("case_id") != case_id:
            raise ValueError(f"case {case_id} result artifact has mismatched case_id")
        stored_metadata = result.get("metadata")
        if not isinstance(stored_metadata, dict) or stored_metadata != metadata:
            raise ValueError(f"case {case_id} result artifact has mismatched run metadata")
        if entry.get("execution_status") != result.get("execution_status"):
            raise ValueError(
                f"case {case_id} result artifact has mismatched execution status"
            )
        results.append(result)

    _validate_case_counts(aggregates, results, label="run")
    _validate_rate_aggregates(aggregates, label="run")

    bad_cases: tuple[dict[str, Any], ...] = ()
    bad_case_reference = manifest.get("bad_cases")
    if bad_case_reference is not None:
        bad_case_path = _artifact_path(
            run_directory,
            bad_case_reference,
            label="Bad Case artifact",
        )
        bad_case_payload = _object(bad_case_path, "Bad Case artifact")
        items = bad_case_payload.get("bad_cases")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("Bad Case artifact bad_cases must be an object array")
        bad_cases = tuple(dict(item) for item in items)

    return PersistedRun(
        format_version=version,
        metadata=metadata,
        aggregates=aggregates,
        results=tuple(results),
        bad_cases=bad_cases,
        run_directory=run_directory,
    )


def persisted_run_payload(run: PersistedRun) -> dict[str, Any]:
    return {
        "format_version": run.format_version,
        "metadata": run.metadata,
        "aggregates": run.aggregates,
        "results": list(run.results),
        "bad_cases": list(run.bad_cases),
    }


def persisted_run_from_payload(value: object) -> PersistedRun:
    payload = _mapping(value, "embedded source_run")
    version = payload.get("format_version")
    if version != RUN_FORMAT_VERSION:
        raise ValueError(f"unsupported embedded Run format_version: {version!r}")
    metadata = _mapping(payload.get("metadata"), "embedded run metadata")
    aggregates = _mapping(payload.get("aggregates"), "embedded run aggregates")
    run_id = _validate_run_metadata(metadata, label="embedded run metadata")
    raw_results = payload.get("results")
    raw_bad_cases = payload.get("bad_cases", [])
    if not isinstance(raw_results, list) or any(
        not isinstance(item, dict) for item in raw_results
    ):
        raise ValueError("embedded run results must be an object array")
    if not isinstance(raw_bad_cases, list) or any(
        not isinstance(item, dict) for item in raw_bad_cases
    ):
        raise ValueError("embedded run bad_cases must be an object array")
    case_ids = [
        item.get("case", {}).get("case_id")
        if isinstance(item.get("case"), dict)
        else None
        for item in raw_results
    ]
    if any(not isinstance(item, str) or not item for item in case_ids) or len(
        set(case_ids)
    ) != len(case_ids):
        raise ValueError("embedded run contains duplicate or invalid case IDs")
    for case_id, item in zip(case_ids, raw_results, strict=True):
        stored_metadata = item.get("metadata")
        if not isinstance(stored_metadata, dict) or stored_metadata != metadata:
            raise ValueError(
                f"embedded case {case_id} result has mismatched run metadata"
            )
    _validate_case_counts(aggregates, raw_results, label="embedded run")
    _validate_rate_aggregates(aggregates, label="embedded run")
    return PersistedRun(
        format_version=version,
        metadata=metadata,
        aggregates=aggregates,
        results=tuple(dict(item) for item in raw_results),
        bad_cases=tuple(dict(item) for item in raw_bad_cases),
    )


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}: {exc.msg}") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _artifact_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} path must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the Run directory") from exc
    return resolved


def _validate_run_metadata(metadata: dict[str, Any], *, label: str) -> str:
    run_id = metadata.get("eval_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{label} eval_run_id must be a non-empty string")
    pair = (metadata.get("mode"), metadata.get("evaluation_type"))
    if pair not in {("offline", "mock_test"), ("live", "live_test")}:
        raise ValueError(
            f"{label} mode and evaluation_type must be offline/mock_test "
            "or live/live_test"
        )
    return run_id


def _validate_case_counts(
    aggregates: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    statuses = [result.get("execution_status") for result in results]
    if any(status not in {"completed", "failed"} for status in statuses):
        raise ValueError(f"{label} contains an invalid case execution status")
    case_aggregates = aggregates.get("cases")
    if not isinstance(case_aggregates, dict):
        raise ValueError(f"{label} aggregates.cases must be a JSON object")
    expected = {
        "total": len(results),
        "completed": statuses.count("completed"),
        "failed": statuses.count("failed"),
    }
    for name, value in expected.items():
        actual = case_aggregates.get(name)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != value:
            if name == "total":
                raise ValueError(
                    f"{label} aggregate case count does not match persisted results"
                )
            raise ValueError(
                f"{label} aggregate case status counts do not match persisted results"
            )


def _validate_rate_aggregates(aggregates: dict[str, Any], *, label: str) -> None:
    missing = object()
    for path in _RATE_AGGREGATE_PATHS:
        value: object = aggregates
        for component in path.split("."):
            if not isinstance(value, dict) or component not in value:
                value = missing
                break
            value = value[component]
        if value is missing or value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{label} aggregate {path} must be between 0 and 1")
