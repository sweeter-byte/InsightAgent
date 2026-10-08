"""Explicit, immutable registration of versioned Evaluation Baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from evals.results import (
    PersistedRun,
    load_evaluation_run,
    persisted_run_from_payload,
    persisted_run_payload,
)
from evals.serialization import atomic_write_json_artifact


BASELINE_FORMAT_VERSION = "evaluation_baseline.v1"


class BaselineKind(str, Enum):
    MOCK_ONLY = "mock_only"
    REAL_QUALITY = "real_quality"


class BaselineRegistrationError(ValueError):
    """A Run is not eligible for the requested Baseline role."""


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    format_version: str
    kind: BaselineKind
    confirmed: bool
    source_run: PersistedRun


def create_baseline(
    run_path: str | Path,
    destination: str | Path,
    *,
    confirmed: bool,
    kind: BaselineKind,
    overwrite: bool = False,
) -> BaselineRecord:
    """Register one complete Run; normal Evaluation never calls this."""
    if confirmed is not True:
        raise BaselineRegistrationError(
            "Baseline registration must be explicitly confirmed"
        )
    if not isinstance(kind, BaselineKind):
        raise TypeError("kind must be a BaselineKind")
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Baseline already exists: {destination_path}")

    loaded = load_evaluation_run(run_path)
    evaluation_type = loaded.metadata.get("evaluation_type")
    if evaluation_type == "mock_test" and kind is BaselineKind.REAL_QUALITY:
        raise BaselineRegistrationError(
            "mock_test Run cannot be registered as a real quality Baseline"
        )
    if evaluation_type == "live_test" and kind is BaselineKind.MOCK_ONLY:
        raise BaselineRegistrationError(
            "live_test Run cannot be registered as a mock-only Baseline"
        )
    if evaluation_type not in {"mock_test", "live_test"}:
        raise BaselineRegistrationError("Run has an unknown evaluation_type")

    record = BaselineRecord(
        format_version=BASELINE_FORMAT_VERSION,
        kind=kind,
        confirmed=True,
        source_run=replace(loaded, run_directory=None),
    )
    atomic_write_json_artifact(
        destination_path,
        baseline_payload(record),
        overwrite=overwrite,
    )
    return record


def load_baseline(path: str | Path) -> BaselineRecord:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Baseline does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Baseline is invalid JSON: {source}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Baseline must be a JSON object")
    version = payload.get("format_version")
    if version != BASELINE_FORMAT_VERSION:
        raise ValueError(f"unsupported Baseline format_version: {version!r}")
    try:
        kind = BaselineKind(payload.get("kind"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Baseline kind is invalid") from exc
    if payload.get("confirmed") is not True:
        raise ValueError("Baseline confirmation marker is missing")
    source_run = persisted_run_from_payload(payload.get("source_run"))
    expected_type = (
        "mock_test" if kind is BaselineKind.MOCK_ONLY else "live_test"
    )
    if source_run.metadata.get("evaluation_type") != expected_type:
        raise ValueError("Baseline kind does not match source Run evaluation_type")
    return BaselineRecord(
        format_version=version,
        kind=kind,
        confirmed=True,
        source_run=source_run,
    )


def baseline_payload(record: BaselineRecord) -> dict[str, Any]:
    return {
        "format_version": record.format_version,
        "kind": record.kind.value,
        "confirmed": record.confirmed,
        "source_run": persisted_run_payload(record.source_run),
    }
