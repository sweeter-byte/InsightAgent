"""Strict JSONL loading for human-authored Evaluation Cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.models import EvalCase


_CASE_FIELDS = frozenset(
    {
        "case_id",
        "query",
        "tags",
        "gold_retrieval_ids",
        "required_points",
        "constraints",
        "fixture_set",
        "metadata",
    }
)


class EvalDatasetError(ValueError):
    """A JSONL evaluation dataset failed source-located validation."""


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load and validate human-authored Cases without generating labels."""
    source = Path(path)
    cases: list[EvalCase] = []
    first_line_by_id: dict[str, int] = {}

    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(
                f"{source}:{line_number}:{exc.colno}: invalid JSON: {exc.msg}"
            ) from exc

        case = _parse_case(raw, source=source, line_number=line_number)
        first_line = first_line_by_id.get(case.case_id)
        if first_line is not None:
            raise EvalDatasetError(
                f"{source}:{line_number}: duplicate case_id {case.case_id!r}; "
                f"first defined at line {first_line}"
            )
        first_line_by_id[case.case_id] = line_number
        cases.append(case)

    return cases


def _parse_case(raw: Any, *, source: Path, line_number: int) -> EvalCase:
    if not isinstance(raw, dict):
        _invalid(source, line_number, "row must be a JSON object")

    unknown = set(raw) - _CASE_FIELDS
    if unknown:
        _invalid(
            source,
            line_number,
            "unknown field(s): " + ", ".join(sorted(unknown)),
        )

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        _invalid(source, line_number, "metadata must be a JSON object")

    return EvalCase(
        case_id=_required_text(raw, "case_id", source, line_number),
        query=_required_text(raw, "query", source, line_number),
        tags=_string_tuple(raw, "tags", source, line_number),
        gold_retrieval_ids=_string_tuple(
            raw, "gold_retrieval_ids", source, line_number
        ),
        required_points=_string_tuple(
            raw, "required_points", source, line_number
        ),
        constraints=_string_tuple(raw, "constraints", source, line_number),
        fixture_set=_optional_text(
            raw.get("fixture_set"), "fixture_set", source, line_number
        ),
        metadata=dict(metadata),
    )


def _required_text(
    raw: dict[str, Any], field: str, source: Path, line_number: int
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        _invalid(source, line_number, f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(
    value: Any,
    field: str,
    source: Path,
    line_number: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _invalid(source, line_number, f"{field} must be null or a non-empty string")
    return value.strip()


def _string_tuple(
    raw: dict[str, Any],
    field: str,
    source: Path,
    line_number: int,
) -> tuple[str, ...]:
    value = raw.get(field, [])
    if not isinstance(value, list):
        _invalid(source, line_number, f"{field} must be an array")

    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            _invalid(
                source,
                line_number,
                f"{field} items must be non-empty strings",
            )
        normalized = item.strip()
        if normalized in seen:
            _invalid(
                source,
                line_number,
                f"{field} contains duplicate value {normalized!r}",
            )
        seen.add(normalized)
        items.append(normalized)
    return tuple(items)


def _invalid(source: Path, line_number: int, message: str) -> None:
    raise EvalDatasetError(f"{source}:{line_number}: {message}")
