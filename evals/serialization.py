"""Strict JSON serialization for Evaluation Harness result artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from evals.models import EvaluationResult


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Convert one result to JSON-safe built-in values."""
    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be an EvaluationResult")
    value = _json_value(result)
    if not isinstance(value, dict):
        raise TypeError("evaluation result must serialize to a JSON object")
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def evaluation_result_to_json(
    result: EvaluationResult,
    *,
    indent: int = 2,
) -> str:
    """Serialize one result as standards-compliant UTF-8 JSON text."""
    return json.dumps(
        evaluation_result_to_dict(result),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
    )


def write_evaluation_result(
    path: str | Path,
    result: EvaluationResult,
    *,
    indent: int = 2,
) -> None:
    """Write one Evaluation Result to an explicit path."""
    Path(path).write_text(
        evaluation_result_to_json(result, indent=indent) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON mapping keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")
