"""Deterministic, bounded metadata views for LLM-facing Evidence payloads."""

from __future__ import annotations

import json
from typing import Any


MAX_METADATA_BYTES_PER_EVIDENCE = 2048
MAX_TOTAL_METADATA_BYTES = 12000
MAX_METADATA_STRING_CHARS = 1000
MAX_METADATA_DEPTH = 4
MAX_METADATA_COLLECTION_ITEMS = 20
METADATA_OMISSION_MARKER = "[... metadata omitted ...]"
METADATA_PRIORITY_KEYS = (
    "final_url",
    "original_url",
    "filename",
    "page",
    "page_count",
    "chunk_id",
    "document_id",
    "source_type",
    "chunk_index",
    "start_char",
    "end_char",
    "retrieval_score",
    "content_type",
    "mime_type",
    "search_rank",
    "search_score",
    "search_query",
    "title",
    "search_title",
    "search_snippet",
)


def project_metadata_collection(
    metadata_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return detached metadata views within per-item and collection budgets."""
    bounded = [_bound_metadata(metadata) for metadata in metadata_items]
    budgets = _allocate_metadata_budgets(bounded)
    return [
        _project_metadata(metadata, budget)
        for metadata, budget in zip(bounded, budgets, strict=True)
    ]


def _serialize_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _serialized_size(value: Any) -> int:
    return len(_serialize_json(value).encode("utf-8"))


def _bound_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Detach and structurally bound metadata before serializing it."""
    bounded = _bound_metadata_value(metadata, depth=0, active=set())
    if not isinstance(bounded, dict):  # pragma: no cover - guarded by typing
        raise TypeError("metadata must project to a JSON object")
    return bounded


def _allocate_metadata_budgets(
    metadata: list[dict[str, Any]],
) -> list[int]:
    """Fairly allocate the serialized metadata budget across Evidence."""
    target_sizes = [
        min(_serialized_size(item), MAX_METADATA_BYTES_PER_EVIDENCE)
        for item in metadata
    ]
    collection_overhead = len(metadata) + 1
    available_value_bytes = MAX_TOTAL_METADATA_BYTES - collection_overhead
    if sum(target_sizes) <= available_value_bytes:
        return target_sizes

    budgets = [0] * len(target_sizes)
    remaining_budget = available_value_bytes
    active = set(range(len(target_sizes)))
    while active:
        fair_share = remaining_budget // len(active)
        completed = [
            index for index in active if target_sizes[index] <= fair_share
        ]
        if completed:
            for index in completed:
                budgets[index] = target_sizes[index]
                remaining_budget -= budgets[index]
                active.remove(index)
            continue

        ordered_active = sorted(active)
        for index in ordered_active:
            budgets[index] = fair_share
        remainder = remaining_budget - fair_share * len(ordered_active)
        for index in ordered_active[:remainder]:
            budgets[index] += 1
        break
    return budgets


def _project_metadata(
    metadata: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    """Build a deterministic, detached metadata view within ``budget``."""
    projected: dict[str, Any] = {}
    priority = [key for key in METADATA_PRIORITY_KEYS if key in metadata]
    priority_set = set(priority)
    remaining = sorted(
        (key for key in metadata if key not in priority_set),
        key=lambda key: (type(key).__name__, repr(key)),
    )
    for key in [*priority, *remaining]:
        value = metadata[key]
        candidate = {**projected, key: value}
        if _serialized_size(candidate) <= budget:
            projected[key] = value
            continue
        if isinstance(value, str):
            fitted = _fit_string_field(projected, key, value, budget)
            if fitted is not None:
                projected[key] = fitted
            continue
        candidate = {**projected, key: METADATA_OMISSION_MARKER}
        if _serialized_size(candidate) <= budget:
            projected[key] = METADATA_OMISSION_MARKER
    return projected


def _bound_metadata_value(
    value: Any,
    *,
    depth: int,
    active: set[int],
) -> Any:
    if isinstance(value, str):
        if len(value) <= MAX_METADATA_STRING_CHARS:
            return value
        retained = MAX_METADATA_STRING_CHARS - len(METADATA_OMISSION_MARKER)
        return value[:retained] + METADATA_OMISSION_MARKER
    if isinstance(value, dict):
        if depth >= MAX_METADATA_DEPTH:
            return METADATA_OMISSION_MARKER
        identity = id(value)
        if identity in active:
            raise ValueError("metadata contains a cyclic dictionary")
        active.add(identity)
        if depth == 0:
            priority = [key for key in METADATA_PRIORITY_KEYS if key in value]
            priority_set = set(priority)
            keys = list(priority[:MAX_METADATA_COLLECTION_ITEMS])
            remaining_slots = MAX_METADATA_COLLECTION_ITEMS - len(keys)
            if remaining_slots:
                for key in value:
                    if key in priority_set:
                        continue
                    keys.append(key)
                    remaining_slots -= 1
                    if not remaining_slots:
                        break
        else:
            keys = []
            for key in value:
                keys.append(key)
                if len(keys) == MAX_METADATA_COLLECTION_ITEMS:
                    break
        try:
            return {
                key: _bound_metadata_value(
                    value[key],
                    depth=depth + 1,
                    active=active,
                )
                for key in keys
            }
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        if depth >= MAX_METADATA_DEPTH:
            return METADATA_OMISSION_MARKER
        identity = id(value)
        if identity in active:
            raise ValueError("metadata contains a cyclic sequence")
        active.add(identity)
        try:
            return [
                _bound_metadata_value(
                    item,
                    depth=depth + 1,
                    active=active,
                )
                for item in value[:MAX_METADATA_COLLECTION_ITEMS]
            ]
        finally:
            active.remove(identity)
    return value


def _fit_string_field(
    projected: dict[str, Any],
    key: str,
    value: str,
    budget: int,
) -> str | None:
    """Return the longest deterministic string prefix that fits."""
    low = 0
    high = len(value)
    fitted: str | None = None
    while low <= high:
        retained = (low + high) // 2
        candidate_value = value[:retained] + METADATA_OMISSION_MARKER
        candidate = {**projected, key: candidate_value}
        if _serialized_size(candidate) <= budget:
            fitted = candidate_value
            low = retained + 1
        else:
            high = retained - 1
    return fitted
