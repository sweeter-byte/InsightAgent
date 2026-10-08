"""Deterministic retrieval metrics with explicit input contracts."""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from numbers import Real


def recall_at_k(
    retrieved_ids: Iterable[str],
    gold_ids: Collection[str],
    k: int,
) -> float | None:
    """Return the fraction of Gold IDs found in the first ``k`` results."""
    retrieved = _validated_ids(retrieved_ids, name="retrieved_ids")
    gold = set(_validated_ids(gold_ids, name="gold_ids"))
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if not gold:
        return None
    return len(set(retrieved[:k]) & gold) / len(gold)


def reciprocal_rank(
    retrieved_ids: Iterable[str],
    gold_ids: Collection[str],
) -> float:
    """Return the reciprocal position of the first relevant result."""
    retrieved = _validated_ids(retrieved_ids, name="retrieved_ids")
    gold = set(_validated_ids(gold_ids, name="gold_ids"))
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Iterable[float]) -> float:
    """Average validated per-Case reciprocal ranks."""
    values: list[float] = []
    for value in reciprocal_ranks:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("reciprocal rank values must be real numbers")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                "reciprocal rank values must be finite and between 0 and 1"
            )
        values.append(numeric)
    return sum(values) / len(values) if values else 0.0


def _validated_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of strings")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings") from exc
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"{name} items must be strings")
        if not item.strip():
            raise ValueError(f"{name} items must be non-empty strings")
    return items
