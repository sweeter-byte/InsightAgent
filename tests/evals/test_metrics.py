from __future__ import annotations

import math

import pytest

from evals import mean_reciprocal_rank, recall_at_k, reciprocal_rank


def test_recall_at_k_counts_unique_gold_hits_in_prefix() -> None:
    assert recall_at_k(("A", "X", "B"), {"A", "B"}, 2) == 0.5
    assert recall_at_k(("A", "A"), {"A", "B"}, 2) == 0.5
    assert recall_at_k(("A", "B"), {"A", "B", "C"}, 5) == pytest.approx(
        2 / 3
    )


def test_recall_at_k_is_undefined_for_empty_gold_set() -> None:
    assert recall_at_k(("A",), set(), 5) is None


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(("X", "A", "B"), {"A", "B"}) == 0.5
    assert reciprocal_rank(("X",), {"A"}) == 0.0
    assert reciprocal_rank(("A",), set()) == 0.0


def test_mean_reciprocal_rank_averages_cases() -> None:
    assert mean_reciprocal_rank((1.0, 0.5, 0.0)) == 0.5
    assert mean_reciprocal_rank(()) == 0.0


@pytest.mark.parametrize("k", [True, False, 1.5, "2", None])
def test_recall_at_k_rejects_non_integer_k(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        recall_at_k(("A",), {"A"}, k)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(("A",), {"A"}, k)


@pytest.mark.parametrize(
    ("retrieved", "gold", "message"),
    [
        (("A", ""), {"A"}, "retrieved_ids"),
        (("A", 3), {"A"}, "retrieved_ids"),
        (("A",), {" "}, "gold_ids"),
    ],
)
def test_metrics_reject_invalid_ids(
    retrieved: tuple[object, ...],
    gold: set[object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        reciprocal_rank(retrieved, gold)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, -0.1, 1.1, math.nan, math.inf])
def test_mean_reciprocal_rank_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="reciprocal rank"):
        mean_reciprocal_rank((value,))  # type: ignore[arg-type]
