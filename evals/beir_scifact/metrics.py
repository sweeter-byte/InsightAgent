"""Official BEIR metric adapter for persisted document rankings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def official_evaluator_factory() -> Any:
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as exc:
        raise RuntimeError(
            'BEIR benchmark dependencies are missing; run: pip install -e ".[benchmark]"'
        ) from exc
    return EvaluateRetrieval


def compute_official_metrics(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    *,
    evaluator_factory: Callable[..., Any] = official_evaluator_factory,
) -> dict[str, float]:
    """Compute required metrics using BEIR's published evaluation methods."""
    evaluator = evaluator_factory()
    ndcg, _map, recall, _precision = evaluator.evaluate(
        qrels,
        results,
        [5, 10],
    )
    mrr = evaluator.evaluate_custom(qrels, results, [10], metric="mrr")
    return {
        "ndcg_at_10": float(ndcg["NDCG@10"]),
        "recall_at_5": float(recall["Recall@5"]),
        "recall_at_10": float(recall["Recall@10"]),
        "mrr_at_10": float(mrr["MRR@10"]),
    }
