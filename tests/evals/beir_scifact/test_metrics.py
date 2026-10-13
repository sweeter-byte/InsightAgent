from __future__ import annotations

import pytest

from evals.beir_scifact.metrics import compute_official_metrics


class FakeEvaluateRetrieval:
    calls: list[tuple] = []

    @staticmethod
    def evaluate(qrels, results, k_values):
        FakeEvaluateRetrieval.calls.append(("evaluate", qrels, results, k_values))
        return (
            {"NDCG@10": 0.75},
            {"MAP@10": 0.5},
            {"Recall@5": 0.25, "Recall@10": 0.5},
            {"P@10": 0.1},
        )

    @staticmethod
    def evaluate_custom(qrels, results, k_values, metric):
        FakeEvaluateRetrieval.calls.append(
            ("custom", qrels, results, k_values, metric)
        )
        return {"MRR@10": 0.625}


def test_official_metric_adapter_passes_graded_qrels_unchanged() -> None:
    qrels = {"q1": {"d1": 2, "d2": 1}}
    results = {"q1": {"d2": 2.0, "d3": 1.0}}
    FakeEvaluateRetrieval.calls.clear()

    metrics = compute_official_metrics(
        qrels,
        results,
        evaluator_factory=FakeEvaluateRetrieval,
    )

    assert metrics == {
        "ndcg_at_10": 0.75,
        "recall_at_5": 0.25,
        "recall_at_10": 0.5,
        "mrr_at_10": 0.625,
    }
    assert FakeEvaluateRetrieval.calls == [
        ("evaluate", qrels, results, [5, 10]),
        ("custom", qrels, results, [10], "mrr"),
    ]


def test_real_beir_metrics_match_hand_computed_graded_example() -> None:
    pytest.importorskip("beir")

    metrics = compute_official_metrics(
        {"q1": {"d1": 2, "d2": 1}, "q2": {"x": 1}},
        {"q1": {"d1": 3.0, "d3": 2.0, "d2": 1.0}, "q2": {"x": 1.0}},
    )

    assert metrics == {
        "ndcg_at_10": pytest.approx(0.98197),
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr_at_10": 1.0,
    }
