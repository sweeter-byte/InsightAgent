from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.beir_scifact.artifacts import (
    append_prediction,
    build_prediction_record,
    extract_bad_cases,
    finalize_run,
    initialize_run,
    mark_run_failed,
)
from evals.beir_scifact.ranking import DocumentRankingEntry
from evals.beir_scifact.retrieval import ExperimentQueryResult


class FakeEvaluator:
    @staticmethod
    def evaluate(qrels, results, k_values):
        return (
            {"NDCG@10": 1.0},
            {},
            {"Recall@5": 1.0, "Recall@10": 1.0},
            {},
        )

    @staticmethod
    def evaluate_custom(qrels, results, k_values, metric):
        return {"MRR@10": 1.0}


def _result(experiment: str, *, doc_id: str = "d1", latency: float = 10.0):
    return ExperimentQueryResult(
        query_id="q1",
        experiment=experiment,
        documents=(DocumentRankingEntry(doc_id, f"chunk-{doc_id}", 1, 0.5),),
        latency_ms=latency,
        candidate_chunk_ids=(f"chunk-{doc_id}",),
        candidate_pool_digest="digest" if experiment in {"C", "D"} else None,
    )


def test_prediction_contains_all_rankings_hits_and_latency() -> None:
    results = {name: _result(name) for name in "ABCD"}

    record = build_prediction_record(
        "q1",
        "query text",
        results,
        {"d1": 2, "d2": 1},
    )

    assert record["query_id"] == "q1"
    assert record["qrels"] == {"d1": 2, "d2": 1}
    assert record["experiments"]["A"]["ranking"][0] == {
        "document_id": "d1",
        "chunk_id": "chunk-d1",
        "rank": 1,
        "score": 1.0,
        "relevance": 2,
        "hit": True,
    }
    assert record["experiments"]["D"]["candidate_pool_digest"] == "digest"


def test_finalize_writes_required_artifacts_and_completed_manifest(tmp_path: Path) -> None:
    manifest = {"run_id": "run-1", "smoke": False, "config": {"dense_k": 50}}
    initialize_run(tmp_path, manifest)
    record = build_prediction_record(
        "q1",
        "query",
        {name: _result(name, latency=float(index)) for index, name in enumerate("ABCD", 1)},
        {"d1": 1},
    )
    append_prediction(tmp_path, record)

    metrics = finalize_run(
        tmp_path,
        [record],
        evaluator_factory=FakeEvaluator,
    )

    saved_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert saved_manifest["status"] == "complete"
    assert metrics["A"]["ndcg_at_10"] == 1.0
    assert metrics["D"]["latency_ms"] == {"mean": 4.0, "p50": 4.0, "p95": 4.0}
    for filename in (
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "bad_cases.json",
        "summary.md",
    ):
        assert (tmp_path / filename).is_file()


def test_failed_run_keeps_predictions_and_never_publishes_metrics(tmp_path: Path) -> None:
    initialize_run(tmp_path, {"run_id": "failed", "smoke": True})
    append_prediction(tmp_path, {"query_id": "q1"})

    mark_run_failed(tmp_path, RuntimeError("real retrieval broke"))

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "real retrieval broke"
    assert (tmp_path / "predictions.jsonl").read_text()
    assert not (tmp_path / "metrics.json").exists()


def test_initialize_run_rejects_preexisting_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        initialize_run(tmp_path, {"run_id": "collision"})


def test_finalize_rejects_missing_experiment_instead_of_partial_metrics(
    tmp_path: Path,
) -> None:
    initialize_run(tmp_path, {"run_id": "partial", "smoke": False})
    record = build_prediction_record(
        "q1", "query", {name: _result(name) for name in "ABC"}, {"d1": 1}
    )

    with pytest.raises(ValueError, match="four experiments"):
        finalize_run(tmp_path, [record], evaluator_factory=FakeEvaluator)


def test_bad_cases_include_relevant_documents_ranked_below_five() -> None:
    ranking = [
        {
            "document_id": f"d{rank}",
            "rank": rank,
            "hit": rank == 6,
        }
        for rank in range(1, 7)
    ]
    record = {
        "query_id": "q-low",
        "experiments": {
            name: {"ranking": ranking if name == "A" else []} for name in "ABCD"
        },
    }

    bad_cases = extract_bad_cases([record])

    assert bad_cases["low_rank"] == [
        {"query_id": "q-low", "experiment": "A", "first_relevant_rank": 6}
    ]
