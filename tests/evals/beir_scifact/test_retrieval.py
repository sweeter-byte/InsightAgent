from __future__ import annotations

import pytest

from evals.beir_scifact.config import BenchmarkConfig
from evals.beir_scifact.retrieval import SciFactExperimentRunner
from insight_agent.ingestion import SourceType
from insight_agent.retrieval import RankedResult, RetrievalResult


def _result(chunk_id: str, doc_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=f"content {chunk_id}",
        document_id=f"internal-{doc_id}",
        source=f"{doc_id}.txt",
        source_type=SourceType.TEXT,
        chunk_index=0,
        start_char=0,
        end_char=10,
        metadata={"beir_doc_id": doc_id},
    )


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return list(self.results[:top_k])


class RecordingReranker:
    def __init__(self, *, fail: bool = False) -> None:
        self.candidate_ids: list[str] = []
        self.fail = fail

    def rerank(self, query: str, candidates, top_k: int):
        self.candidate_ids = [candidate.result.chunk_id for candidate in candidates]
        if self.fail:
            raise RuntimeError("real reranker failed")
        return [
            RankedResult(
                result=candidate.result,
                fusion_score=candidate.fusion_score,
                rerank_score=float(len(candidates) - index),
            )
            for index, candidate in enumerate(reversed(candidates))
        ][:top_k]


def test_four_experiments_share_query_and_fixed_candidate_depths() -> None:
    dense = FakeRetriever([_result("d1", "doc-1", 1), _result("d2", "doc-2", 0.5)])
    sparse = FakeRetriever([_result("s1", "doc-2", 3), _result("s2", "doc-3", 2)])
    reranker = RecordingReranker()
    config = BenchmarkConfig(dense_k=7, sparse_k=9, rerank_k=4)

    results = SciFactExperimentRunner(
        dense,
        sparse,
        reranker,
        config=config,
    ).run_query("q-1", "same query")

    assert list(results) == ["A", "B", "C", "D"]
    assert dense.calls == [("same query", 7)]
    assert sparse.calls == [("same query", 9)]
    assert results["A"].query_id == results["D"].query_id == "q-1"


def test_reranker_receives_exact_rrf_candidate_pool() -> None:
    dense = FakeRetriever([_result("a", "doc-a", 1), _result("b", "doc-b", 0.5)])
    sparse = FakeRetriever([_result("b", "doc-b", 2), _result("c", "doc-c", 1)])
    reranker = RecordingReranker()

    results = SciFactExperimentRunner(
        dense,
        sparse,
        reranker,
        config=BenchmarkConfig(rerank_k=3),
    ).run_query("q", "query")

    assert reranker.candidate_ids == list(results["C"].candidate_chunk_ids)
    assert results["C"].candidate_pool_digest == results["D"].candidate_pool_digest


def test_document_top_k_is_applied_after_chunk_deduplication() -> None:
    dense = FakeRetriever(
        [
            _result("a-1", "doc-a", 3),
            _result("a-2", "doc-a", 2),
            _result("b-1", "doc-b", 1),
        ]
    )
    sparse = FakeRetriever([])

    results = SciFactExperimentRunner(
        dense,
        sparse,
        RecordingReranker(),
        config=BenchmarkConfig(document_top_k=2),
    ).run_query("q", "query")

    assert [entry.document_id for entry in results["A"].documents] == [
        "doc-a",
        "doc-b",
    ]


def test_real_reranker_failure_propagates_without_mock_fallback() -> None:
    runner = SciFactExperimentRunner(
        FakeRetriever([_result("a", "doc-a", 1)]),
        FakeRetriever([]),
        RecordingReranker(fail=True),
        config=BenchmarkConfig(),
    )

    with pytest.raises(RuntimeError, match="real reranker failed"):
        runner.run_query("q", "query")


def test_reranker_cannot_replace_or_drop_shared_rrf_candidates() -> None:
    class DroppingReranker:
        def rerank(self, query, candidates, top_k):
            return candidates[:-1]

    runner = SciFactExperimentRunner(
        FakeRetriever([_result("a", "doc-a", 1), _result("b", "doc-b", 0.5)]),
        FakeRetriever([]),
        DroppingReranker(),
        config=BenchmarkConfig(),
    )

    with pytest.raises(RuntimeError, match="each RRF candidate exactly once"):
        runner.run_query("q", "query")


def test_every_experiment_records_nonnegative_latency() -> None:
    ticks = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3, 4.0, 4.4])
    runner = SciFactExperimentRunner(
        FakeRetriever([_result("a", "doc-a", 1)]),
        FakeRetriever([]),
        RecordingReranker(),
        config=BenchmarkConfig(),
        clock=lambda: next(ticks),
    )

    results = runner.run_query("q", "query")

    assert [round(result.latency_ms) for result in results.values()] == [100, 200, 300, 400]
