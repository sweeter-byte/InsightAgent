"""Control-flow tests for dense + sparse + RRF + reranker orchestration."""

from __future__ import annotations

from dataclasses import replace

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.retrieval.hybrid import HybridRetriever
from insight_agent.retrieval.models import RankedResult, RetrievalResult, RetrievalTrace


def _result(chunk_id: str, score: float, *, branch: str) -> RetrievalResult:
    content = f"{branch} content {chunk_id}"
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=content,
        document_id=f"document-{chunk_id}",
        source=f"{branch}.md",
        source_type=SourceType.MARKDOWN,
        chunk_index=1,
        start_char=10,
        end_char=10 + len(content),
        metadata={"branch": branch},
    )


def _config(**overrides: object) -> HybridRetrievalConfig:
    values: dict[str, object] = {
        "dense_k": 4,
        "sparse_k": 3,
        "rerank_k": 3,
        "final_top_k": 2,
        "rrf_k": 60,
        "reranker_model": "configured/model",
    }
    values.update(overrides)
    return HybridRetrievalConfig(**values)  # type: ignore[arg-type]


class FakeRetriever:
    def __init__(
        self,
        name: str,
        results: list[RetrievalResult],
        events: list[str],
    ) -> None:
        self.name = name
        self.results = results
        self.events = events
        self.calls: list[tuple[str, int]] = []
        self.refresh_calls = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.events.append(self.name)
        self.calls.append((query, top_k))
        return self.results[:top_k]

    def refresh(self) -> None:
        self.refresh_calls += 1


class FakeReranker:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[str, list[RankedResult], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        self.events.append("rerank")
        self.calls.append((query, list(candidates), top_k))
        scored = [
            replace(candidate, rerank_score=float(len(candidates) - index))
            for index, candidate in enumerate(candidates)
        ]
        return scored[:top_k]


def test_hybrid_runs_both_branches_then_cuts_candidates_and_reranks() -> None:
    events: list[str] = []
    dense = FakeRetriever(
        "dense",
        [
            _result("A", 0.9, branch="dense"),
            _result("B", 0.8, branch="dense"),
            _result("C", 0.7, branch="dense"),
            _result("E", 0.6, branch="dense"),
        ],
        events,
    )
    sparse = FakeRetriever(
        "sparse",
        [
            _result("B", 11.0, branch="sparse"),
            _result("A", 9.0, branch="sparse"),
            _result("D", 7.0, branch="sparse"),
        ],
        events,
    )
    reranker = FakeReranker(events)
    traces: list[RetrievalTrace] = []
    hybrid = HybridRetriever(
        dense,
        sparse,
        reranker,
        config=_config(rerank_k=2, final_top_k=1),
        trace_callback=traces.append,
    )

    results = hybrid.retrieve("query")

    assert events == ["dense", "sparse", "rerank"]
    assert dense.calls == [("query", 4)]
    assert sparse.calls == [("query", 3)]
    assert len(reranker.calls[0][1]) == 2
    assert reranker.calls[0][2] == 2
    assert len(results) == 1
    assert results[0].chunk_id == "A"
    assert results[0].source == "dense.md"
    assert results[0].metadata == {"branch": "dense"}
    assert [entry.chunk_id for entry in traces[0].dense] == ["A", "B", "C", "E"]
    assert [entry.chunk_id for entry in traces[0].sparse] == ["B", "A", "D"]
    assert [entry.chunk_id for entry in traces[0].fused[:2]] == ["A", "B"]
    assert [entry.chunk_id for entry in traces[0].reranked] == ["A", "B"]
    assert [entry.chunk_id for entry in traces[0].final] == ["A"]


@pytest.mark.parametrize(
    ("dense_results", "sparse_results", "expected"),
    [
        ([], [_result("S", 3.0, branch="sparse")], ["S"]),
        ([_result("D", 0.8, branch="dense")], [], ["D"]),
    ],
)
def test_hybrid_continues_when_one_branch_is_empty(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    expected: list[str],
) -> None:
    events: list[str] = []
    dense = FakeRetriever("dense", dense_results, events)
    sparse = FakeRetriever("sparse", sparse_results, events)
    reranker = FakeReranker(events)
    hybrid = HybridRetriever(dense, sparse, reranker, config=_config())

    results = hybrid.retrieve("query")

    assert [result.chunk_id for result in results] == expected
    assert events == ["dense", "sparse", "rerank"]


def test_hybrid_bypasses_reranker_when_both_branches_are_empty() -> None:
    events: list[str] = []
    traces: list[RetrievalTrace] = []
    dense = FakeRetriever("dense", [], events)
    sparse = FakeRetriever("sparse", [], events)
    reranker = FakeReranker(events)
    hybrid = HybridRetriever(
        dense,
        sparse,
        reranker,
        config=_config(),
        trace_callback=traces.append,
    )

    assert hybrid.retrieve("query") == []
    assert events == ["dense", "sparse"]
    assert reranker.calls == []
    assert traces[0].fused == traces[0].reranked == traces[0].final == ()


def test_hybrid_explicit_final_top_k_overrides_configured_default() -> None:
    events: list[str] = []
    dense = FakeRetriever(
        "dense",
        [_result(str(index), 1.0 / index, branch="dense") for index in range(1, 5)],
        events,
    )
    sparse = FakeRetriever("sparse", [], events)
    reranker = FakeReranker(events)
    hybrid = HybridRetriever(dense, sparse, reranker, config=_config(final_top_k=1))

    results = hybrid.retrieve("query", top_k=3)

    assert len(results) == 3
    assert reranker.calls[0][2] == 3


def test_hybrid_refresh_delegates_to_sparse_snapshot() -> None:
    events: list[str] = []
    dense = FakeRetriever("dense", [], events)
    sparse = FakeRetriever("sparse", [], events)
    hybrid = HybridRetriever(
        dense,
        sparse,
        FakeReranker(events),
        config=_config(),
    )

    hybrid.refresh()

    assert sparse.refresh_calls == 1


def test_hybrid_close_releases_owned_resource_once() -> None:
    events: list[str] = []
    hybrid = HybridRetriever(
        FakeRetriever("dense", [], events),
        FakeRetriever("sparse", [], events),
        FakeReranker(events),
        config=_config(),
        close_callback=lambda: events.append("close"),
    )

    hybrid.close()
    hybrid.close()

    assert events == ["close"]


def test_hybrid_pushes_source_types_to_both_candidate_retrievers() -> None:
    class FilterAwareRetriever(FakeRetriever):
        def __init__(self, name: str, events: list[str]) -> None:
            super().__init__(name, [], events)
            self.filtered_calls: list[tuple[str, int, set[str] | None]] = []

        def retrieve(
            self,
            query: str,
            top_k: int = 5,
            *,
            source_types: set[str] | None = None,
        ) -> list[RetrievalResult]:
            self.events.append(self.name)
            self.filtered_calls.append((query, top_k, source_types))
            return []

    events: list[str] = []
    dense = FilterAwareRetriever("dense", events)
    sparse = FilterAwareRetriever("sparse", events)
    hybrid = HybridRetriever(dense, sparse, FakeReranker(events), config=_config())

    assert hybrid.retrieve("diagram", source_types={"image"}) == []
    assert dense.filtered_calls == [("diagram", 4, {"image"})]
    assert sparse.filtered_calls == [("diagram", 3, {"image"})]


def test_hybrid_omitted_source_types_preserves_legacy_retriever_call_shape() -> None:
    events: list[str] = []
    dense = FakeRetriever("dense", [], events)
    sparse = FakeRetriever("sparse", [], events)
    hybrid = HybridRetriever(dense, sparse, FakeReranker(events), config=_config())

    assert hybrid.retrieve("query") == []
    assert dense.calls == [("query", 4)]
    assert sparse.calls == [("query", 3)]


@pytest.mark.parametrize("top_k", [0, -1, 9, True])
def test_hybrid_rejects_final_top_k_outside_agent_contract(top_k: int) -> None:
    events: list[str] = []
    hybrid = HybridRetriever(
        FakeRetriever("dense", [], events),
        FakeRetriever("sparse", [], events),
        FakeReranker(events),
        config=_config(),
    )

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 8"):
        hybrid.retrieve("query", top_k=top_k)
