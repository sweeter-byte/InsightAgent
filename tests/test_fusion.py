"""Rank-only Reciprocal Rank Fusion tests."""

from __future__ import annotations

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.retrieval.fusion import rrf_fuse
from insight_agent.retrieval.models import RetrievalResult


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=f"content-{chunk_id}",
        document_id=f"document-{chunk_id}",
        source=f"{chunk_id}.md",
        source_type=SourceType.MARKDOWN,
        chunk_index=0,
        start_char=0,
        end_char=9,
        metadata={"chunk": chunk_id},
    )


def test_rrf_deduplicates_and_accumulates_rank_contributions() -> None:
    dense = [_result("A", 0.99), _result("B", 0.8), _result("C", 0.7)]
    sparse = [_result("B", 12.0), _result("A", 8.0), _result("D", 4.0)]

    fused = rrf_fuse([dense, sparse], rrf_k=60)

    assert [item.result.chunk_id for item in fused] == ["A", "B", "C", "D"]
    assert len({item.result.chunk_id for item in fused}) == 4
    assert fused[0].fusion_score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[2].fusion_score == pytest.approx(1 / 63)
    assert fused[3].fusion_score == pytest.approx(1 / 63)


def test_rrf_ignores_raw_dense_and_sparse_scores() -> None:
    baseline = rrf_fuse(
        [
            [_result("A", 0.9), _result("B", 0.1)],
            [_result("B", 100.0), _result("A", -50.0)],
        ],
        rrf_k=10,
    )
    changed = rrf_fuse(
        [
            [_result("A", -9999.0), _result("B", 9999.0)],
            [_result("B", -0.001), _result("A", 1_000_000.0)],
        ],
        rrf_k=10,
    )

    assert [item.result.chunk_id for item in changed] == [
        item.result.chunk_id for item in baseline
    ]
    assert [item.fusion_score for item in changed] == pytest.approx(
        [item.fusion_score for item in baseline]
    )


def test_rrf_uses_first_seen_result_as_canonical_candidate() -> None:
    dense_a = _result("A", 0.8)
    sparse_a = _result("A", 7.0)
    sparse_a.source = "different-copy.md"

    fused = rrf_fuse([[dense_a], [sparse_a]])

    assert fused[0].result is dense_a


@pytest.mark.parametrize("rrf_k", [0, -1, True])
def test_rrf_rejects_non_positive_constant(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
        rrf_fuse([], rrf_k=rrf_k)
