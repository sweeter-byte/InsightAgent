"""Rank-only Reciprocal Rank Fusion for retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence

from insight_agent.retrieval.models import RankedResult, RetrievalResult


def rrf_fuse(
    rankings: Sequence[Sequence[RetrievalResult]],
    *,
    rrf_k: int = 60,
) -> list[RankedResult]:
    """Fuse rankings by rank and deduplicate candidates by stable chunk ID."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    results: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    next_order = 0
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk_id
            if chunk_id not in results:
                results[chunk_id] = result
                scores[chunk_id] = 0.0
                first_seen[chunk_id] = next_order
                next_order += 1
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ordered_ids = sorted(
        results,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
    )
    return [
        RankedResult(result=results[chunk_id], fusion_score=scores[chunk_id])
        for chunk_id in ordered_ids
    ]
