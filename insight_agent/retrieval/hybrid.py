"""Coordination of dense recall, BM25, RRF, and cross-encoder reranking."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.retrieval.fusion import rrf_fuse
from insight_agent.retrieval.models import (
    RankedResult,
    RankingTraceEntry,
    RetrievalResult,
    RetrievalTrace,
)
from insight_agent.retrieval.reranker import Reranker


logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


class RefreshableRetriever(Retriever, Protocol):
    def refresh(self) -> None:
        ...


class HybridRetriever:
    """Coordinate retrieval stages while keeping them independently testable."""

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: RefreshableRetriever,
        reranker: Reranker,
        *,
        config: HybridRetrievalConfig,
        trace_callback: Callable[[RetrievalTrace], None] | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.config = config
        self.trace_callback = trace_callback

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Run the hybrid pipeline and return Agent-compatible results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        final_top_k = self.config.final_top_k if top_k is None else top_k
        if (
            isinstance(final_top_k, bool)
            or not isinstance(final_top_k, int)
            or not 1 <= final_top_k <= 8
        ):
            raise ValueError("top_k must be an integer between 1 and 8")

        dense = self.dense_retriever.retrieve(query, top_k=self.config.dense_k)
        sparse = self.sparse_retriever.retrieve(query, top_k=self.config.sparse_k)
        fused = rrf_fuse([dense, sparse], rrf_k=self.config.rrf_k)
        candidates = fused[: self.config.rerank_k]
        reranked = (
            self.reranker.rerank(query, candidates, top_k=len(candidates))
            if candidates
            else []
        )
        final_ranked = reranked[:final_top_k]
        final = [candidate.result for candidate in final_ranked]

        trace = RetrievalTrace(
            query=query,
            dense=tuple(
                RankingTraceEntry(result.chunk_id, result.score) for result in dense
            ),
            sparse=tuple(
                RankingTraceEntry(result.chunk_id, result.score) for result in sparse
            ),
            fused=_ranked_trace(fused, score_name="fusion"),
            reranked=_ranked_trace(reranked, score_name="rerank"),
            final=_ranked_trace(final_ranked, score_name="rerank"),
        )
        logger.debug("Hybrid retrieval trace: %s", trace)
        if self.trace_callback is not None:
            self.trace_callback(trace)
        return final

    def refresh(self) -> None:
        """Explicitly rebuild and atomically replace the sparse snapshot."""
        self.sparse_retriever.refresh()


def _ranked_trace(
    results: list[RankedResult],
    *,
    score_name: str,
) -> tuple[RankingTraceEntry, ...]:
    return tuple(
        RankingTraceEntry(
            item.result.chunk_id,
            item.fusion_score if score_name == "fusion" else item.rerank_score,
        )
        for item in results
    )
