"""Benchmark-only orchestration of the four InsightAgent retrieval experiments."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from evals.beir_scifact.config import BenchmarkConfig
from evals.beir_scifact.ranking import DocumentRankingEntry, collapse_chunks_to_documents
from insight_agent.retrieval import RankedResult, RetrievalResult, rrf_fuse
from insight_agent.retrieval.reranker import Reranker


class _Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]: ...


@dataclass(frozen=True, slots=True)
class ExperimentQueryResult:
    query_id: str
    experiment: str
    documents: tuple[DocumentRankingEntry, ...]
    latency_ms: float
    candidate_chunk_ids: tuple[str, ...]
    candidate_pool_digest: str | None = None


class SciFactExperimentRunner:
    """Reuse project retrievers while bypassing the online final Top-8 contract."""

    def __init__(
        self,
        dense_retriever: _Retriever,
        sparse_retriever: _Retriever,
        reranker: Reranker,
        *,
        config: BenchmarkConfig,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.config = config
        self.clock = clock

    def run_query(
        self,
        query_id: str,
        query: str,
    ) -> dict[str, ExperimentQueryResult]:
        dense_start = self.clock()
        dense = self.dense_retriever.retrieve(query, top_k=self.config.dense_k)
        dense_latency = _milliseconds(dense_start, self.clock())

        sparse_start = self.clock()
        sparse = self.sparse_retriever.retrieve(query, top_k=self.config.sparse_k)
        sparse_latency = _milliseconds(sparse_start, self.clock())

        fusion_start = self.clock()
        fused = rrf_fuse([dense, sparse], rrf_k=self.config.rrf_k)
        candidates = fused[: self.config.rerank_k]
        fused_results = [
            replace(candidate.result, score=candidate.fusion_score)
            for candidate in candidates
        ]
        fusion_latency = _milliseconds(fusion_start, self.clock())
        candidate_ids = tuple(item.result.chunk_id for item in candidates)
        digest = _candidate_digest(candidate_ids)

        rerank_start = self.clock()
        reranked = (
            self.reranker.rerank(query, candidates, top_k=len(candidates))
            if candidates
            else []
        )
        if tuple(item.result.chunk_id for item in candidates) != candidate_ids:
            raise RuntimeError("reranker mutated the shared RRF candidate pool")
        reranked_ids = tuple(item.result.chunk_id for item in reranked)
        if len(reranked_ids) != len(candidate_ids) or set(reranked_ids) != set(
            candidate_ids
        ):
            raise RuntimeError(
                "reranker must return each RRF candidate exactly once"
            )
        reranked_results = [
            replace(
                candidate.result,
                score=(
                    candidate.rerank_score
                    if candidate.rerank_score is not None
                    else candidate.fusion_score
                ),
            )
            for candidate in reranked
        ]
        rerank_latency = _milliseconds(rerank_start, self.clock())

        return {
            "A": self._record(query_id, "A", dense, dense_latency),
            "B": self._record(query_id, "B", sparse, sparse_latency),
            "C": self._record(
                query_id,
                "C",
                fused_results,
                fusion_latency,
                candidate_ids=candidate_ids,
                candidate_digest=digest,
            ),
            "D": self._record(
                query_id,
                "D",
                reranked_results,
                rerank_latency,
                candidate_ids=candidate_ids,
                candidate_digest=digest,
            ),
        }

    def _record(
        self,
        query_id: str,
        experiment: str,
        chunks: list[RetrievalResult],
        latency_ms: float,
        *,
        candidate_ids: tuple[str, ...] | None = None,
        candidate_digest: str | None = None,
    ) -> ExperimentQueryResult:
        return ExperimentQueryResult(
            query_id=query_id,
            experiment=experiment,
            documents=collapse_chunks_to_documents(
                chunks,
                top_k=self.config.document_top_k,
            ),
            latency_ms=latency_ms,
            candidate_chunk_ids=(
                tuple(chunk.chunk_id for chunk in chunks)
                if candidate_ids is None
                else candidate_ids
            ),
            candidate_pool_digest=candidate_digest,
        )


def _candidate_digest(chunk_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(chunk_ids).encode("utf-8")).hexdigest()


def _milliseconds(start: float, end: float) -> float:
    elapsed = (end - start) * 1000.0
    if elapsed < 0:
        raise ValueError("benchmark clock moved backwards")
    return elapsed
