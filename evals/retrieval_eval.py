"""Direct, deterministic Evaluation over the production Retriever interface."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from evals.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from evals.models import EvalCase, MetricStatus, RetrievalEvalResult
from insight_agent.retrieval import RetrievalResult, RetrievalTrace


class EvaluationRetriever(Protocol):
    """Production-compatible retrieval capability consumed by Evaluation."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


class RetrievalTraceRecorder:
    """Callable sink for the production HybridRetriever trace callback."""

    def __init__(self) -> None:
        self._latest: RetrievalTrace | None = None

    @property
    def latest(self) -> RetrievalTrace | None:
        return self._latest

    def __call__(self, trace: RetrievalTrace) -> None:
        if not isinstance(trace, RetrievalTrace):
            raise TypeError("trace must be a RetrievalTrace")
        self._latest = trace

    def clear(self) -> None:
        self._latest = None


class RetrievalEvaluator:
    """Call one Retriever directly and score its ordered output."""

    def __init__(
        self,
        retriever: EvaluationRetriever,
        *,
        trace_recorder: RetrievalTraceRecorder | None = None,
    ) -> None:
        self.retriever = retriever
        self.trace_recorder = trace_recorder

    def evaluate(self, case: EvalCase, *, top_k: int = 5) -> RetrievalEvalResult:
        if not isinstance(case, EvalCase):
            raise TypeError("case must be an EvalCase")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 8
        ):
            raise ValueError("top_k must be an integer between 1 and 8")
        if self.trace_recorder is not None:
            self.trace_recorder.clear()

        results = self.retriever.retrieve(case.query, top_k=top_k)
        if not isinstance(results, list):
            raise TypeError("retriever must return a list")
        retrieved_ids = tuple(
            _retrieval_id(result, index=index)
            for index, result in enumerate(results, start=1)
        )

        has_gold = bool(case.gold_retrieval_ids)
        if has_gold:
            recall = recall_at_k(retrieved_ids, case.gold_retrieval_ids, top_k)
            rr = reciprocal_rank(retrieved_ids, case.gold_retrieval_ids)
            status = MetricStatus.COMPUTED
            metric_reason = ""
        else:
            recall = None
            rr = None
            status = MetricStatus.NOT_COMPUTABLE
            metric_reason = "case has no retrieval Gold Label"

        pre_ids: tuple[str, ...] = ()
        post_ids: tuple[str, ...] = ()
        pre_rr: float | None = None
        post_rr: float | None = None
        trace = self.trace_recorder.latest if self.trace_recorder is not None else None
        if trace is not None and trace.query == case.query:
            pre_ids = tuple(entry.chunk_id for entry in trace.fused)
            post_ids = tuple(entry.chunk_id for entry in trace.reranked)
            if has_gold:
                pre_rr = reciprocal_rank(pre_ids, case.gold_retrieval_ids)
                post_rr = reciprocal_rank(post_ids, case.gold_retrieval_ids)

        return RetrievalEvalResult(
            case_id=case.case_id,
            retrieved_ids=retrieved_ids,
            recall_at_k=recall,
            reciprocal_rank=rr,
            metric_status=status,
            metric_reason=metric_reason,
            pre_rerank_ids=pre_ids,
            post_rerank_ids=post_ids,
            pre_rerank_reciprocal_rank=pre_rr,
            post_rerank_reciprocal_rank=post_rr,
        )


def retrieval_mrr(results: Iterable[RetrievalEvalResult]) -> float | None:
    """Average labelled Case reciprocal ranks, excluding unavailable Gold."""
    values: list[float] = []
    for result in results:
        if not isinstance(result, RetrievalEvalResult):
            raise TypeError("results must contain RetrievalEvalResult values")
        if result.reciprocal_rank is not None:
            values.append(result.reciprocal_rank)
    return mean_reciprocal_rank(values) if values else None


def _retrieval_id(result: object, *, index: int) -> str:
    chunk_id = getattr(result, "chunk_id", None)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise TypeError(
            f"retriever result {index} chunk_id must be a non-empty string"
        )
    return chunk_id
