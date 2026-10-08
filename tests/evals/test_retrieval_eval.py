from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals import EvalCase, MetricStatus
from evals.retrieval_eval import (
    RetrievalEvaluator,
    RetrievalTraceRecorder,
    retrieval_mrr,
)
from insight_agent.ingestion import SourceType
from insight_agent.retrieval import RetrievalResult
from insight_agent.retrieval.models import RankingTraceEntry, RetrievalTrace


def _result(chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.5,
        content=f"content {chunk_id}",
        document_id=f"doc-{chunk_id}",
        source="fixture.md",
        source_type=SourceType.MARKDOWN,
        chunk_index=0,
        start_char=0,
        end_char=10,
    )


class FakeRetriever:
    def __init__(
        self,
        results: list[object],
        *,
        recorder: RetrievalTraceRecorder | None = None,
        trace: RetrievalTrace | None = None,
    ) -> None:
        self.results = results
        self.recorder = recorder
        self.trace = trace
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[object]:
        self.calls.append((query, top_k))
        if self.recorder is not None and self.trace is not None:
            self.recorder(self.trace)
        return self.results[:top_k]


def test_retrieval_evaluator_calls_retriever_and_computes_case_metrics() -> None:
    retriever = FakeRetriever([_result("X"), _result("gold"), _result("Y")])
    case = EvalCase(
        case_id="case-1",
        query="fixed query",
        gold_retrieval_ids=("gold", "missing"),
    )

    result = RetrievalEvaluator(retriever).evaluate(case, top_k=3)

    assert retriever.calls == [("fixed query", 3)]
    assert result.retrieved_ids == ("X", "gold", "Y")
    assert result.recall_at_k == 0.5
    assert result.reciprocal_rank == 0.5
    assert result.metric_status is MetricStatus.COMPUTED
    assert result.metric_reason == ""


def test_retrieval_evaluator_marks_missing_gold_as_not_computable() -> None:
    result = RetrievalEvaluator(FakeRetriever([_result("X")])).evaluate(
        EvalCase(case_id="case-1", query="query"),
        top_k=1,
    )

    assert result.recall_at_k is None
    assert result.reciprocal_rank is None
    assert result.metric_status is MetricStatus.NOT_COMPUTABLE
    assert "Gold Label" in result.metric_reason


def test_retrieval_evaluator_consumes_existing_hybrid_trace_callback() -> None:
    recorder = RetrievalTraceRecorder()
    recorder(
        RetrievalTrace(
            query="stale query",
            dense=(),
            sparse=(),
            fused=(RankingTraceEntry("stale", 1.0),),
            reranked=(RankingTraceEntry("stale", 1.0),),
            final=(RankingTraceEntry("stale", 1.0),),
        )
    )
    trace = RetrievalTrace(
        query="query",
        dense=(),
        sparse=(),
        fused=(
            RankingTraceEntry("X", 0.5),
            RankingTraceEntry("gold", 0.4),
        ),
        reranked=(
            RankingTraceEntry("gold", 0.9),
            RankingTraceEntry("X", 0.2),
        ),
        final=(RankingTraceEntry("gold", 0.9),),
    )
    retriever = FakeRetriever([_result("gold")], recorder=recorder, trace=trace)

    result = RetrievalEvaluator(retriever, trace_recorder=recorder).evaluate(
        EvalCase(
            case_id="case-1",
            query="query",
            gold_retrieval_ids=("gold",),
        ),
        top_k=1,
    )

    assert result.pre_rerank_ids == ("X", "gold")
    assert result.post_rerank_ids == ("gold", "X")
    assert result.pre_rerank_reciprocal_rank == 0.5
    assert result.post_rerank_reciprocal_rank == 1.0


def test_retrieval_evaluator_without_trace_does_not_fabricate_one() -> None:
    result = RetrievalEvaluator(FakeRetriever([_result("gold")])).evaluate(
        EvalCase(
            case_id="case-1",
            query="query",
            gold_retrieval_ids=("gold",),
        ),
        top_k=1,
    )

    assert result.pre_rerank_ids == ()
    assert result.post_rerank_ids == ()
    assert result.pre_rerank_reciprocal_rank is None
    assert result.post_rerank_reciprocal_rank is None


def test_retrieval_evaluator_ignores_trace_for_another_query() -> None:
    recorder = RetrievalTraceRecorder()
    trace = RetrievalTrace(
        query="other query",
        dense=(),
        sparse=(),
        fused=(RankingTraceEntry("gold", 1.0),),
        reranked=(RankingTraceEntry("gold", 1.0),),
        final=(RankingTraceEntry("gold", 1.0),),
    )

    result = RetrievalEvaluator(
        FakeRetriever([_result("gold")], recorder=recorder, trace=trace),
        trace_recorder=recorder,
    ).evaluate(
        EvalCase(
            case_id="case-1",
            query="query",
            gold_retrieval_ids=("gold",),
        ),
        top_k=1,
    )

    assert result.pre_rerank_ids == ()
    assert result.post_rerank_ids == ()


@pytest.mark.parametrize(
    "bad_result",
    [SimpleNamespace(), SimpleNamespace(chunk_id=" "), SimpleNamespace(chunk_id=1)],
)
def test_retrieval_evaluator_rejects_malformed_retriever_results(
    bad_result: object,
) -> None:
    with pytest.raises(TypeError, match="chunk_id"):
        RetrievalEvaluator(FakeRetriever([bad_result])).evaluate(
            EvalCase(case_id="case-1", query="query"),
            top_k=1,
        )


@pytest.mark.parametrize("top_k", [0, -1, 9, True, 1.5])
def test_retrieval_evaluator_enforces_production_top_k_contract(
    top_k: object,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        RetrievalEvaluator(FakeRetriever([])).evaluate(
            EvalCase(case_id="case-1", query="query"),
            top_k=top_k,  # type: ignore[arg-type]
        )


def test_retrieval_mrr_excludes_unlabelled_cases() -> None:
    hit = RetrievalEvaluator(FakeRetriever([_result("gold")])).evaluate(
        EvalCase("hit", "query", gold_retrieval_ids=("gold",)), top_k=1
    )
    miss = RetrievalEvaluator(FakeRetriever([])).evaluate(
        EvalCase("miss", "query", gold_retrieval_ids=("missing",)), top_k=1
    )
    unlabelled = RetrievalEvaluator(FakeRetriever([])).evaluate(
        EvalCase("none", "query"), top_k=1
    )

    assert retrieval_mrr([hit, miss, unlabelled]) == 0.5
    assert retrieval_mrr([unlabelled]) is None
