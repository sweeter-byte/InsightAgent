"""Unit tests for the lazy cross-encoder reranker adapter."""

from __future__ import annotations

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.retrieval.models import RankedResult, RetrievalResult
from insight_agent.retrieval.reranker import CrossEncoderReranker


def _candidate(chunk_id: str, content: str) -> RankedResult:
    return RankedResult(
        result=RetrievalResult(
            chunk_id=chunk_id,
            score=0.5,
            content=content,
            document_id=f"document-{chunk_id}",
            source=f"{chunk_id}.md",
            source_type=SourceType.MARKDOWN,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            metadata={"id": chunk_id},
        ),
        fusion_score=0.02,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def predict(self, pairs, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append({"pairs": list(pairs), **kwargs})
        return self.scores


def test_cross_encoder_reranks_query_content_pairs_and_limits_output() -> None:
    model = FakeCrossEncoder([0.2, 0.95, -0.1])
    factory_calls: list[str] = []

    def factory(model_name: str) -> FakeCrossEncoder:
        factory_calls.append(model_name)
        return model

    candidates = [
        _candidate("A", "alpha"),
        _candidate("B", "beta"),
        _candidate("C", "gamma"),
    ]
    reranker = CrossEncoderReranker(
        "configured/reranker",
        batch_size=7,
        model_factory=factory,
    )

    reranked = reranker.rerank("which is relevant?", candidates, top_k=2)

    assert factory_calls == ["configured/reranker"]
    assert model.calls == [
        {
            "pairs": [
                ("which is relevant?", "alpha"),
                ("which is relevant?", "beta"),
                ("which is relevant?", "gamma"),
            ],
            "batch_size": 7,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        }
    ]
    assert [item.result.chunk_id for item in reranked] == ["B", "A"]
    assert [item.rerank_score for item in reranked] == [0.95, 0.2]
    assert reranked[0].result.metadata == {"id": "B"}


def test_cross_encoder_model_is_lazy_and_reused() -> None:
    model = FakeCrossEncoder([0.8])
    factory_calls: list[str] = []

    def factory(model_name: str) -> FakeCrossEncoder:
        factory_calls.append(model_name)
        return model

    reranker = CrossEncoderReranker("model", model_factory=factory)

    assert factory_calls == []
    reranker.rerank("q1", [_candidate("A", "a")], top_k=1)
    reranker.rerank("q2", [_candidate("B", "b")], top_k=1)

    assert factory_calls == ["model"]


def test_cross_encoder_empty_candidates_do_not_load_model() -> None:
    factory_calls: list[str] = []
    reranker = CrossEncoderReranker(
        "model",
        model_factory=lambda name: factory_calls.append(name),
    )

    assert reranker.rerank("query", [], top_k=3) == []
    assert factory_calls == []


def test_cross_encoder_rejects_non_scalar_or_missing_scores() -> None:
    model = FakeCrossEncoder([0.5])
    reranker = CrossEncoderReranker("model", model_factory=lambda _name: model)

    with pytest.raises(RuntimeError, match="one scalar score per candidate"):
        reranker.rerank(
            "query",
            [_candidate("A", "a"), _candidate("B", "b")],
            top_k=2,
        )


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_cross_encoder_rejects_non_positive_top_k(top_k: int) -> None:
    reranker = CrossEncoderReranker("model", model_factory=lambda _name: object())

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank("query", [], top_k=top_k)
