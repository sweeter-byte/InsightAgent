from __future__ import annotations

import pytest

from evals.beir_scifact.ranking import (
    collapse_chunks_to_documents,
    rankings_to_beir_results,
)
from insight_agent.ingestion import SourceType
from insight_agent.retrieval import RetrievalResult


def _result(chunk_id: str, doc_id: str | None, score: float) -> RetrievalResult:
    metadata = {} if doc_id is None else {"beir_doc_id": doc_id}
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=f"content {chunk_id}",
        document_id=f"internal-{chunk_id}",
        source=f"{chunk_id}.txt",
        source_type=SourceType.TEXT,
        chunk_index=0,
        start_char=0,
        end_char=10,
        metadata=metadata,
    )


def test_collapse_uses_first_chunk_rank_and_removes_duplicate_documents() -> None:
    chunks = [
        _result("a-1", "doc-a", 0.9),
        _result("a-2", "doc-a", 0.8),
        _result("b-1", "doc-b", 0.7),
    ]

    documents = collapse_chunks_to_documents(chunks, top_k=10)

    assert [item.document_id for item in documents] == ["doc-a", "doc-b"]
    assert [item.chunk_id for item in documents] == ["a-1", "b-1"]
    assert [item.rank for item in documents] == [1, 2]


def test_collapse_applies_top_k_after_document_deduplication() -> None:
    chunks = [
        _result("a-1", "doc-a", 3),
        _result("a-2", "doc-a", 2),
        _result("b-1", "doc-b", 1),
    ]

    documents = collapse_chunks_to_documents(chunks, top_k=2)

    assert [item.document_id for item in documents] == ["doc-a", "doc-b"]


def test_collapse_requires_original_beir_document_id() -> None:
    with pytest.raises(ValueError, match="beir_doc_id"):
        collapse_chunks_to_documents([_result("missing", None, 1)], top_k=10)


def test_beir_scores_are_unique_and_preserve_exact_rank_order() -> None:
    rankings = {
        "query-1": collapse_chunks_to_documents(
            [
                _result("a", "doc-a", 0.5),
                _result("b", "doc-b", 0.5),
                _result("c", "doc-c", 0.5),
            ],
            top_k=10,
        )
    }

    converted = rankings_to_beir_results(rankings)

    assert list(converted["query-1"]) == ["doc-a", "doc-b", "doc-c"]
    assert list(converted["query-1"].values()) == [3.0, 2.0, 1.0]
