"""Unit tests for the Vector RAG retrieval layer and knowledge-search tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.retrieval import (
    KnowledgeSearchTool,
    RetrievalPayloadError,
    RetrievalResult,
    VectorRetriever,
    format_results,
)
from insight_agent.retrieval.hybrid import HybridRetriever
from insight_agent.retrieval.tool import (
    SEARCH_KNOWLEDGE_BASE_SCHEMA,
    build_default_knowledge_search_tool,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": "Overlap keeps context across adjacent chunks.",
        "document_id": "document-7",
        "source": "notes/rag.md",
        "source_type": "markdown",
        "chunk_index": 2,
        "start_char": 120,
        "end_char": 170,
        "metadata": {"title": "RAG notes", "section": "chunking"},
    }
    payload.update(overrides)
    return payload


def _point(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "id": "chunk-9",
        "score": 0.84213,
        "payload": _payload(),
        "vector": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or [[0.1, 0.2, 0.3]]
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.vectors


class FakeVectorStore:
    def __init__(self, points: list[Any] | None = None) -> None:
        self.points = [_point()] if points is None else points
        self.calls: list[dict[str, Any]] = []

    def search(self, vector: list[float], limit: int = 5) -> list[Any]:
        self.calls.append({"vector": vector, "limit": limit})
        return self.points


def test_retriever_embeds_one_query_and_restores_complete_result() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    retriever = VectorRetriever(embedder=embedder, vector_store=store)

    results = retriever.retrieve("为什么分块需要 overlap？", top_k=3)

    assert embedder.calls == [["为什么分块需要 overlap？"]]
    assert store.calls == [{"vector": [0.1, 0.2, 0.3], "limit": 3}]
    assert results == [
        RetrievalResult(
            chunk_id="chunk-9",
            score=0.84213,
            content="Overlap keeps context across adjacent chunks.",
            document_id="document-7",
            source="notes/rag.md",
            source_type=SourceType.MARKDOWN,
            chunk_index=2,
            start_char=120,
            end_char=170,
            metadata={"title": "RAG notes", "section": "chunking"},
        )
    ]


def test_retriever_allows_internal_recall_above_tool_limit() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore(points=[])
    retriever = VectorRetriever(embedder=embedder, vector_store=store)

    assert retriever.retrieve("broad recall", top_k=20) == []
    assert store.calls == [{"vector": [0.1, 0.2, 0.3], "limit": 20}]


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_retriever_rejects_empty_query(query: str) -> None:
    retriever = VectorRetriever(FakeEmbedder(), FakeVectorStore())

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_retriever_rejects_non_positive_top_k(top_k: int) -> None:
    retriever = VectorRetriever(FakeEmbedder(), FakeVectorStore())

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=top_k)


@pytest.mark.parametrize(
    "missing_field",
    [
        "content",
        "document_id",
        "source",
        "source_type",
        "chunk_index",
        "start_char",
        "end_char",
        "metadata",
    ],
)
def test_retriever_rejects_missing_critical_payload(missing_field: str) -> None:
    payload = _payload()
    del payload[missing_field]
    retriever = VectorRetriever(
        FakeEmbedder(), FakeVectorStore([_point(payload=payload)])
    )

    with pytest.raises(RetrievalPayloadError, match=missing_field):
        retriever.retrieve("query")


def test_retriever_does_not_turn_none_payload_values_into_strings() -> None:
    retriever = VectorRetriever(
        FakeEmbedder(),
        FakeVectorStore([_point(payload=_payload(source=None))]),
    )

    with pytest.raises(RetrievalPayloadError, match="source"):
        retriever.retrieve("query")


def test_retriever_preserves_embedding_exception() -> None:
    original = RuntimeError("embedding failed")

    class BrokenEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise original

    retriever = VectorRetriever(BrokenEmbedder(), FakeVectorStore())

    with pytest.raises(RuntimeError) as caught:
        retriever.retrieve("query")
    assert caught.value is original


def test_retriever_preserves_vector_store_exception() -> None:
    original = RuntimeError("qdrant unavailable")

    class BrokenStore:
        def search(self, vector: list[float], limit: int = 5) -> list[Any]:
            raise original

    retriever = VectorRetriever(FakeEmbedder(), BrokenStore())

    with pytest.raises(RuntimeError) as caught:
        retriever.retrieve("query")
    assert caught.value is original


def test_format_results_keeps_result_boundaries_and_provenance() -> None:
    results = [
        RetrievalResult(
            chunk_id="chunk-9",
            score=0.84213,
            content="First body",
            document_id="document-7",
            source="notes/rag.md",
            source_type=SourceType.MARKDOWN,
            chunk_index=2,
            start_char=120,
            end_char=170,
            metadata={"section": "chunking"},
        ),
        RetrievalResult(
            chunk_id="chunk-10",
            score=0.7,
            content="Second body",
            document_id="document-8",
            source="paper.pdf",
            source_type=SourceType.PDF,
            chunk_index=0,
            start_char=0,
            end_char=11,
            metadata={"page": 3},
        ),
    ]

    formatted = format_results(results)

    assert "[Result 1]" in formatted
    assert "[Result 2]" in formatted
    assert "score: 0.8421" in formatted
    assert "source: notes/rag.md" in formatted
    assert 'metadata: {"section": "chunking"}' in formatted
    assert "content:\nFirst body" in formatted
    assert "source: paper.pdf" in formatted
    assert "content:\nSecond body" in formatted


def test_format_results_makes_empty_retrieval_explicit() -> None:
    assert format_results([]) == "知识库中没有找到相关内容。"


def test_knowledge_search_tool_delegates_and_formats_observation() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            self.calls.append((query, top_k))
            return [
                RetrievalResult(
                    chunk_id="chunk-9",
                    score=0.9,
                    content="Grounded evidence",
                    document_id="document-7",
                    source="knowledge.txt",
                    source_type=SourceType.TEXT,
                    chunk_index=0,
                    start_char=0,
                    end_char=17,
                    metadata={"kind": "fixture"},
                )
            ]

    retriever = FakeRetriever()
    tool = KnowledgeSearchTool(retriever)

    observation = tool(query="helios", top_k=2)

    assert retriever.calls == [("helios", 2)]
    assert "[Result 1]" in observation
    assert "source: knowledge.txt" in observation
    assert "Grounded evidence" in observation


def test_knowledge_search_tool_uses_injected_default_top_k() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            self.calls.append((query, top_k))
            return []

    retriever = FakeRetriever()
    tool = KnowledgeSearchTool(retriever, default_top_k=7)

    tool(query="configured default")

    assert retriever.calls == [("configured default", 7)]


@pytest.mark.parametrize("top_k", [0, -1, 9, True])
def test_knowledge_search_tool_rejects_final_top_k_outside_contract(
    top_k: int,
) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            raise AssertionError("invalid final top_k must not reach retriever")

    tool = KnowledgeSearchTool(FakeRetriever())

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 8"):
        tool(query="query", top_k=top_k)


def test_knowledge_search_schema_describes_query_and_bounded_top_k() -> None:
    function = SEARCH_KNOWLEDGE_BASE_SCHEMA["function"]
    parameters = function["parameters"]

    assert function["name"] == "search_knowledge_base"
    assert parameters["required"] == ["query"]
    assert parameters["properties"]["query"]["type"] == "string"
    assert parameters["properties"]["top_k"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 8,
        "default": 5,
        "description": "Number of the most relevant chunks to return.",
    }


def test_default_knowledge_tool_reads_configured_final_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "configured/reranker")
    monkeypatch.setenv("HYBRID_FINAL_TOP_K", "7")

    tool = build_default_knowledge_search_tool()

    assert tool._default_top_k == 7


def test_default_knowledge_tool_factory_builds_hybrid_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import insight_agent.indexing as indexing

    class FakeStore:
        def load_chunks(self):  # noqa: ANN201
            return []

        def search(self, vector, limit=5):  # noqa: ANN001, ANN201, ARG002
            return []

    class FakeEmbedder:
        def embed_documents(self, texts):  # noqa: ANN001, ANN201, ARG002
            return [[1.0]]

    monkeypatch.setenv("RERANKER_MODEL", "configured/reranker")
    monkeypatch.setattr(indexing, "QdrantVectorStore", FakeStore)
    monkeypatch.setattr(indexing, "SentenceTransformerEmbedder", FakeEmbedder)

    retriever = build_default_knowledge_search_tool()._get_retriever()

    assert isinstance(retriever, HybridRetriever)
    assert retriever.reranker.model_name == "configured/reranker"
    assert retriever.reranker._model is None
