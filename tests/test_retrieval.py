"""Unit tests for the Vector RAG retrieval layer and knowledge-search tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.retrieval import (
    HybridRetrievalConfig,
    KnowledgeSearchTool,
    RetrievalPayloadError,
    RetrievalResult,
    VectorRetriever,
    build_default_hybrid_retriever,
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


class CloseableRetriever:
    def __init__(self) -> None:
        self.close_calls = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return []

    def close(self) -> None:
        self.close_calls += 1


def _hybrid_config() -> HybridRetrievalConfig:
    return HybridRetrievalConfig(
        dense_k=4,
        sparse_k=4,
        rerank_k=4,
        final_top_k=2,
        rrf_k=60,
        reranker_model="reranker",
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_default_hybrid_accepts_explicit_qdrant_and_embedding_settings(
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    from insight_agent.indexing import QdrantConfig

    calls: list[tuple[str, Any]] = []
    store = SimpleNamespace(load_chunks=lambda: [], close=lambda: None)

    def create_store(*, config=None):
        calls.append(("qdrant", config))
        return store

    def create_embedder(model_name=None):
        calls.append(("embedding", model_name))
        return object()

    monkeypatch.setattr("insight_agent.indexing.QdrantVectorStore", create_store)
    monkeypatch.setattr("insight_agent.indexing.SentenceTransformerEmbedder", create_embedder)
    monkeypatch.setattr("insight_agent.retrieval.sparse.BM25Retriever", lambda loader: object())
    monkeypatch.setattr("insight_agent.retrieval.reranker.CrossEncoderReranker", lambda model: object())
    qdrant_config = QdrantConfig(path="custom/qdrant", collection_name="custom")

    hybrid = (
        build_default_hybrid_retriever(
            _hybrid_config(), qdrant_config=qdrant_config, embedding_model="custom/embedder"
        )
        if explicit
        else build_default_hybrid_retriever(_hybrid_config())
    )

    assert calls == [
        ("qdrant", qdrant_config if explicit else None),
        ("embedding", "custom/embedder" if explicit else None),
    ]
    hybrid.close()


def test_knowledge_tool_closes_constructed_retriever() -> None:
    retriever = CloseableRetriever()
    tool = KnowledgeSearchTool(retriever)

    tool.close()

    assert retriever.close_calls == 1


def test_default_hybrid_closes_its_owned_store_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.close_calls = 0

        def load_chunks(self) -> list[Any]:
            return []

        def close(self) -> None:
            self.close_calls += 1

    store = FakeStore()
    monkeypatch.setattr("insight_agent.indexing.QdrantVectorStore", lambda *, config=None: store)
    monkeypatch.setattr(
        "insight_agent.indexing.SentenceTransformerEmbedder", lambda model_name=None: object()
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.sparse.BM25Retriever",
        lambda chunk_loader: object(),
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.reranker.CrossEncoderReranker",
        lambda model_name: object(),
    )

    hybrid = build_default_hybrid_retriever(_hybrid_config())
    hybrid.close()
    hybrid.close()

    assert store.close_calls == 1


def test_default_hybrid_closes_owned_store_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.close_calls = 0

        def load_chunks(self) -> list[Any]:
            return []

        def close(self) -> None:
            self.close_calls += 1

    store = FakeStore()
    monkeypatch.setattr("insight_agent.indexing.QdrantVectorStore", lambda *, config=None: store)
    monkeypatch.setattr(
        "insight_agent.indexing.SentenceTransformerEmbedder", lambda model_name=None: object()
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.sparse.BM25Retriever",
        lambda chunk_loader: object(),
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.reranker.CrossEncoderReranker",
        lambda model_name: (_ for _ in ()).throw(RuntimeError("reranker failed")),
    )

    with pytest.raises(RuntimeError, match="reranker failed"):
        build_default_hybrid_retriever(_hybrid_config())

    assert store.close_calls == 1


def test_default_hybrid_does_not_close_injected_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.close_calls = 0

        def load_chunks(self) -> list[Any]:
            return []

        def close(self) -> None:
            self.close_calls += 1

    store = FakeStore()
    monkeypatch.setattr(
        "insight_agent.indexing.SentenceTransformerEmbedder", lambda model_name=None: object()
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.sparse.BM25Retriever",
        lambda chunk_loader: object(),
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.reranker.CrossEncoderReranker",
        lambda model_name: object(),
    )

    hybrid = build_default_hybrid_retriever(_hybrid_config(), store)  # type: ignore[arg-type]
    hybrid.close()

    assert store.close_calls == 0


def test_default_hybrid_uses_injected_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def load_chunks(self) -> list[Any]:
            return []

    injected_embedder = object()
    monkeypatch.setattr(
        "insight_agent.indexing.SentenceTransformerEmbedder",
        lambda model_name=None: pytest.fail("must not construct another embedder"),
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.sparse.BM25Retriever",
        lambda chunk_loader: object(),
    )
    monkeypatch.setattr(
        "insight_agent.retrieval.reranker.CrossEncoderReranker",
        lambda model_name: object(),
    )

    hybrid = build_default_hybrid_retriever(
        _hybrid_config(), FakeStore(), embedder=injected_embedder
    )

    assert hybrid.dense_retriever.embedder is injected_embedder


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


def test_retriever_pushes_source_types_into_vector_store() -> None:
    class FilterAwareStore(FakeVectorStore):
        def search(
            self,
            vector: list[float],
            limit: int = 5,
            *,
            source_types: set[str] | None = None,
        ) -> list[Any]:
            self.calls.append(
                {
                    "vector": vector,
                    "limit": limit,
                    "source_types": source_types,
                }
            )
            return []

    store = FilterAwareStore()
    retriever = VectorRetriever(FakeEmbedder(), store)

    assert retriever.retrieve("diagram", top_k=2, source_types={"image"}) == []
    assert store.calls == [
        {
            "vector": [0.1, 0.2, 0.3],
            "limit": 2,
            "source_types": {"image"},
        }
    ]


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


def test_knowledge_search_tool_structured_and_formatted_calls_share_lazy_retriever() -> None:
    expected_results = [
        RetrievalResult(
            chunk_id="chunk-structured",
            score=0.75,
            content="Grounded evidence",
            document_id="document-structured",
            source="knowledge.txt",
            source_type=SourceType.TEXT,
            chunk_index=0,
            start_char=0,
            end_char=17,
            metadata={},
        )
    ]

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            self.calls.append((query, top_k))
            return expected_results

    factory_calls = 0
    retriever = FakeRetriever()

    def factory() -> FakeRetriever:
        nonlocal factory_calls
        factory_calls += 1
        return retriever

    tool = KnowledgeSearchTool(retriever_factory=factory, default_top_k=6)

    results = tool.retrieve("structured")
    observation = tool("formatted", top_k=3)

    assert results is expected_results
    assert factory_calls == 1
    assert retriever.calls == [("structured", 6), ("formatted", 3)]
    assert "Grounded evidence" in observation


@pytest.mark.parametrize("top_k", [0, -1, 9, True])
def test_knowledge_search_tool_structured_retrieval_rejects_invalid_top_k(
    top_k: int,
) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            raise AssertionError("invalid final top_k must not reach retriever")

    tool = KnowledgeSearchTool(FakeRetriever())

    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 8"):
        tool.retrieve("query", top_k=top_k)


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
        def __init__(self, *, config=None) -> None:
            pass

        def load_chunks(self):  # noqa: ANN201
            return []

        def search(self, vector, limit=5):  # noqa: ANN001, ANN201, ARG002
            return []

        def close(self) -> None:
            pass

    class FakeEmbedder:
        def __init__(self, model_name=None) -> None:
            pass

        def embed_documents(self, texts):  # noqa: ANN001, ANN201, ARG002
            return [[1.0]]

    monkeypatch.setenv("RERANKER_MODEL", "configured/reranker")
    monkeypatch.setattr(indexing, "QdrantVectorStore", FakeStore)
    monkeypatch.setattr(indexing, "SentenceTransformerEmbedder", FakeEmbedder)

    retriever = build_default_knowledge_search_tool()._get_retriever()

    assert isinstance(retriever, HybridRetriever)
    assert retriever.reranker.model_name == "configured/reranker"
    assert retriever.reranker._model is None
