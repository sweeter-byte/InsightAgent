"""Agent tool adapter for local knowledge-base retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.retrieval.formatting import format_results
from insight_agent.retrieval.models import RetrievalResult


SEARCH_KNOWLEDGE_BASE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the indexed local knowledge base and return relevant text "
            "chunks together with their sources and metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 5,
                    "description": "Number of the most relevant chunks to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class Retriever(Protocol):
    """Retrieval behavior required by the Agent tool adapter."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


class KnowledgeSearchTool:
    """Convert a local-knowledge query into a formatted tool observation."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        retriever_factory: Callable[[], Retriever] | None = None,
        default_top_k: int = 5,
    ) -> None:
        if (retriever is None) == (retriever_factory is None):
            raise ValueError("provide exactly one of retriever or retriever_factory")
        _validate_final_top_k(default_top_k)
        self._retriever = retriever
        self._retriever_factory = retriever_factory
        self._default_top_k = default_top_k

    def __call__(self, query: str, top_k: int | None = None) -> str:
        final_top_k = self._default_top_k if top_k is None else top_k
        _validate_final_top_k(final_top_k)
        return format_results(
            self._get_retriever().retrieve(query, top_k=final_top_k)
        )

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            assert self._retriever_factory is not None
            self._retriever = self._retriever_factory()
        return self._retriever


def build_default_knowledge_search_tool() -> KnowledgeSearchTool:
    """Build a lazy hybrid search tool after validating its configuration."""
    config = HybridRetrievalConfig.from_env()

    def create_retriever() -> Retriever:
        # Keep heavy model loading and local Qdrant opening out of application
        # composition; both occur only when the Agent actually invokes the tool.
        from insight_agent.indexing import QdrantVectorStore, SentenceTransformerEmbedder
        from insight_agent.retrieval.hybrid import HybridRetriever
        from insight_agent.retrieval.reranker import CrossEncoderReranker
        from insight_agent.retrieval.retriever import VectorRetriever
        from insight_agent.retrieval.sparse import BM25Retriever

        vector_store = QdrantVectorStore()
        dense_retriever = VectorRetriever(
            embedder=SentenceTransformerEmbedder(),
            vector_store=vector_store,
        )
        sparse_retriever = BM25Retriever(vector_store.load_chunks)
        reranker = CrossEncoderReranker(config.reranker_model)
        return HybridRetriever(
            dense_retriever,
            sparse_retriever,
            reranker,
            config=config,
        )

    return KnowledgeSearchTool(
        retriever_factory=create_retriever,
        default_top_k=config.final_top_k,
    )


def _validate_final_top_k(top_k: int) -> None:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 8
    ):
        raise ValueError("top_k must be an integer between 1 and 8")
