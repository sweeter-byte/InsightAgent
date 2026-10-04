"""Agent tool adapter for local vector retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

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
    ) -> None:
        if (retriever is None) == (retriever_factory is None):
            raise ValueError("provide exactly one of retriever or retriever_factory")
        self._retriever = retriever
        self._retriever_factory = retriever_factory

    def __call__(self, query: str, top_k: int = 5) -> str:
        return format_results(self._get_retriever().retrieve(query, top_k=top_k))

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            assert self._retriever_factory is not None
            self._retriever = self._retriever_factory()
        return self._retriever


def build_default_knowledge_search_tool() -> KnowledgeSearchTool:
    """Build a lazily initialized tool using the configured Chapter 4 stack."""

    def create_retriever() -> Retriever:
        # Keep heavy model loading and local Qdrant opening out of application
        # composition; both occur only when the Agent actually invokes the tool.
        from insight_agent.indexing import QdrantVectorStore, SentenceTransformerEmbedder
        from insight_agent.retrieval.retriever import VectorRetriever

        return VectorRetriever(
            embedder=SentenceTransformerEmbedder(),
            vector_store=QdrantVectorStore(),
        )

    return KnowledgeSearchTool(retriever_factory=create_retriever)
