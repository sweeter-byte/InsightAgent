"""Agent tool adapter for local knowledge-base retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.retrieval.formatting import format_results
from insight_agent.retrieval.hybrid import HybridRetriever
from insight_agent.retrieval.models import RetrievalResult

if TYPE_CHECKING:
    from insight_agent.indexing import QdrantVectorStore


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
        return format_results(self.retrieve(query, top_k=top_k))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return raw results through the same validated lazy retriever."""
        final_top_k = self._default_top_k if top_k is None else top_k
        _validate_final_top_k(final_top_k)
        return self._get_retriever().retrieve(query, top_k=final_top_k)

    def _get_retriever(self) -> Retriever:
        if self._retriever is None:
            assert self._retriever_factory is not None
            self._retriever = self._retriever_factory()
        return self._retriever

    def close(self) -> None:
        """Release resources only if the lazy retriever was constructed."""
        if self._retriever is None:
            return
        close = getattr(self._retriever, "close", None)
        if callable(close):
            close()


def build_default_knowledge_search_tool(
    config: HybridRetrievalConfig | None = None,
) -> KnowledgeSearchTool:
    """Build a lazy hybrid search tool after validating its configuration."""
    resolved_config = config or HybridRetrievalConfig.from_env()

    def create_retriever() -> Retriever:
        # Keep heavy model loading and local Qdrant opening out of application
        # composition; both occur only when the Agent actually invokes the tool.
        return build_default_hybrid_retriever(resolved_config)

    return KnowledgeSearchTool(
        retriever_factory=create_retriever,
        default_top_k=resolved_config.final_top_k,
    )


def build_default_hybrid_retriever(
    config: HybridRetrievalConfig,
    vector_store: QdrantVectorStore | None = None,
) -> HybridRetriever:
    """Compose the project Hybrid Retriever, optionally over a shared store."""
    from insight_agent.indexing import QdrantVectorStore, SentenceTransformerEmbedder
    from insight_agent.retrieval.reranker import CrossEncoderReranker
    from insight_agent.retrieval.retriever import VectorRetriever
    from insight_agent.retrieval.sparse import BM25Retriever

    owns_store = vector_store is None
    store = QdrantVectorStore() if owns_store else vector_store
    assert store is not None
    try:
        dense_retriever = VectorRetriever(
            embedder=SentenceTransformerEmbedder(),
            vector_store=store,
        )
        sparse_retriever = BM25Retriever(store.load_chunks)
        reranker = CrossEncoderReranker(config.reranker_model)
        return HybridRetriever(
            dense_retriever,
            sparse_retriever,
            reranker,
            config=config,
            close_callback=store.close if owns_store else None,
        )
    except BaseException:
        if owns_store:
            store.close()
        raise


def _validate_final_top_k(top_k: int) -> None:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 8
    ):
        raise ValueError("top_k must be an integer between 1 and 8")
