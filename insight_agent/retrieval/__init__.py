"""Public API for InsightAgent's local vector retrieval layer."""

from insight_agent.retrieval.formatting import EMPTY_RESULTS_MESSAGE, format_results
from insight_agent.retrieval.models import RetrievalPayloadError, RetrievalResult
from insight_agent.retrieval.retriever import SearchableVectorStore, VectorRetriever
from insight_agent.retrieval.tool import (
    SEARCH_KNOWLEDGE_BASE_SCHEMA,
    KnowledgeSearchTool,
    build_default_knowledge_search_tool,
)

__all__ = [
    "EMPTY_RESULTS_MESSAGE",
    "KnowledgeSearchTool",
    "RetrievalPayloadError",
    "RetrievalResult",
    "SEARCH_KNOWLEDGE_BASE_SCHEMA",
    "SearchableVectorStore",
    "VectorRetriever",
    "build_default_knowledge_search_tool",
    "format_results",
]
