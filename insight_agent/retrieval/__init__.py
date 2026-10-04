"""Public API for InsightAgent's local hybrid retrieval layer."""

from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.retrieval.fusion import rrf_fuse
from insight_agent.retrieval.formatting import EMPTY_RESULTS_MESSAGE, format_results
from insight_agent.retrieval.hybrid import HybridRetriever
from insight_agent.retrieval.models import (
    RankedResult,
    RankingTraceEntry,
    RetrievalPayloadError,
    RetrievalResult,
    RetrievalTrace,
)
from insight_agent.retrieval.reranker import CrossEncoderReranker, Reranker
from insight_agent.retrieval.retriever import SearchableVectorStore, VectorRetriever
from insight_agent.retrieval.sparse import BM25Retriever, tokenize
from insight_agent.retrieval.tool import (
    SEARCH_KNOWLEDGE_BASE_SCHEMA,
    KnowledgeSearchTool,
    build_default_knowledge_search_tool,
)

__all__ = [
    "EMPTY_RESULTS_MESSAGE",
    "BM25Retriever",
    "CrossEncoderReranker",
    "HybridRetrievalConfig",
    "HybridRetriever",
    "KnowledgeSearchTool",
    "RankedResult",
    "RankingTraceEntry",
    "Reranker",
    "RetrievalPayloadError",
    "RetrievalResult",
    "RetrievalTrace",
    "SEARCH_KNOWLEDGE_BASE_SCHEMA",
    "SearchableVectorStore",
    "VectorRetriever",
    "build_default_knowledge_search_tool",
    "format_results",
    "rrf_fuse",
    "tokenize",
]
