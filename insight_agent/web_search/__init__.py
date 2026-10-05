"""Public API for provider-neutral Web Search retrieval."""

from insight_agent.web_search.config import WebSearchConfig
from insight_agent.web_search.errors import (
    WebSearchConfigurationError,
    WebSearchError,
)
from insight_agent.web_search.models import (
    WebFetchFailure,
    WebRetrievalResult,
    WebSearchHit,
)
from insight_agent.web_search.provider import SearchProvider, TavilySearchProvider
from insight_agent.web_search.retriever import WebRetriever

__all__ = [
    "WebFetchFailure",
    "WebRetrievalResult",
    "WebSearchConfig",
    "WebSearchConfigurationError",
    "WebSearchError",
    "WebSearchHit",
    "SearchProvider",
    "TavilySearchProvider",
    "WebRetriever",
]
