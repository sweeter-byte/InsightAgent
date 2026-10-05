"""Provider-neutral data models for Web Search retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from insight_agent.ingestion import Document


@dataclass(frozen=True, slots=True)
class WebSearchHit:
    """One candidate page returned by a search provider."""

    rank: int
    title: str
    url: str
    snippet: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class WebFetchFailure:
    """A candidate page that the URL Loader could not turn into content."""

    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class WebRetrievalResult:
    """Search hits, fetched documents, and page-level failures for one query."""

    query: str
    hits: list[WebSearchHit]
    documents: list[Document]
    failures: list[WebFetchFailure]
