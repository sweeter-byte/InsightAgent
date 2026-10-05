"""Orchestration from candidate search to existing URL ingestion."""

from __future__ import annotations

from typing import Protocol

from insight_agent.ingestion import Document, load_url
from insight_agent.web_search.models import (
    WebFetchFailure,
    WebRetrievalResult,
    WebSearchHit,
)
from insight_agent.web_search.provider import SearchProvider


class UrlLoader(Protocol):
    """Public URL Loader call shape used by the retriever."""

    def __call__(self, url: str, *, timeout: float) -> list[Document]:
        ...


class WebRetriever:
    """Search, deduplicate candidate URLs, fetch pages, and collect failures."""

    def __init__(
        self,
        provider: SearchProvider,
        *,
        url_loader: UrlLoader = load_url,
        timeout: float = 30.0,
        search_limit: int = 5,
        fetch_limit: int = 3,
    ) -> None:
        _validate_limits(search_limit, fetch_limit)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        self.provider = provider
        self.url_loader = url_loader
        self.timeout = float(timeout)
        self.search_limit = search_limit
        self.fetch_limit = fetch_limit

    def retrieve(
        self,
        query: str,
        *,
        search_limit: int | None = None,
        fetch_limit: int | None = None,
    ) -> WebRetrievalResult:
        """Retrieve a bounded set of public pages for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        effective_search_limit = (
            self.search_limit if search_limit is None else search_limit
        )
        effective_fetch_limit = self.fetch_limit if fetch_limit is None else fetch_limit
        _validate_limits(effective_search_limit, effective_fetch_limit)

        hits = self.provider.search(query, limit=effective_search_limit)
        candidates = _first_unique_hits(hits, limit=effective_fetch_limit)
        documents: list[Document] = []
        failures: list[WebFetchFailure] = []

        for hit in candidates:
            try:
                loaded = self.url_loader(hit.url, timeout=self.timeout)
                usable = _usable_documents(loaded)
                if not usable:
                    raise ValueError("URL Loader returned empty page content")
                documents.extend(_with_search_metadata(document, query, hit) for document in usable)
            except Exception as exc:
                failures.append(
                    WebFetchFailure(
                        url=hit.url,
                        reason=str(exc).strip() or type(exc).__name__,
                    )
                )

        return WebRetrievalResult(
            query=query,
            hits=list(hits),
            documents=documents,
            failures=failures,
        )


def _validate_limits(search_limit: int, fetch_limit: int) -> None:
    for name, value in (
        ("search_limit", search_limit),
        ("fetch_limit", fetch_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if fetch_limit > search_limit:
        raise ValueError("fetch_limit must not exceed search_limit")


def _first_unique_hits(
    hits: list[WebSearchHit],
    *,
    limit: int,
) -> list[WebSearchHit]:
    unique: list[WebSearchHit] = []
    seen_urls: set[str] = set()
    for hit in hits:
        if hit.url in seen_urls:
            continue
        seen_urls.add(hit.url)
        unique.append(hit)
        if len(unique) == limit:
            break
    return unique


def _usable_documents(documents: list[Document]) -> list[Document]:
    if not isinstance(documents, list):
        raise TypeError("URL Loader must return a list of Documents")
    usable: list[Document] = []
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("URL Loader must return only Documents")
        if document.content.strip():
            usable.append(document)
    return usable


def _with_search_metadata(
    document: Document,
    query: str,
    hit: WebSearchHit,
) -> Document:
    metadata = {
        **document.metadata,
        "search_query": query,
        "search_rank": hit.rank,
        "search_title": hit.title,
        "search_snippet": hit.snippet,
    }
    if hit.score is not None:
        metadata["search_score"] = hit.score
    return Document(
        content=document.content,
        source=document.source,
        source_type=document.source_type,
        metadata=metadata,
    )
