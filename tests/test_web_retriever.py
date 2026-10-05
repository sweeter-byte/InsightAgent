"""Tests for search-to-URL-Loader Web retrieval orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from insight_agent.ingestion import Document, IngestionError, SourceType
from insight_agent.web_search import (
    WebRetriever,
    WebSearchError,
    WebSearchHit,
)


def _hit(
    rank: int,
    url: str,
    *,
    score: float | None = None,
) -> WebSearchHit:
    return WebSearchHit(
        rank=rank,
        title=f"Search title {rank}",
        url=url,
        snippet=f"Search snippet {rank}",
        score=score,
    )


class FakeSearchProvider:
    def __init__(
        self,
        hits: list[WebSearchHit],
        *,
        error: Exception | None = None,
    ) -> None:
        self.hits = hits
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        self.calls.append({"query": query, "limit": limit})
        if self.error is not None:
            raise self.error
        return list(self.hits)


class FakeUrlLoader:
    def __init__(self, outcomes: dict[str, list[Document] | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, timeout: float) -> list[Document]:
        self.calls.append({"url": url, "timeout": timeout})
        outcome = self.outcomes[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _document(url: str, content: str = "Fetched body") -> Document:
    return Document(
        content=content,
        source=f"{url}/final",
        source_type=SourceType.URL,
        metadata={
            "title": "Loader title",
            "final_url": f"{url}/final",
            "content_type": "text/html",
        },
    )


def test_retriever_passes_query_limits_deduplicates_and_enriches_documents() -> None:
    first = "https://example.com/first"
    second = "https://example.com/second"
    third = "https://example.com/third"
    provider = FakeSearchProvider(
        [
            _hit(1, first, score=0.9),
            _hit(2, first, score=0.8),
            _hit(3, second),
            _hit(4, third),
        ]
    )
    original_first = _document(first)
    loader = FakeUrlLoader(
        {
            first: [original_first],
            second: [_document(second)],
            third: [_document(third)],
        }
    )
    retriever = WebRetriever(
        provider,
        url_loader=loader,
        timeout=9.0,
        search_limit=7,
        fetch_limit=2,
    )

    result = retriever.retrieve("agent memory implementations")

    assert provider.calls == [
        {"query": "agent memory implementations", "limit": 7}
    ]
    assert loader.calls == [
        {"url": first, "timeout": 9.0},
        {"url": second, "timeout": 9.0},
    ]
    assert result.query == "agent memory implementations"
    assert result.hits == provider.hits
    assert len(result.documents) == 2
    assert result.failures == []
    first_result = result.documents[0]
    assert first_result is not original_first
    assert first_result.content == "Fetched body"
    assert first_result.source == f"{first}/final"
    assert first_result.metadata == {
        "title": "Loader title",
        "final_url": f"{first}/final",
        "content_type": "text/html",
        "search_query": "agent memory implementations",
        "search_rank": 1,
        "search_title": "Search title 1",
        "search_snippet": "Search snippet 1",
        "search_score": 0.9,
    }
    assert original_first.metadata == {
        "title": "Loader title",
        "final_url": f"{first}/final",
        "content_type": "text/html",
    }
    assert "search_score" not in result.documents[1].metadata


def test_retriever_call_limits_override_configured_defaults() -> None:
    url = "https://example.com/one"
    provider = FakeSearchProvider([_hit(1, url)])
    loader = FakeUrlLoader({url: [_document(url)]})
    retriever = WebRetriever(
        provider,
        url_loader=loader,
        search_limit=5,
        fetch_limit=3,
    )

    retriever.retrieve("query", search_limit=1, fetch_limit=1)

    assert provider.calls == [{"query": "query", "limit": 1}]
    assert [call["url"] for call in loader.calls] == [url]


def test_retriever_records_page_failure_and_continues() -> None:
    broken = "https://example.com/broken"
    good = "https://example.com/good"
    provider = FakeSearchProvider([_hit(1, broken), _hit(2, good)])
    loader = FakeUrlLoader(
        {
            broken: IngestionError("HTTP 503"),
            good: [_document(good, "Good page")],
        }
    )

    result = WebRetriever(provider, url_loader=loader).retrieve("query")

    assert [call["url"] for call in loader.calls] == [broken, good]
    assert [document.content for document in result.documents] == ["Good page"]
    assert len(result.failures) == 1
    assert result.failures[0].url == broken
    assert result.failures[0].reason == "HTTP 503"


@pytest.mark.parametrize(
    "documents",
    [
        [],
        [_document("https://example.com/empty", "   ")],
    ],
)
def test_retriever_records_empty_page_as_failure(
    documents: list[Document],
) -> None:
    url = "https://example.com/empty"
    provider = FakeSearchProvider([_hit(1, url)])
    loader = FakeUrlLoader({url: documents})

    result = WebRetriever(provider, url_loader=loader).retrieve("query")

    assert result.documents == []
    assert len(result.failures) == 1
    assert result.failures[0].url == url
    assert "empty" in result.failures[0].reason.lower()


def test_retriever_does_not_disguise_provider_failure_as_success() -> None:
    error = WebSearchError("provider unavailable")
    provider = FakeSearchProvider([], error=error)
    loader = FakeUrlLoader({})

    with pytest.raises(WebSearchError, match="provider unavailable") as excinfo:
        WebRetriever(provider, url_loader=loader).retrieve("query")

    assert excinfo.value is error
    assert loader.calls == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"query": ""}, "query"),
        ({"search_limit": 0}, "search_limit"),
        ({"fetch_limit": 0}, "fetch_limit"),
        ({"search_limit": 1, "fetch_limit": 2}, "fetch_limit"),
    ],
)
def test_retriever_validates_runtime_inputs(
    overrides: dict[str, Any],
    message: str,
) -> None:
    retriever = WebRetriever(FakeSearchProvider([]), url_loader=FakeUrlLoader({}))
    arguments: dict[str, Any] = {"query": "query"}
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        retriever.retrieve(**arguments)
