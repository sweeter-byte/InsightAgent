"""Network-free tests for the Tavily candidate-search adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from insight_agent.web_search import (
    TavilySearchProvider,
    WebSearchConfigurationError,
    WebSearchError,
    WebSearchHit,
)


class FakeResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error
        self.request = httpx.Request("POST", "https://api.tavily.com/search")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


def test_tavily_provider_maps_ordered_hits_and_candidate_only_request() -> None:
    response = FakeResponse(
        {
            "answer": "must be ignored",
            "results": [
                {
                    "title": "First",
                    "url": "https://example.com/first",
                    "content": "First snippet",
                    "score": 0.92,
                    "raw_content": "must be ignored",
                },
                {
                    "title": "Second",
                    "url": "https://example.com/second",
                    "content": "Second snippet",
                },
            ],
        }
    )

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        return_value=response,
    ) as post:
        hits = TavilySearchProvider(
            api_key="tvly-secret",
            timeout=7.5,
        ).search("agent memory", limit=2)

    assert hits == [
        WebSearchHit(
            rank=1,
            title="First",
            url="https://example.com/first",
            snippet="First snippet",
            score=0.92,
        ),
        WebSearchHit(
            rank=2,
            title="Second",
            url="https://example.com/second",
            snippet="Second snippet",
            score=None,
        ),
    ]
    post.assert_called_once_with(
        "https://api.tavily.com/search",
        json={
            "query": "agent memory",
            "search_depth": "basic",
            "max_results": 2,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
        },
        headers={"Authorization": "Bearer tvly-secret"},
        timeout=7.5,
    )


@pytest.mark.parametrize(
    "transient_error",
    [
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("connection failed"),
    ],
    ids=["connect-timeout", "read-timeout", "connect-error"],
)
def test_tavily_provider_retries_transient_network_error_then_succeeds(
    transient_error: httpx.RequestError,
) -> None:
    response = FakeResponse({"results": []})
    delays: list[float] = []

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        side_effect=[transient_error, response],
    ) as post:
        hits = TavilySearchProvider(
            api_key="tvly-test",
            sleeper=delays.append,
        ).search("query", limit=3)

    assert hits == []
    assert post.call_count == 2
    assert delays == [0.5]


def test_tavily_provider_stops_after_three_transient_failures() -> None:
    timeout = httpx.ConnectTimeout("connect timed out")
    delays: list[float] = []

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        side_effect=timeout,
    ) as post:
        with pytest.raises(
            WebSearchError,
            match="Tavily search failed after 3 attempts: ConnectTimeout",
        ) as excinfo:
            TavilySearchProvider(
                api_key="tvly-test",
                sleeper=delays.append,
            ).search("query", limit=3)

    assert post.call_count == 3
    assert delays == [0.5, 1.0]
    assert excinfo.value.__cause__ is timeout


def test_tavily_provider_rejects_missing_api_key() -> None:
    with pytest.raises(WebSearchConfigurationError, match="TAVILY_API_KEY"):
        TavilySearchProvider(api_key="  ")


@pytest.mark.parametrize(
    ("query", "limit", "message"),
    [
        ("", 3, "query"),
        ("valid", 0, "limit"),
        ("valid", True, "limit"),
    ],
)
def test_tavily_provider_validates_search_input(
    query: str,
    limit: int,
    message: str,
) -> None:
    provider = TavilySearchProvider(api_key="tvly-test")

    with pytest.raises(WebSearchError, match=message):
        provider.search(query, limit=limit)


def test_tavily_provider_does_not_retry_other_request_errors() -> None:
    request_error = httpx.WriteError("write failed")

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        side_effect=request_error,
    ) as post:
        with pytest.raises(WebSearchError, match="Tavily search request failed") as excinfo:
            TavilySearchProvider(
                api_key="tvly-test",
                sleeper=lambda _: pytest.fail("non-transient error must not sleep"),
            ).search("query", limit=3)

    assert post.call_count == 1
    assert excinfo.value.__cause__ is request_error


def test_tavily_provider_wraps_http_status_errors() -> None:
    response = FakeResponse({}, status_code=401)

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        return_value=response,
    ) as post:
        with pytest.raises(WebSearchError, match="HTTP 401"):
            TavilySearchProvider(
                api_key="tvly-test",
                sleeper=lambda _: pytest.fail("HTTP errors must not sleep"),
            ).search("query", limit=3)

    assert post.call_count == 1


def test_tavily_provider_rejects_invalid_json() -> None:
    response = FakeResponse(None, json_error=ValueError("not json"))

    with patch(
        "insight_agent.web_search.provider.httpx.post",
        return_value=response,
    ):
        with pytest.raises(WebSearchError, match="invalid JSON"):
            TavilySearchProvider(api_key="tvly-test").search("query", limit=3)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"results": "not-a-list"},
        {"results": [{"title": "Missing URL", "content": "snippet"}]},
        {
            "results": [
                {"title": "Bad score", "url": "https://a.test", "score": "high"}
            ]
        },
    ],
)
def test_tavily_provider_rejects_malformed_response(payload: Any) -> None:
    with patch(
        "insight_agent.web_search.provider.httpx.post",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(WebSearchError, match="invalid response"):
            TavilySearchProvider(api_key="tvly-test").search("query", limit=3)
