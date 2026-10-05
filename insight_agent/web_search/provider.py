"""Search-provider abstraction and Tavily REST implementation."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from insight_agent.web_search.errors import (
    WebSearchConfigurationError,
    WebSearchError,
)
from insight_agent.web_search.models import WebSearchHit


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchProvider(Protocol):
    """Discover candidate public pages for one query."""

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        """Return provider-neutral hits in provider ranking order."""
        ...


class TavilySearchProvider:
    """Thin adapter around Tavily's candidate Search REST endpoint."""

    def __init__(self, *, api_key: str, timeout: float = 30.0) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise WebSearchConfigurationError("TAVILY_API_KEY must not be empty")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise WebSearchConfigurationError(
                "WEB_SEARCH_TIMEOUT must be a positive number"
            )
        self.api_key = api_key.strip()
        self.timeout = float(timeout)

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise WebSearchError("search query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise WebSearchError("search limit must be a positive integer")

        try:
            response = httpx.post(
                TAVILY_SEARCH_URL,
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": limit,
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": False,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise WebSearchError(
                f"Tavily search failed with HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise WebSearchError(f"Tavily search request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchError("Tavily search returned invalid JSON") from exc

        try:
            return _parse_hits(payload, limit=limit)
        except (KeyError, TypeError, ValueError) as exc:
            raise WebSearchError(f"Tavily search returned an invalid response: {exc}") from exc


def _parse_hits(payload: Any, *, limit: int) -> list[WebSearchHit]:
    if not isinstance(payload, dict):
        raise TypeError("response must be an object")
    results = payload["results"]
    if not isinstance(results, list):
        raise TypeError("results must be a list")

    hits: list[WebSearchHit] = []
    for rank, item in enumerate(results[:limit], start=1):
        if not isinstance(item, dict):
            raise TypeError(f"result {rank} must be an object")
        title = item.get("title", "")
        url = item["url"]
        snippet = item.get("content", "")
        score = item.get("score")
        if not isinstance(title, str):
            raise TypeError(f"result {rank} title must be a string")
        if not isinstance(url, str) or not url.strip():
            raise TypeError(f"result {rank} url must be a non-empty string")
        if not isinstance(snippet, str):
            raise TypeError(f"result {rank} content must be a string")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(f"result {rank} score must be numeric or null")
            score = float(score)
        hits.append(
            WebSearchHit(
                rank=rank,
                title=title,
                url=url,
                snippet=snippet,
                score=score,
            )
        )
    return hits
