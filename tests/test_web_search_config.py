"""Configuration and domain-model tests for Web Search."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from insight_agent.ingestion import Document, SourceType
from insight_agent.web_search import (
    WebFetchFailure,
    WebRetrievalResult,
    WebSearchConfig,
    WebSearchConfigurationError,
    WebSearchHit,
)


def test_web_search_models_keep_search_and_document_concepts_separate() -> None:
    hit = WebSearchHit(
        rank=1,
        title="Project page",
        url="https://example.com/project",
        snippet="A public project page.",
        score=0.91,
    )
    document = Document(
        content="Fetched page body",
        source="https://example.com/project",
        source_type=SourceType.URL,
        metadata={"title": "Fetched title"},
    )
    failure = WebFetchFailure(
        url="https://example.com/broken",
        reason="HTTP 503",
    )

    result = WebRetrievalResult(
        query="agent memory implementations",
        hits=[hit],
        documents=[document],
        failures=[failure],
    )

    assert result.query == "agent memory implementations"
    assert result.hits == [hit]
    assert result.documents == [document]
    assert result.failures == [failure]
    with pytest.raises(FrozenInstanceError):
        hit.rank = 2  # type: ignore[misc]


def test_web_search_config_is_absent_without_api_key() -> None:
    assert WebSearchConfig.from_env({}, required=False) is None


def test_web_search_config_requires_api_key_when_requested() -> None:
    with pytest.raises(WebSearchConfigurationError, match="TAVILY_API_KEY"):
        WebSearchConfig.from_env({}, required=True)


def test_web_search_config_uses_defaults() -> None:
    config = WebSearchConfig.from_env({"TAVILY_API_KEY": " tvly-test "})

    assert config == WebSearchConfig(
        api_key="tvly-test",
        timeout=30.0,
        search_limit=5,
        fetch_limit=3,
    )


def test_web_search_config_parses_explicit_values() -> None:
    config = WebSearchConfig.from_env(
        {
            "TAVILY_API_KEY": "tvly-test",
            "WEB_SEARCH_TIMEOUT": "12.5",
            "WEB_SEARCH_LIMIT": "8",
            "WEB_FETCH_LIMIT": "4",
        }
    )

    assert config == WebSearchConfig(
        api_key="tvly-test",
        timeout=12.5,
        search_limit=8,
        fetch_limit=4,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("WEB_SEARCH_TIMEOUT", "zero"),
        ("WEB_SEARCH_TIMEOUT", "0"),
        ("WEB_SEARCH_TIMEOUT", "nan"),
        ("WEB_SEARCH_LIMIT", "1.5"),
        ("WEB_SEARCH_LIMIT", "0"),
        ("WEB_FETCH_LIMIT", "-1"),
    ],
)
def test_web_search_config_rejects_invalid_numbers(name: str, value: str) -> None:
    environ = {"TAVILY_API_KEY": "tvly-test", name: value}

    with pytest.raises(WebSearchConfigurationError, match=name):
        WebSearchConfig.from_env(environ)


def test_web_search_config_rejects_fetch_limit_above_search_limit() -> None:
    with pytest.raises(WebSearchConfigurationError, match="WEB_FETCH_LIMIT"):
        WebSearchConfig.from_env(
            {
                "TAVILY_API_KEY": "tvly-test",
                "WEB_SEARCH_LIMIT": "2",
                "WEB_FETCH_LIMIT": "3",
            }
        )
