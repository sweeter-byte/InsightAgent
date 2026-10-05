"""Errors exposed by the Web Search retrieval layer."""

from __future__ import annotations


class WebSearchError(RuntimeError):
    """Raised when candidate discovery cannot produce a valid result set."""


class WebSearchConfigurationError(WebSearchError):
    """Raised when Web Search configuration is missing or invalid."""
