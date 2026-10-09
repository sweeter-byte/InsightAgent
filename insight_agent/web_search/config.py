"""Environment-backed configuration for Web Search retrieval."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from insight_agent.web_search.errors import WebSearchConfigurationError


DEFAULT_WEB_SEARCH_TIMEOUT = 30.0
DEFAULT_WEB_SEARCH_LIMIT = 5
DEFAULT_WEB_FETCH_LIMIT = 3


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """Validated Tavily and page-fetch settings for one application runtime."""

    api_key: str = field(repr=False)
    timeout: float = DEFAULT_WEB_SEARCH_TIMEOUT
    search_limit: int = DEFAULT_WEB_SEARCH_LIMIT
    fetch_limit: int = DEFAULT_WEB_FETCH_LIMIT

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise WebSearchConfigurationError("TAVILY_API_KEY must not be empty")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise WebSearchConfigurationError(
                "WEB_SEARCH_TIMEOUT must be a positive finite number"
            )
        for name, value in (
            ("WEB_SEARCH_LIMIT", self.search_limit),
            ("WEB_FETCH_LIMIT", self.fetch_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WebSearchConfigurationError(
                    f"{name} must be a positive integer"
                )
        if self.fetch_limit > self.search_limit:
            raise WebSearchConfigurationError(
                "WEB_FETCH_LIMIT must not exceed WEB_SEARCH_LIMIT"
            )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        required: bool = False,
    ) -> WebSearchConfig | None:
        """Read settings, returning ``None`` when optional Web Search is off."""
        values = os.environ if environ is None else environ
        api_key = values.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            if required:
                raise WebSearchConfigurationError(
                    "Missing required environment variable 'TAVILY_API_KEY'. "
                    "Set it in your shell or `.env` file to enable Web Search."
                )
            return None

        timeout = _positive_float(
            values,
            "WEB_SEARCH_TIMEOUT",
            DEFAULT_WEB_SEARCH_TIMEOUT,
        )
        search_limit = _positive_int(
            values,
            "WEB_SEARCH_LIMIT",
            DEFAULT_WEB_SEARCH_LIMIT,
        )
        fetch_limit = _positive_int(
            values,
            "WEB_FETCH_LIMIT",
            DEFAULT_WEB_FETCH_LIMIT,
        )
        return cls(
            api_key=api_key,
            timeout=timeout,
            search_limit=search_limit,
            fetch_limit=fetch_limit,
        )


def _positive_float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise WebSearchConfigurationError(
            f"{name} must be a positive finite number; received {raw_value!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise WebSearchConfigurationError(
            f"{name} must be a positive finite number; received {raw_value!r}."
        )
    return value


def _positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise WebSearchConfigurationError(
            f"{name} must be a positive integer; received {raw_value!r}."
        ) from exc
    if value <= 0:
        raise WebSearchConfigurationError(
            f"{name} must be a positive integer; received {raw_value!r}."
        )
    return value
