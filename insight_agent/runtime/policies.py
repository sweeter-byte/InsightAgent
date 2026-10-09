"""Centralized configuration for single-process runtime execution."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_REDIS_URL = "redis://localhost:6379/0"


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    max_concurrency: int = 2
    run_timeout_seconds: float = 900.0
    infra_retry_attempts: int = 3
    infra_retry_backoff_seconds: float = 0.2
    event_ttl_seconds: int = 86_400

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be at least 1")
        if (
            isinstance(self.run_timeout_seconds, bool)
            or not isinstance(self.run_timeout_seconds, (int, float))
            or not math.isfinite(self.run_timeout_seconds)
            or self.run_timeout_seconds <= 0
        ):
            raise ValueError("run_timeout_seconds must be greater than 0")
        if (
            isinstance(self.infra_retry_attempts, bool)
            or not isinstance(self.infra_retry_attempts, int)
            or self.infra_retry_attempts < 1
        ):
            raise ValueError("infra_retry_attempts must be at least 1")
        if (
            isinstance(self.infra_retry_backoff_seconds, bool)
            or not isinstance(self.infra_retry_backoff_seconds, (int, float))
            or not math.isfinite(self.infra_retry_backoff_seconds)
            or self.infra_retry_backoff_seconds < 0
        ):
            raise ValueError("infra_retry_backoff_seconds must not be negative")
        if (
            isinstance(self.event_ttl_seconds, bool)
            or not isinstance(self.event_ttl_seconds, int)
            or self.event_ttl_seconds < 1
        ):
            raise ValueError("event_ttl_seconds must be at least 1")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimePolicy:
        values = os.environ if environ is None else environ
        return cls(
            max_concurrency=_env_int(values, "RUNTIME_MAX_CONCURRENCY", 2),
            run_timeout_seconds=_env_float(
                values,
                "RUNTIME_RUN_TIMEOUT_SECONDS",
                900.0,
            ),
            infra_retry_attempts=_env_int(
                values,
                "RUNTIME_INFRA_RETRY_ATTEMPTS",
                3,
            ),
            infra_retry_backoff_seconds=_env_float(
                values,
                "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS",
                0.2,
            ),
            event_ttl_seconds=_env_int(
                values,
                "RUNTIME_EVENT_TTL_SECONDS",
                86_400,
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Redis connection and execution policy for one runtime instance."""

    redis_url: str = DEFAULT_REDIS_URL
    policy: RuntimePolicy = RuntimePolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.redis_url, str) or not self.redis_url.strip():
            raise ValueError("redis_url must not be empty")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        values = os.environ if environ is None else environ
        redis_url = values.get("RUNTIME_REDIS_URL", DEFAULT_REDIS_URL).strip()
        return cls(
            redis_url=redis_url or DEFAULT_REDIS_URL,
            policy=RuntimePolicy.from_env(values),
        )


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value
