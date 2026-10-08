"""Centralized configuration for single-process runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    max_concurrency: int = 2
    run_timeout_seconds: float = 900.0
    infra_retry_attempts: int = 3
    infra_retry_backoff_seconds: float = 0.2
    event_ttl_seconds: int = 86_400

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.run_timeout_seconds <= 0:
            raise ValueError("run_timeout_seconds must be greater than 0")
        if self.infra_retry_attempts < 1:
            raise ValueError("infra_retry_attempts must be at least 1")
        if self.infra_retry_backoff_seconds < 0:
            raise ValueError("infra_retry_backoff_seconds must not be negative")
        if self.event_ttl_seconds < 1:
            raise ValueError("event_ttl_seconds must be at least 1")

    @classmethod
    def from_env(cls) -> RuntimePolicy:
        return cls(
            max_concurrency=_env_int("RUNTIME_MAX_CONCURRENCY", 2),
            run_timeout_seconds=_env_float(
                "RUNTIME_RUN_TIMEOUT_SECONDS", 900.0
            ),
            infra_retry_attempts=_env_int("RUNTIME_INFRA_RETRY_ATTEMPTS", 3),
            infra_retry_backoff_seconds=_env_float(
                "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS", 0.2
            ),
            event_ttl_seconds=_env_int("RUNTIME_EVENT_TTL_SECONDS", 86_400),
        )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
