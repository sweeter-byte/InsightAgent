"""Stable public data models for research runs and runtime events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
        RunStatus.INTERRUPTED,
    }
)


class CreateResearchRunRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class RunRecord(BaseModel):
    """Runtime metadata; it deliberately does not contain ResearchState."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    thread_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    final_output: str | None = None

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        thread_id: str,
        status: RunStatus = RunStatus.QUEUED,
    ) -> RunRecord:
        now = utc_now()
        return cls(
            run_id=run_id,
            thread_id=thread_id,
            status=status,
            created_at=now,
            updated_at=now,
        )


class RuntimeEvent(BaseModel):
    """A sanitized, client-facing event persisted independently of graph state."""

    model_config = ConfigDict(frozen=True)

    event_id: str | None = None
    run_id: str
    type: str
    created_at: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
