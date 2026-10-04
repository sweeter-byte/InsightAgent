"""Project-owned models and errors for vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_agent.ingestion import SourceType


class RetrievalPayloadError(ValueError):
    """Raised when an indexed point cannot be restored from its payload."""


@dataclass(slots=True)
class RetrievalResult:
    """One chunk recalled for a query, including its similarity score."""

    chunk_id: str
    score: float
    content: str
    document_id: str
    source: str
    source_type: SourceType
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
