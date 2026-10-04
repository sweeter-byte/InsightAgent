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


@dataclass(frozen=True, slots=True)
class RankedResult:
    """Retrieval-internal scores layered over one unchanged domain result."""

    result: RetrievalResult
    fusion_score: float = 0.0
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RankingTraceEntry:
    """One compact debug entry for a retrieval ranking stage."""

    chunk_id: str
    score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Debug-only snapshots of every hybrid retrieval stage."""

    query: str
    dense: tuple[RankingTraceEntry, ...]
    sparse: tuple[RankingTraceEntry, ...]
    fused: tuple[RankingTraceEntry, ...]
    reranked: tuple[RankingTraceEntry, ...]
    final: tuple[RankingTraceEntry, ...]
