"""Unified candidate Evidence produced from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from insight_agent.routing import RetrievalSource


@dataclass(frozen=True, slots=True)
class Evidence:
    """One ungraded candidate with enough provenance to trace its origin."""

    id: str
    task_id: str
    retrieval_source: RetrievalSource
    origin_id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

