"""Domain models for per-task retrieval-source routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetrievalSource(str, Enum):
    """The finite set of source families a research task may target."""

    LOCAL = "local"
    WEB = "web"
    VISION = "vision"


@dataclass(slots=True)
class RouteDecision:
    """Control information selecting one primary source for one task."""

    task_id: str
    source: RetrievalSource
    reason: str


class RoutingError(RuntimeError):
    """Raised when a route cannot be produced or validated."""

