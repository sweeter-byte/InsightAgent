"""Provider-neutral result models for task-conditioned Vision Retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VisionAnalysis:
    """One task-focused analysis of one original image."""

    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisionFailure:
    """One indexed image that could not produce an analysis."""

    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class VisionRetrievalResult:
    """All visual analyses and per-image failures for one research task."""

    task_id: str
    query: str
    analyses: list[VisionAnalysis]
    failures: list[VisionFailure]
    no_candidates: bool = False

    def __post_init__(self) -> None:
        if self.no_candidates:
            if self.analyses or self.failures:
                raise ValueError(
                    "no_candidates=True requires empty analyses and failures"
                )
            return
        if not self.analyses and not self.failures:
            raise ValueError(
                "a candidate result requires at least one analysis or failure"
            )
