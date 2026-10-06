"""Unified candidate Evidence produced from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from insight_agent.routing import RetrievalSource


class EvidenceRelevance(str, Enum):
    """How directly one Evidence item addresses the current task."""

    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class EvidenceQuality(str, Enum):
    """How usable and traceable one Evidence item is as research material."""

    STRONG = "strong"
    USABLE = "usable"
    WEAK = "weak"


class EvidenceCoverage(str, Enum):
    """How fully the complete Evidence pool covers the current task."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class EvidenceGradingError(RuntimeError):
    """Raised when an Evidence assessment cannot be parsed or validated."""


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


@dataclass(slots=True)
class EvidenceJudgment:
    """One grader judgment linked to an existing Evidence identity."""

    evidence_id: str
    relevance: EvidenceRelevance
    quality: EvidenceQuality
    reason: str


@dataclass(slots=True)
class EvidenceAssessment:
    """Task-level assessment of an accumulated Evidence pool."""

    task_id: str
    evidence_judgments: list[EvidenceJudgment]
    coverage: EvidenceCoverage
    sufficient: bool
    missing_information: list[str] = field(default_factory=list)
    reason: str = ""
