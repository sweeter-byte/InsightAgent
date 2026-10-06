"""Minimal domain models for evidence-grounded research reports."""

from __future__ import annotations

from dataclasses import dataclass, field


class ReportGenerationError(RuntimeError):
    """Raised when report inputs or model output violate runtime contracts."""


@dataclass(slots=True)
class Claim:
    """One factual conclusion bound to existing Evidence identities."""

    id: str
    task_id: str
    text: str
    evidence_ids: list[str]


@dataclass(frozen=True, slots=True)
class Citation:
    """Display provenance assigned deterministically to one Evidence item."""

    number: int
    evidence_id: str
    label: str
    source: str
    locator: str | None = None


@dataclass(slots=True)
class ReportSection:
    """Claims and the latest grading status for one research task."""

    task_id: str
    title: str
    claims: list[Claim] = field(default_factory=list)
    sufficient: bool = False
    missing_information: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructuredReport:
    """Plan-ordered sections plus the citations they use."""

    objective: str
    sections: list[ReportSection] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
