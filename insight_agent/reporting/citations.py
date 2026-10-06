"""Deterministic Evidence-to-Citation mapping and report assembly."""

from __future__ import annotations

from pathlib import Path

from insight_agent.evidence import Evidence
from insight_agent.reporting.models import (
    Citation,
    ReportGenerationError,
    ReportSection,
    StructuredReport,
)
from insight_agent.routing import RetrievalSource


class CitationRegistry:
    """Assign stable consecutive numbers on first Evidence use."""

    def __init__(self, evidence_by_id: dict[str, Evidence]) -> None:
        if not isinstance(evidence_by_id, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, Evidence)
            or value.id != key
            for key, value in evidence_by_id.items()
        ):
            raise ReportGenerationError(
                "citation Evidence must be keyed by its Evidence ID"
            )
        self._evidence_by_id = dict(evidence_by_id)
        self._number_by_id: dict[str, int] = {}
        self._citations: list[Citation] = []

    @property
    def citations(self) -> list[Citation]:
        return list(self._citations)

    def number_for(self, evidence_id: str) -> int:
        if evidence_id in self._number_by_id:
            return self._number_by_id[evidence_id]
        evidence = self._evidence_by_id.get(evidence_id)
        if evidence is None:
            raise ReportGenerationError(
                f"Claim references unknown Evidence ID {evidence_id!r}"
            )
        number = len(self._citations) + 1
        self._number_by_id[evidence_id] = number
        self._citations.append(build_citation(number, evidence))
        return number


def build_citation(number: int, evidence: Evidence) -> Citation:
    """Construct display provenance from trusted Evidence fields only."""
    metadata = evidence.metadata
    if evidence.retrieval_source is RetrievalSource.LOCAL:
        label = _filename(metadata.get("filename"), evidence.source)
        page = metadata.get("page")
        locator = (
            f"第 {page} 页"
            if isinstance(page, int) and not isinstance(page, bool) and page > 0
            else None
        )
        source = evidence.source
    elif evidence.retrieval_source is RetrievalSource.WEB:
        source = _non_empty(metadata.get("final_url")) or evidence.source
        label = (
            _non_empty(metadata.get("title"))
            or _non_empty(metadata.get("search_title"))
            or source
        )
        locator = None
    elif evidence.retrieval_source is RetrievalSource.VISION:
        label = _filename(metadata.get("filename"), evidence.source)
        source = evidence.source
        locator = "图片"
    else:
        raise ReportGenerationError("Evidence has an unsupported retrieval source")
    return Citation(number, evidence.id, label, source, locator)


def assemble_report(
    objective: str,
    sections: list[ReportSection],
    evidence_pool: dict[str, list[Evidence]],
) -> StructuredReport:
    """Validate Claim bindings and build citations in first-use order."""
    evidence_by_id: dict[str, Evidence] = {}
    for task_id, items in evidence_pool.items():
        for evidence in items:
            if evidence.task_id != task_id:
                raise ReportGenerationError(
                    "Evidence pool contains an item under the wrong task"
                )
            if evidence.id in evidence_by_id:
                raise ReportGenerationError(
                    f"Evidence ID {evidence.id!r} is duplicated across the pool"
                )
            evidence_by_id[evidence.id] = evidence

    registry = CitationRegistry(evidence_by_id)
    for section in sections:
        for claim in section.claims:
            if claim.task_id != section.task_id:
                raise ReportGenerationError(
                    "Claim task_id does not match its ReportSection"
                )
            if not claim.text.strip() or not claim.evidence_ids:
                raise ReportGenerationError(
                    "every factual Claim must contain text and Evidence IDs"
                )
            if len(claim.evidence_ids) != len(set(claim.evidence_ids)):
                raise ReportGenerationError("Claim contains duplicate Evidence IDs")
            for evidence_id in claim.evidence_ids:
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    raise ReportGenerationError(
                        f"Claim references unknown Evidence ID {evidence_id!r}"
                    )
                if evidence.task_id != section.task_id:
                    raise ReportGenerationError(
                        "Claim references Evidence belonging to another task"
                    )
                registry.number_for(evidence_id)
    return StructuredReport(
        objective=objective,
        sections=list(sections),
        citations=registry.citations,
    )


def _filename(value: object, source: str) -> str:
    return _non_empty(value) or Path(source).name or source


def _non_empty(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
