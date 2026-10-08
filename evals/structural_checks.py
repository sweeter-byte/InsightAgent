"""Deterministic checks over production Evidence and report structures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from urllib.parse import urlparse

from insight_agent.evidence import Evidence
from insight_agent.reporting import (
    ReportGenerationError,
    StructuredReport,
    build_citation,
)
from insight_agent.routing import RetrievalSource


def validate_claim_bindings(
    report: StructuredReport,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[str, ...]:
    """Return structural errors in Claim-to-Evidence relationships."""
    errors: list[str] = []
    for section in report.sections:
        for claim in section.claims:
            if claim.task_id != section.task_id:
                errors.append(
                    "section_claim_task_mismatch:"
                    f"{section.task_id}:{claim.id}:{claim.task_id}"
                )
            if not claim.evidence_ids:
                errors.append(f"claim_without_evidence:{claim.id}")
                continue

            seen: set[str] = set()
            for evidence_id in claim.evidence_ids:
                if evidence_id in seen:
                    errors.append(
                        f"duplicate_evidence_binding:{claim.id}:{evidence_id}"
                    )
                    continue
                seen.add(evidence_id)
                evidence = evidence_by_id.get(evidence_id)
                if evidence is None:
                    errors.append(f"missing_evidence:{claim.id}:{evidence_id}")
                elif evidence.task_id != claim.task_id:
                    errors.append(
                        f"cross_task_binding:{claim.id}:{evidence_id}"
                    )
    return tuple(errors)


def validate_evidence_provenance(evidence: Evidence) -> tuple[str, ...]:
    """Return deterministic traceability errors for one production Evidence."""
    evidence_key = (
        evidence.id.strip()
        if isinstance(evidence.id, str) and evidence.id.strip()
        else "<missing>"
    )
    errors: list[str] = []
    for field_name in ("id", "task_id", "origin_id", "source"):
        value = getattr(evidence, field_name, None)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"invalid_provenance:{evidence_key}:{field_name}")

    if not isinstance(evidence.retrieval_source, RetrievalSource):
        errors.append(f"invalid_provenance:{evidence_key}:retrieval_source")
        return tuple(errors)

    if evidence.retrieval_source is RetrievalSource.WEB:
        final_url = evidence.metadata.get("final_url")
        url = (
            final_url.strip()
            if isinstance(final_url, str) and final_url.strip()
            else evidence.source.strip()
            if isinstance(evidence.source, str)
            else ""
        )
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append(f"invalid_provenance:{evidence_key}:web_url")

    return tuple(errors)


def provenance_valid_rate(evidence: Iterable[Evidence]) -> float | None:
    """Return the fraction of Evidence items with valid provenance."""
    items = tuple(evidence)
    if not items:
        return None
    valid = sum(not validate_evidence_provenance(item) for item in items)
    return valid / len(items)


def validate_citations(
    report: StructuredReport,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[str, ...]:
    """Check Citation identity and production-derived provenance fields."""
    errors: list[str] = []
    bound_ids = {
        evidence_id
        for section in report.sections
        for claim in section.claims
        for evidence_id in claim.evidence_ids
    }
    seen_numbers: set[int] = set()
    seen_evidence_ids: set[str] = set()
    cited_ids: set[str] = set()

    for citation in report.citations:
        if (
            isinstance(citation.number, bool)
            or not isinstance(citation.number, int)
            or citation.number <= 0
        ):
            errors.append(
                f"invalid_citation_number:{citation.evidence_id}:{citation.number}"
            )
        elif citation.number in seen_numbers:
            errors.append(
                f"duplicate_citation_number:{citation.number}:"
                f"{citation.evidence_id}"
            )
        seen_numbers.add(citation.number)

        if citation.evidence_id in seen_evidence_ids:
            errors.append(f"duplicate_citation_evidence:{citation.evidence_id}")
        seen_evidence_ids.add(citation.evidence_id)
        cited_ids.add(citation.evidence_id)

        evidence = evidence_by_id.get(citation.evidence_id)
        if evidence is None:
            errors.append(f"citation_missing_evidence:{citation.evidence_id}")
            continue
        try:
            expected = build_citation(citation.number, evidence)
        except ReportGenerationError:
            errors.append(
                f"citation_provenance_unbuildable:{citation.evidence_id}"
            )
            continue
        for field_name in ("label", "source", "locator"):
            if getattr(citation, field_name) != getattr(expected, field_name):
                errors.append(
                    "citation_provenance_mismatch:"
                    f"{citation.evidence_id}:{field_name}"
                )

    for evidence_id in sorted(bound_ids - cited_ids):
        errors.append(f"missing_citation:{evidence_id}")
    for evidence_id in sorted(cited_ids - bound_ids):
        errors.append(f"unused_citation:{evidence_id}")
    return tuple(errors)


def validate_report_structure(
    report: StructuredReport,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[str, ...]:
    """Run every deterministic report, binding, and provenance check."""
    errors = list(validate_claim_bindings(report, evidence_by_id))
    errors.extend(validate_citations(report, evidence_by_id))
    for key, evidence in evidence_by_id.items():
        if key != evidence.id:
            errors.append(f"evidence_key_mismatch:{key}:{evidence.id}")
        errors.extend(validate_evidence_provenance(evidence))
    return tuple(errors)
