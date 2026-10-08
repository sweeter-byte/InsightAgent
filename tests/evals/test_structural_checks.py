from __future__ import annotations

from typing import Any

import pytest

from evals import (
    provenance_valid_rate,
    validate_citations,
    validate_claim_bindings,
    validate_evidence_provenance,
    validate_report_structure,
)
from insight_agent.evidence import Evidence
from insight_agent.reporting import (
    Citation,
    Claim,
    ReportSection,
    StructuredReport,
    build_citation,
)
from insight_agent.routing import RetrievalSource


def _evidence(
    evidence_id: str,
    *,
    task_id: str = "T1",
    retrieval_source: RetrievalSource = RetrievalSource.LOCAL,
    origin_id: str | None = None,
    source: str = "notes/research.md",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id=task_id,
        retrieval_source=retrieval_source,
        origin_id=origin_id if origin_id is not None else f"origin-{evidence_id}",
        content=f"Content for {evidence_id}",
        source=source,
        metadata=dict(metadata or {}),
    )


def _report(*claims: Claim, task_id: str = "T1") -> StructuredReport:
    return StructuredReport(
        objective="Objective",
        sections=[
            ReportSection(
                task_id=task_id,
                title="Question",
                claims=list(claims),
                sufficient=True,
            )
        ],
        citations=[],
    )


def test_validate_claim_bindings_accepts_task_scoped_evidence() -> None:
    evidence = _evidence("E1")
    report = _report(
        Claim(id="T1-C1", task_id="T1", text="Fact", evidence_ids=["E1"])
    )

    assert validate_claim_bindings(report, {"E1": evidence}) == ()


def test_validate_claim_bindings_reports_empty_and_duplicate_bindings() -> None:
    evidence = _evidence("E1")
    report = _report(
        Claim(id="T1-C1", task_id="T1", text="No source", evidence_ids=[]),
        Claim(
            id="T1-C2",
            task_id="T1",
            text="Repeated source",
            evidence_ids=["E1", "E1"],
        ),
    )

    assert validate_claim_bindings(report, {"E1": evidence}) == (
        "claim_without_evidence:T1-C1",
        "duplicate_evidence_binding:T1-C2:E1",
    )


def test_validate_claim_bindings_reports_missing_and_cross_task_evidence() -> None:
    other_task = _evidence("E2", task_id="T2")
    report = _report(
        Claim(
            id="T1-C1",
            task_id="T1",
            text="Invalid sources",
            evidence_ids=["missing", "E2"],
        )
    )

    assert validate_claim_bindings(report, {"E2": other_task}) == (
        "missing_evidence:T1-C1:missing",
        "cross_task_binding:T1-C1:E2",
    )


def test_validate_claim_bindings_reports_section_claim_task_mismatch() -> None:
    evidence = _evidence("E2", task_id="T2")
    report = _report(
        Claim(id="T2-C1", task_id="T2", text="Fact", evidence_ids=["E2"]),
        task_id="T1",
    )

    assert validate_claim_bindings(report, {"E2": evidence}) == (
        "section_claim_task_mismatch:T1:T2-C1:T2",
    )


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence("L1"),
        _evidence(
            "W1",
            retrieval_source=RetrievalSource.WEB,
            source="https://example.com/page",
        ),
        _evidence(
            "W2",
            retrieval_source=RetrievalSource.WEB,
            source="https://redirect.example/page",
            metadata={"final_url": "https://example.com/final"},
        ),
        _evidence(
            "V1",
            retrieval_source=RetrievalSource.VISION,
            source="images/chart.png",
        ),
    ],
)
def test_validate_evidence_provenance_accepts_traceable_sources(
    evidence: Evidence,
) -> None:
    assert validate_evidence_provenance(evidence) == ()


@pytest.mark.parametrize(
    ("evidence", "error"),
    [
        (_evidence("", origin_id="origin"), "invalid_provenance:<missing>:id"),
        (
            _evidence("E1", task_id=" "),
            "invalid_provenance:E1:task_id",
        ),
        (
            _evidence("E1", origin_id=" "),
            "invalid_provenance:E1:origin_id",
        ),
        (
            _evidence("E1", source=" "),
            "invalid_provenance:E1:source",
        ),
        (
            _evidence(
                "W1",
                retrieval_source=RetrievalSource.WEB,
                source="not-a-url",
            ),
            "invalid_provenance:W1:web_url",
        ),
        (
            _evidence(
                "W1",
                retrieval_source=RetrievalSource.WEB,
                source="https://example.com/fallback",
                metadata={"final_url": "ftp://example.com/file"},
            ),
            "invalid_provenance:W1:web_url",
        ),
        (
            _evidence("E1", retrieval_source="other"),  # type: ignore[arg-type]
            "invalid_provenance:E1:retrieval_source",
        ),
    ],
)
def test_validate_evidence_provenance_reports_invalid_fields(
    evidence: Evidence,
    error: str,
) -> None:
    assert error in validate_evidence_provenance(evidence)


def test_provenance_valid_rate_counts_evidence_without_errors() -> None:
    valid = _evidence("L1")
    invalid = _evidence(
        "W1",
        retrieval_source=RetrievalSource.WEB,
        source="not-a-url",
    )

    assert provenance_valid_rate([]) is None
    assert provenance_valid_rate([valid, invalid]) == 0.5


def test_validate_citations_accepts_production_built_provenance() -> None:
    evidence = _evidence("E1")
    report = _report(
        Claim(id="T1-C1", task_id="T1", text="Fact", evidence_ids=["E1"])
    )
    report.citations = [build_citation(1, evidence)]

    assert validate_citations(report, {"E1": evidence}) == ()
    assert validate_report_structure(report, {"E1": evidence}) == ()


def test_validate_citations_reports_number_and_identity_conflicts() -> None:
    first = _evidence("E1")
    second = _evidence("E2")
    report = StructuredReport(
        objective="Objective",
        sections=[],
        citations=[
            build_citation(1, first),
            build_citation(1, second),
            build_citation(2, first),
            Citation(0, "unknown", "Unknown", "source"),
        ],
    )

    errors = validate_citations(report, {"E1": first, "E2": second})

    assert "duplicate_citation_number:1:E2" in errors
    assert "duplicate_citation_evidence:E1" in errors
    assert "invalid_citation_number:unknown:0" in errors
    assert "citation_missing_evidence:unknown" in errors
    assert "unused_citation:E1" in errors
    assert "unused_citation:E2" in errors


def test_validate_citations_reports_missing_and_unused_citations() -> None:
    used = _evidence("E1")
    unused = _evidence("E2")
    report = _report(
        Claim(id="T1-C1", task_id="T1", text="Fact", evidence_ids=["E1"])
    )
    report.citations = [build_citation(1, unused)]

    errors = validate_citations(report, {"E1": used, "E2": unused})

    assert "missing_citation:E1" in errors
    assert "unused_citation:E2" in errors


def test_validate_citations_reports_tampered_provenance() -> None:
    evidence = _evidence("E1")
    expected = build_citation(1, evidence)
    report = _report(
        Claim(id="T1-C1", task_id="T1", text="Fact", evidence_ids=["E1"])
    )
    report.citations = [
        Citation(
            number=1,
            evidence_id="E1",
            label="Invented label",
            source=expected.source,
            locator=expected.locator,
        )
    ]

    assert validate_citations(report, {"E1": evidence}) == (
        "citation_provenance_mismatch:E1:label",
    )


def test_validate_report_structure_reports_evidence_mapping_key_mismatch() -> None:
    evidence = _evidence("E1")
    report = StructuredReport(objective="Objective", sections=[], citations=[])

    assert validate_report_structure(report, {"wrong-key": evidence}) == (
        "evidence_key_mismatch:wrong-key:E1",
    )
