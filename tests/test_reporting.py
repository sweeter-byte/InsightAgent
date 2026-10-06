"""Offline tests for Claim synthesis, citation assembly, and rendering."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.evidence import (
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
)
from insight_agent.planning import ResearchTask
from insight_agent.reporting import (
    REPORT_GENERATOR_SYSTEM_PROMPT,
    Citation,
    CitationRegistry,
    Claim,
    MarkdownReportRenderer,
    ReportGenerationError,
    ReportGenerator,
    ReportSection,
    StructuredReport,
    assemble_report,
    select_report_candidates,
)
from insight_agent.routing import RetrievalSource


class FakeLLM:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": messages, "tools": tools})
        message = SimpleNamespace(content=self.payload, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _task(task_id: str = "T1") -> ResearchTask:
    return ResearchTask(id=task_id, question=f"Question for {task_id}")


def _evidence(
    evidence_id: str,
    *,
    task_id: str = "T1",
    source_type: RetrievalSource = RetrievalSource.LOCAL,
    source: str = "notes.md",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id=task_id,
        retrieval_source=source_type,
        origin_id=f"origin-{evidence_id}",
        content=f"Content for {evidence_id}",
        source=source,
        metadata=dict(metadata or {}),
    )


def _judgment(
    evidence_id: str,
    *,
    relevance: EvidenceRelevance = EvidenceRelevance.RELEVANT,
    quality: EvidenceQuality = EvidenceQuality.STRONG,
) -> EvidenceJudgment:
    return EvidenceJudgment(
        evidence_id=evidence_id,
        relevance=relevance,
        quality=quality,
        reason="Test judgment.",
    )


def _assessment(
    *judgments: EvidenceJudgment,
    task_id: str = "T1",
    sufficient: bool = True,
    missing_information: list[str] | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        task_id=task_id,
        evidence_judgments=list(judgments),
        coverage=(
            EvidenceCoverage.COMPLETE
            if sufficient
            else EvidenceCoverage.PARTIAL
        ),
        sufficient=sufficient,
        missing_information=list(missing_information or []),
        reason="Assessment reason.",
    )


def test_reporting_models_construct_with_minimal_fields() -> None:
    claim = Claim(id="T1-C1", task_id="T1", text="A fact.", evidence_ids=["E1"])
    citation = Citation(
        number=1,
        evidence_id="E1",
        label="paper.pdf",
        source="/private/paper.pdf",
        locator="第 7 页",
    )
    section = ReportSection(
        task_id="T1",
        title="Question",
        claims=[claim],
        sufficient=True,
        missing_information=[],
    )
    report = StructuredReport(
        objective="Objective",
        sections=[section],
        citations=[citation],
    )

    assert report.sections[0].claims == [claim]
    assert report.citations == [citation]


def test_candidate_selection_uses_only_relevant_or_partial_non_weak_evidence() -> None:
    pool = [_evidence(f"E{index}") for index in range(1, 5)]
    assessment = _assessment(
        _judgment("E1"),
        _judgment(
            "E2",
            relevance=EvidenceRelevance.PARTIAL,
            quality=EvidenceQuality.USABLE,
        ),
        _judgment("E3", relevance=EvidenceRelevance.IRRELEVANT),
        _judgment("E4", quality=EvidenceQuality.WEAK),
    )

    candidates = select_report_candidates(_task(), pool, assessment)

    assert [item.id for item in candidates] == ["E1", "E2"]
    assert [item.id for item in pool] == ["E1", "E2", "E3", "E4"]


@pytest.mark.parametrize(
    ("pool", "assessment", "message"),
    [
        (
            [_evidence("E1")],
            _assessment(_judgment("missing")),
            "unknown Evidence ID",
        ),
        (
            [_evidence("E1", task_id="T2")],
            _assessment(_judgment("E1")),
            "current task",
        ),
    ],
)
def test_candidate_selection_rejects_inconsistent_assessment_or_pool(
    pool: list[Evidence],
    assessment: EvidenceAssessment,
    message: str,
) -> None:
    with pytest.raises(ReportGenerationError, match=message):
        select_report_candidates(_task(), pool, assessment)


def test_candidate_selection_rejects_invalid_judgment_vocabulary() -> None:
    judgment = _judgment("E1")
    judgment.relevance = "related"  # type: ignore[assignment]

    with pytest.raises(ReportGenerationError, match="relevance"):
        select_report_candidates(
            _task(),
            [_evidence("E1")],
            _assessment(judgment),
        )


def test_report_generator_parses_claims_and_assigns_deterministic_runtime_ids() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "claims": [
                    {"text": "First fact.", "evidence_ids": ["E1", "E2"]},
                    {"text": "Second fact.", "evidence_ids": ["E2"]},
                ]
            }
        )
    )
    generator = ReportGenerator(llm=llm)  # type: ignore[arg-type]
    evidence = [_evidence("E1"), _evidence("E2")]
    assessment = _assessment(_judgment("E1"), _judgment("E2"))

    claims = generator.generate_section(
        task=_task(),
        objective="Objective",
        constraints=["Constraint"],
        evidence=evidence,
        assessment=assessment,
    )

    assert claims == [
        Claim("T1-C1", "T1", "First fact.", ["E1", "E2"]),
        Claim("T1-C2", "T1", "Second fact.", ["E2"]),
    ]
    assert llm.calls[0]["tools"] is None
    assert REPORT_GENERATOR_SYSTEM_PROMPT in llm.calls[0]["messages"][0]["content"]
    assert "citation_number" not in json.loads(
        llm.calls[0]["messages"][1]["content"]
    )


def test_report_generator_rejects_evidence_excluded_by_assessment_before_llm() -> None:
    evidence = _evidence("E1")
    llm = FakeLLM(
        json.dumps({"claims": [{"text": "Fact.", "evidence_ids": ["E1"]}]})
    )
    generator = ReportGenerator(llm=llm)  # type: ignore[arg-type]

    with pytest.raises(ReportGenerationError, match="not allowed"):
        generator.generate_section(
            task=_task(),
            objective="Objective",
            constraints=[],
            evidence=[evidence],
            assessment=_assessment(
                _judgment("E1", quality=EvidenceQuality.WEAK)
            ),
        )

    assert llm.calls == []


@pytest.mark.parametrize(
    ("claim", "message"),
    [
        ({"text": "Fact.", "evidence_ids": ["unknown"]}, "allowed"),
        ({"text": "Fact.", "evidence_ids": ["T2-E1"]}, "allowed"),
        ({"text": "Fact.", "evidence_ids": ["E-weak"]}, "allowed"),
        ({"text": "Fact.", "evidence_ids": []}, "must not be empty"),
        ({"text": "  ", "evidence_ids": ["E1"]}, "text"),
        ({"text": "Fact.", "evidence_ids": ["E1", "E1"]}, "duplicate"),
        ({"text": "Fact [1].", "evidence_ids": ["E1"]}, "citation marker"),
        (
            {"text": "See https://invented.invalid.", "evidence_ids": ["E1"]},
            "URL",
        ),
    ],
)
def test_report_generator_rejects_invalid_claim_bindings(
    claim: dict[str, Any],
    message: str,
) -> None:
    llm = FakeLLM(json.dumps({"claims": [claim]}))
    generator = ReportGenerator(llm=llm)  # type: ignore[arg-type]

    with pytest.raises(ReportGenerationError, match=message):
        generator.generate_section(
            task=_task(),
            objective="Objective",
            constraints=[],
            evidence=[_evidence("E1")],
            assessment=_assessment(_judgment("E1")),
        )


@pytest.mark.parametrize("extra_field", ["citation_number", "url", "page", "source"])
def test_report_generator_rejects_model_generated_provenance(extra_field: str) -> None:
    raw_claim: dict[str, Any] = {"text": "Fact.", "evidence_ids": ["E1"]}
    raw_claim[extra_field] = "invented"
    generator = ReportGenerator(  # type: ignore[arg-type]
        llm=FakeLLM(json.dumps({"claims": [raw_claim]}))
    )

    with pytest.raises(ReportGenerationError, match="unexpected field"):
        generator.generate_section(
            task=_task(),
            objective="Objective",
            constraints=[],
            evidence=[_evidence("E1")],
            assessment=_assessment(_judgment("E1")),
        )


def test_citations_come_from_local_web_and_vision_provenance() -> None:
    evidence = {
        "L1": _evidence(
            "L1",
            source="/private/uploads/paper.pdf",
            metadata={"filename": "paper.pdf", "page": 7},
        ),
        "W1": _evidence(
            "W1",
            source_type=RetrievalSource.WEB,
            source="https://redirect.invalid/page",
            metadata={
                "title": "Web title",
                "final_url": "https://example.com/final",
            },
        ),
        "V1": _evidence(
            "V1",
            source_type=RetrievalSource.VISION,
            source="/private/uploads/architecture.png",
            metadata={"filename": "architecture.png"},
        ),
    }
    registry = CitationRegistry(evidence)

    assert registry.number_for("L1") == 1
    assert registry.number_for("W1") == 2
    assert registry.number_for("V1") == 3
    assert registry.citations == [
        Citation(1, "L1", "paper.pdf", "/private/uploads/paper.pdf", "第 7 页"),
        Citation(
            2,
            "W1",
            "Web title",
            "https://example.com/final",
            None,
        ),
        Citation(
            3,
            "V1",
            "architecture.png",
            "/private/uploads/architecture.png",
            "图片",
        ),
    ]


def test_assembly_reuses_numbers_and_renderer_is_deterministic() -> None:
    pool = {
        "T1": [
            _evidence("E1", metadata={"filename": "one.md"}),
            _evidence(
                "E2",
                source_type=RetrievalSource.WEB,
                source="https://example.com/two",
                metadata={"title": "Two", "final_url": "https://example.com/two"},
            ),
            _evidence(
                "E3",
                source_type=RetrievalSource.VISION,
                source="three.png",
                metadata={"filename": "three.png"},
            ),
        ]
    }
    sections = [
        ReportSection(
            task_id="T1",
            title="Question",
            claims=[
                Claim("T1-C1", "T1", "Claim one.", ["E1", "E2"]),
                Claim("T1-C2", "T1", "Claim two.", ["E2", "E3"]),
            ],
            sufficient=False,
            missing_information=["Missing limitation."],
        )
    ]

    report = assemble_report("Objective", sections, pool)
    renderer = MarkdownReportRenderer()
    first = renderer.render(report)
    second = renderer.render(report)

    assert [item.evidence_id for item in report.citations] == ["E1", "E2", "E3"]
    assert "Claim one. [1][2]" in first
    assert "Claim two. [2][3]" in first
    assert "### 证据缺口" in first
    assert "Missing limitation." in first
    assert "[1] one.md" in first
    assert "[2] Two，https://example.com/two" in first
    assert "[3] three.png（图片）" in first
    assert "/private/" not in first
    assert first == second
