"""Unit tests for report self-check and controlled repair."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

import insight_agent.self_check.repair as repair_module
from insight_agent.evidence import (
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
)
from insight_agent.evidence.metadata_projection import (
    MAX_METADATA_BYTES_PER_EVIDENCE,
    MAX_METADATA_COLLECTION_ITEMS,
    MAX_METADATA_STRING_CHARS,
    MAX_TOTAL_METADATA_BYTES,
    METADATA_OMISSION_MARKER,
)
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import (
    Claim,
    ReportGenerationError,
    ReportSection,
    StructuredReport,
    assemble_report,
)
from insight_agent.routing import RetrievalSource
from insight_agent.self_check import (
    ReportRepairer,
    ReportSelfChecker,
    SelfCheckError,
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)


class FakeLLM:
    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": messages, "tools": tools})
        content = self.payloads.pop(0)
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _research_inputs() -> tuple[
    ResearchPlan,
    StructuredReport,
    dict[str, list[Evidence]],
    dict[str, list[EvidenceAssessment]],
]:
    task = ResearchTask(id="T1", question="How does method A perform?")
    plan = ResearchPlan(
        objective="Assess method A",
        constraints=["Use only collected Evidence"],
        tasks=[task],
    )
    evidence = Evidence(
        id="E1",
        task_id="T1",
        retrieval_source=RetrievalSource.LOCAL,
        origin_id="origin-E1",
        content="Method A reduced tokens in two experiments.",
        source="paper.md",
        metadata={"filename": "paper.md"},
    )
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id="E1",
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Direct experimental result.",
            )
        ],
        coverage=EvidenceCoverage.PARTIAL,
        sufficient=False,
        missing_information=["Long-term maintenance cost is unknown."],
        reason="One required comparison remains unavailable.",
    )
    section = ReportSection(
        task_id="T1",
        title=task.question,
        claims=[
            Claim(
                id="T1-C1",
                task_id="T1",
                text="Method A reduced tokens in two experiments.",
                evidence_ids=["E1"],
            )
        ],
        sufficient=False,
        missing_information=list(assessment.missing_information),
    )
    pool = {"T1": [evidence]}
    report = assemble_report(plan.objective, [section], pool)
    return plan, report, pool, {"T1": [assessment]}


def _two_task_research_inputs() -> tuple[
    ResearchPlan,
    StructuredReport,
    dict[str, list[Evidence]],
    dict[str, list[EvidenceAssessment]],
]:
    plan, report, pool, assessments = _research_inputs()
    task = ResearchTask(id="T2", question="What remains unknown?")
    plan.tasks.append(task)
    assessment = EvidenceAssessment(
        task_id="T2",
        evidence_judgments=[],
        coverage=EvidenceCoverage.INSUFFICIENT,
        sufficient=False,
        missing_information=["No maintenance data is available."],
        reason="No usable Evidence was collected.",
    )
    pool["T2"] = []
    assessments["T2"] = [assessment]
    report = assemble_report(
        plan.objective,
        [
            *report.sections,
            ReportSection(
                task_id="T2",
                title=task.question,
                claims=[],
                sufficient=False,
                missing_information=list(assessment.missing_information),
            ),
        ],
        pool,
    )
    return plan, report, pool, assessments


def _evidence(
    evidence_id: str,
    *,
    content: str | None = None,
    task_id: str = "T1",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id=task_id,
        retrieval_source=RetrievalSource.LOCAL,
        origin_id=f"origin-{evidence_id}",
        content=content if content is not None else f"Content for {evidence_id}",
        source=f"{evidence_id}.md",
        metadata=(
            metadata
            if metadata is not None
            else {"filename": f"{evidence_id}.md"}
        ),
    )


def _compact_json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_self_check_models_use_finite_vocabulary() -> None:
    issue = SelfCheckIssue(
        code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
        reason="Claim is broader than Evidence.",
        task_id="T1",
        claim_id="T1-C1",
    )
    result = SelfCheckResult(
        status=SelfCheckStatus.REVISE,
        issues=[issue],
        summary="Repair required.",
    )

    assert {item.value for item in SelfCheckStatus} == {"pass", "revise"}
    assert {item.value for item in SelfCheckIssueCode} == {
        "unsupported_claim",
        "citation_mismatch",
        "missing_gap_disclosure",
        "constraint_violation",
        "inconsistent_claims",
    }
    assert result.issues == [issue]


def test_research_state_defaults_self_check_fields() -> None:
    state = ResearchState(query="query")

    assert state.self_check_result is None
    assert state.self_check_rounds == 0


def test_checker_parses_pass_and_builds_deduplicated_evidence_catalog() -> None:
    plan, report, pool, assessments = _research_inputs()
    report.sections[0].claims.append(
        Claim(
            id="T1-C2",
            task_id="T1",
            text="The same experiments measured token use.",
            evidence_ids=["E1"],
        )
    )
    report = assemble_report(plan.objective, report.sections, pool)
    llm = FakeLLM(
        json.dumps(
            {
                "status": "pass",
                "issues": [],
                "summary": (
                    "Claims are supported and recorded gaps are disclosed."
                ),
            }
        )
    )

    result = ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool=pool,
        assessments=assessments,
    )

    assert result == SelfCheckResult(
        status=SelfCheckStatus.PASS,
        issues=[],
        summary="Claims are supported and recorded gaps are disclosed.",
    )
    assert llm.calls[0]["tools"] is None
    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["claim_contexts"] == [
        {
            "task": {"id": "T1", "question": "How does method A perform?"},
            "claim": {
                "id": "T1-C1",
                "text": "Method A reduced tokens in two experiments.",
                "evidence_ids": ["E1"],
            },
            "provided_evidence_ids": ["E1"],
            "unprovided_evidence_ids": [],
            "latest_assessment": {
                "sufficient": False,
                "missing_information": [
                    "Long-term maintenance cost is unknown."
                ],
                "reason": "One required comparison remains unavailable.",
            },
        },
        {
            "task": {"id": "T1", "question": "How does method A perform?"},
            "claim": {
                "id": "T1-C2",
                "text": "The same experiments measured token use.",
                "evidence_ids": ["E1"],
            },
            "provided_evidence_ids": ["E1"],
            "unprovided_evidence_ids": [],
            "latest_assessment": {
                "sufficient": False,
                "missing_information": [
                    "Long-term maintenance cost is unknown."
                ],
                "reason": "One required comparison remains unavailable.",
            },
        },
    ]
    assert payload["evidence_catalog"] == {
        "E1": {
            "evidence_id": "E1",
            "content": "Method A reduced tokens in two experiments.",
            "content_is_excerpt": False,
            "original_content_char_count": 43,
            "retrieval_source": "local",
            "origin_id": "origin-E1",
            "source": "paper.md",
            "metadata": {"filename": "paper.md"},
        }
    }
    serialized = llm.calls[0]["messages"][1]["content"]
    assert serialized.count(
        '"content": "Method A reduced tokens in two experiments."'
    ) == 1
    assert "Do not infer" in llm.calls[0]["messages"][0]["content"]
    assert (
        "Metadata may only assist with source identity"
        in llm.calls[0]["messages"][0]["content"]
    )
    assert (
        "cannot independently\nsupport a Claim"
        in llm.calls[0]["messages"][0]["content"]
    )


def test_checker_bounds_catalog_content_without_mutating_evidence() -> None:
    plan, _, _, assessments = _research_inputs()
    evidence = [
        _evidence(
            f"E{index}",
            content=(
                f"HEAD-{index}:"
                + (str(index) * 7000)
                + f":TAIL-{index}"
            ),
        )
        for index in range(1, 12)
    ]
    original_evidence = deepcopy(evidence)
    claims = [
        Claim(
            id=f"T1-C{index}",
            task_id="T1",
            text=f"Claim {index}",
            evidence_ids=[f"E{index}"],
        )
        for index in range(1, 12)
    ]
    pool = {"T1": [_evidence("UNREFERENCED"), *evidence]}
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=plan.tasks[0].question,
                claims=claims,
                sufficient=False,
                missing_information=list(
                    assessments["T1"][-1].missing_information
                ),
            )
        ],
        pool,
    )
    llm = FakeLLM(
        json.dumps(
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": "E11 was not supplied in the bounded view.",
                        "task_id": "T1",
                        "claim_id": "T1-C11",
                    }
                ],
                "summary": "One Claim could not be fully verified.",
            }
        )
    )

    ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool=pool,
        assessments=assessments,
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    catalog = payload["evidence_catalog"]
    assert list(catalog) == [f"E{index}" for index in range(1, 11)]
    assert "UNREFERENCED" not in catalog
    assert len(catalog) == 10
    assert all(len(item["content"]) <= 6000 for item in catalog.values())
    assert sum(len(item["content"]) for item in catalog.values()) <= 40000
    assert all("CONTENT OMITTED" in item["content"] for item in catalog.values())
    assert catalog["E1"]["content"].startswith("HEAD-1:")
    assert catalog["E1"]["content"].endswith(":TAIL-1")
    assert payload["evidence_view"]["referenced_evidence_count"] == 11
    assert payload["evidence_view"]["provided_evidence_count"] == 10
    assert payload["claim_contexts"][-1]["claim"]["evidence_ids"] == ["E11"]
    assert payload["claim_contexts"][-1]["provided_evidence_ids"] == []
    assert payload["claim_contexts"][-1]["unprovided_evidence_ids"] == ["E11"]
    assert evidence == original_evidence


def test_checker_projects_catalog_metadata_in_evidence_order_without_mutation() -> None:
    plan, _, _, assessments = _research_inputs()
    metadata_items = [
        {"filename": "E1.md", "long": "界" * 2000},
        {
            "filename": "E2.md",
            "deep": {"l1": {"l2": {"l3": {"l4": "hidden"}}}},
        },
        {
            "filename": "E3.md",
            "wide": {
                f"key-{index:02d}": index
                for index in range(MAX_METADATA_COLLECTION_ITEMS + 8)
            },
        },
        {
            "filename": "E4.md",
            "title": "Small source",
            "custom": {"reviewed": True, "tags": ["a", "b"]},
        },
        *[
            {
                "filename": f"E{index}.md",
                "title": f"source-{index}-" + ("x" * 3000),
            }
            for index in range(5, 11)
        ],
    ]
    evidence = [
        _evidence(
            f"E{index}",
            content=f"Body uniquely belonging to E{index}",
            metadata=metadata,
        )
        for index, metadata in enumerate(metadata_items, start=1)
    ]
    original_metadata = deepcopy([item.metadata for item in evidence])
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=plan.tasks[0].question,
                claims=[
                    Claim(
                        id=f"T1-C{index}",
                        task_id="T1",
                        text=f"Claim {index}",
                        evidence_ids=[f"E{index}"],
                    )
                    for index in range(1, 11)
                ],
                sufficient=False,
                missing_information=list(
                    assessments["T1"][-1].missing_information
                ),
            )
        ],
        {"T1": evidence},
    )
    llm = FakeLLM(
        json.dumps(
            {
                "status": "pass",
                "issues": [],
                "summary": "All provided Claims are supported.",
            }
        )
    )

    ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool={"T1": evidence},
        assessments=assessments,
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    catalog = payload["evidence_catalog"]
    projected_metadata = [
        catalog[f"E{index}"]["metadata"] for index in range(1, 11)
    ]
    assert _compact_json_size(metadata_items) > MAX_TOTAL_METADATA_BYTES
    assert all(
        _compact_json_size(metadata) <= MAX_METADATA_BYTES_PER_EVIDENCE
        for metadata in projected_metadata
    )
    assert _compact_json_size(projected_metadata) <= MAX_TOTAL_METADATA_BYTES
    assert len(catalog["E1"]["metadata"]["long"]) <= MAX_METADATA_STRING_CHARS
    assert (
        catalog["E2"]["metadata"]["deep"]["l1"]["l2"]["l3"]
        == METADATA_OMISSION_MARKER
    )
    assert len(catalog["E3"]["metadata"]["wide"]) == (
        MAX_METADATA_COLLECTION_ITEMS
    )
    assert catalog["E4"]["metadata"] == metadata_items[3]
    for index in range(1, 11):
        item = catalog[f"E{index}"]
        assert item["evidence_id"] == f"E{index}"
        assert item["content"] == f"Body uniquely belonging to E{index}"
        assert item["metadata"]["filename"] == f"E{index}.md"
        assert item["origin_id"] == f"origin-E{index}"
        assert item["source"] == f"E{index}.md"
        assert item["retrieval_source"] == "local"
    assert [item.metadata for item in evidence] == original_metadata
    assert payload["evidence_view"]["max_metadata_bytes_per_evidence"] == 2048
    assert payload["evidence_view"]["max_total_metadata_bytes"] == 12000
    assert payload["evidence_view"]["metadata_size_unit"] == (
        "compact JSON UTF-8 bytes"
    )


def test_checker_wraps_metadata_projection_errors_without_calling_llm() -> None:
    plan, report, pool, assessments = _research_inputs()
    pool["T1"][0].metadata["cycle"] = pool["T1"][0].metadata
    report = assemble_report(plan.objective, report.sections, pool)
    llm = FakeLLM()

    with pytest.raises(SelfCheckError, match="JSON-serializable"):
        ReportSelfChecker(llm=llm).check(
            plan=plan,
            report=report,
            evidence_pool=pool,
            assessments=assessments,
        )

    assert llm.calls == []


def test_checker_cannot_pass_when_claim_evidence_is_not_provided() -> None:
    plan, _, _, assessments = _research_inputs()
    evidence = [_evidence(f"E{index}") for index in range(1, 12)]
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=plan.tasks[0].question,
                claims=[
                    Claim(
                        id=f"T1-C{index}",
                        task_id="T1",
                        text=f"Claim {index}",
                        evidence_ids=[f"E{index}"],
                    )
                    for index in range(1, 12)
                ],
                sufficient=False,
                missing_information=list(
                    assessments["T1"][-1].missing_information
                ),
            )
        ],
        {"T1": evidence},
    )
    llm = FakeLLM(
        json.dumps(
            {
                "status": "pass",
                "issues": [],
                "summary": "All Claims are supported.",
            }
        )
    )

    result = ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool={"T1": evidence},
        assessments=assessments,
    )

    assert result.status is SelfCheckStatus.REVISE
    assert result.issues == [
        SelfCheckIssue(
            code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
            reason=(
                "Claim could not be fully verified because referenced Evidence "
                "was not provided in the bounded self-check view: E11."
            ),
            task_id="T1",
            claim_id="T1-C11",
        )
    ]


@pytest.mark.parametrize("code", list(SelfCheckIssueCode))
def test_checker_accepts_each_bounded_issue_code(code: SelfCheckIssueCode) -> None:
    plan, report, pool, assessments = _research_inputs()
    llm = FakeLLM(
        json.dumps(
            {
                "status": "revise",
                "issues": [
                    {
                        "code": code.value,
                        "reason": "The report requires a bounded correction.",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Repair required.",
            }
        )
    )

    result = ReportSelfChecker(llm=llm).check(
        plan=plan,
        report=report,
        evidence_pool=pool,
        assessments=assessments,
    )

    assert result.issues[0].code is code


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": "Unknown task.",
                        "task_id": "T99",
                        "claim_id": None,
                    }
                ],
                "summary": "Invalid identity.",
            },
            "unknown task",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "citation_mismatch",
                        "reason": "Unknown Claim.",
                        "task_id": "T1",
                        "claim_id": "T1-C99",
                    }
                ],
                "summary": "Invalid identity.",
            },
            "unknown Claim",
        ),
        (
            {
                "status": "pass",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": "Issue cannot accompany pass.",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Contradictory result.",
            },
            "pass.*issues",
        ),
        (
            {"status": "revise", "issues": [], "summary": "No issue supplied."},
            "revise.*at least one",
        ),
        (
            {"status": "maybe", "issues": [], "summary": "Ambiguous."},
            "status",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "style_problem",
                        "reason": "Invalid vocabulary.",
                        "task_id": None,
                        "claim_id": None,
                    }
                ],
                "summary": "Invalid issue.",
            },
            "code",
        ),
        (
            {
                "status": "revise",
                "issues": [
                    {
                        "code": "unsupported_claim",
                        "reason": " ",
                        "task_id": "T1",
                        "claim_id": "T1-C1",
                    }
                ],
                "summary": "Missing reason.",
            },
            "reason",
        ),
        (
            {"status": "pass", "issues": [], "summary": " "},
            "summary",
        ),
    ],
)
def test_checker_rejects_invalid_result_contract(
    payload: dict[str, Any],
    message: str,
) -> None:
    plan, report, pool, assessments = _research_inputs()

    with pytest.raises(SelfCheckError, match=message):
        ReportSelfChecker(llm=FakeLLM(json.dumps(payload))).check(
            plan=plan,
            report=report,
            evidence_pool=pool,
            assessments=assessments,
        )


def test_checker_rejects_claim_and_task_mismatch() -> None:
    plan, report, pool, assessments = _two_task_research_inputs()
    payload = {
        "status": "revise",
        "issues": [
            {
                "code": "inconsistent_claims",
                "reason": "Claim is assigned to the wrong task.",
                "task_id": "T2",
                "claim_id": "T1-C1",
            }
        ],
        "summary": "Invalid identity relationship.",
    }

    with pytest.raises(SelfCheckError, match="does not belong"):
        ReportSelfChecker(llm=FakeLLM(json.dumps(payload))).check(
            plan=plan,
            report=report,
            evidence_pool=pool,
            assessments=assessments,
        )


def test_repairer_rebuilds_a_narrowed_report_from_runtime_state() -> None:
    plan, report, pool, assessments = _research_inputs()
    report.sections[0].claims[0].text = (
        "Method A reduces token use in all research scenarios."
    )
    issue = SelfCheckIssue(
        code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
        reason="The Claim overgeneralizes two experiments.",
        task_id="T1",
        claim_id="T1-C1",
    )
    result = SelfCheckResult(
        status=SelfCheckStatus.REVISE,
        issues=[issue],
        summary="Narrow the Claim to the observed experiments.",
    )
    llm = FakeLLM(
        json.dumps(
            {
                "objective": "Assess method A",
                "sections": [
                    {
                        "task_id": "T1",
                        "title": "How does method A perform?",
                        "claims": [
                            {
                                "text": (
                                    "Method A reduced tokens in two experiments."
                                ),
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "sufficient": False,
                        "missing_information": [
                            "Long-term maintenance cost is unknown."
                        ],
                    }
                ],
            }
        )
    )

    repaired = ReportRepairer(llm=llm).repair(
        plan=plan,
        report=report,
        self_check_result=result,
        evidence_pool=pool,
        assessments=assessments,
    )

    assert repaired.sections[0].claims == [
        Claim(
            id="T1-C1",
            task_id="T1",
            text="Method A reduced tokens in two experiments.",
            evidence_ids=["E1"],
        )
    ]
    assert repaired.sections[0].sufficient is False
    assert repaired.sections[0].missing_information == [
        "Long-term maintenance cost is unknown."
    ]
    assert repaired.citations == report.citations
    assert llm.calls[0]["tools"] is None


def test_repairer_sends_a_bounded_deduplicated_issue_first_evidence_view() -> None:
    plan, _, _, _ = _research_inputs()
    evidence = [
        _evidence(
            f"E{index}",
            content=f"HEAD-{index}:" + (str(index % 10) * 7000) + f":TAIL-{index}",
        )
        for index in range(1, 12)
    ]
    original_evidence = deepcopy(evidence)
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id=item.id,
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Usable for repair.",
            )
            for item in evidence
        ],
        coverage=EvidenceCoverage.PARTIAL,
        sufficient=False,
        missing_information=["Long-term maintenance cost is unknown."],
        reason="One required comparison remains unavailable.",
    )
    claims = [
        Claim(
            id=f"T1-C{index}",
            task_id="T1",
            text=f"Claim {index}",
            evidence_ids=[f"E{index}"],
        )
        for index in range(1, 12)
    ]
    claims[1].evidence_ids.append("E1")
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=plan.tasks[0].question,
                claims=claims,
                sufficient=False,
                missing_information=list(assessment.missing_information),
            )
        ],
        {"T1": evidence},
    )
    issue = SelfCheckIssue(
        code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
        reason="The last Claim must be checked first.",
        task_id="T1",
        claim_id="T1-C11",
    )
    llm = FakeLLM(
        json.dumps(
            {
                "objective": plan.objective,
                "sections": [
                    {
                        "task_id": "T1",
                        "title": plan.tasks[0].question,
                        "claims": [
                            {
                                "text": claim.text,
                                "evidence_ids": list(claim.evidence_ids),
                            }
                            for claim in claims
                        ],
                        "sufficient": False,
                        "missing_information": list(
                            assessment.missing_information
                        ),
                    }
                ],
            }
        )
    )

    ReportRepairer(llm=llm).repair(
        plan=plan,
        report=report,
        self_check_result=SelfCheckResult(
            status=SelfCheckStatus.REVISE,
            issues=[issue],
            summary="Repair the last Claim.",
        ),
        evidence_pool={"T1": evidence},
        assessments={"T1": [assessment]},
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    catalog = payload["evidence_catalog"]
    assert list(catalog) == ["E11", *[f"E{index}" for index in range(1, 10)]]
    assert len(catalog) == 10
    assert all(len(item["content"]) <= 6000 for item in catalog.values())
    assert sum(len(item["content"]) for item in catalog.values()) <= 40000
    assert all("CONTENT OMITTED" in item["content"] for item in catalog.values())
    assert catalog["E11"]["content"].startswith("HEAD-11:")
    assert catalog["E11"]["content"].endswith(":TAIL-11")
    assert catalog["E11"]["task_id"] == "T1"
    assert catalog["E11"]["retrieval_source"] == "local"
    assert catalog["E11"]["origin_id"] == "origin-E11"
    assert catalog["E11"]["source"] == "E11.md"
    assert catalog["E11"]["metadata"] == {"filename": "E11.md"}
    assert payload["task_contexts"][0]["provided_evidence_ids"][0] == "E11"
    assert payload["evidence_view"]["provided_evidence_count"] == 10
    serialized = llm.calls[0]["messages"][1]["content"]
    assert serialized.count('"evidence_id": "E1"') == 1
    assert "finite evidence view" in llm.calls[0]["messages"][0]["content"]
    assert evidence == original_evidence


def test_repairer_projects_selected_metadata_in_evidence_order_without_mutation(
) -> None:
    plan, report, _, _ = _research_inputs()
    metadata_items = [
        {"filename": "E1.md", "long": "界" * 2000},
        {
            "filename": "E2.md",
            "deep": {"l1": {"l2": {"l3": {"l4": "hidden"}}}},
        },
        {
            "filename": "E3.md",
            "wide": {
                f"key-{index:02d}": index
                for index in range(MAX_METADATA_COLLECTION_ITEMS + 8)
            },
        },
        {
            "filename": "E4.md",
            "title": "Small source",
            "custom": {"reviewed": True, "tags": ["a", "b"]},
        },
        *[
            {
                "filename": f"E{index}.md",
                "title": f"source-{index}-" + ("x" * 3000),
            }
            for index in range(5, 11)
        ],
    ]
    evidence = [
        _evidence(
            f"E{index}",
            content=f"Body uniquely belonging to E{index}",
            metadata=metadata,
        )
        for index, metadata in enumerate(metadata_items, start=1)
    ]
    original_metadata = deepcopy([item.metadata for item in evidence])
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id=item.id,
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Usable for repair.",
            )
            for item in evidence
        ],
        coverage=EvidenceCoverage.PARTIAL,
        sufficient=False,
        missing_information=["Long-term maintenance cost is unknown."],
        reason="One required comparison remains unavailable.",
    )
    report = assemble_report(plan.objective, report.sections, {"T1": evidence})
    llm = FakeLLM(json.dumps(_repair_payload()))

    ReportRepairer(llm=llm).repair(
        plan=plan,
        report=report,
        self_check_result=_revise_result(),
        evidence_pool={"T1": evidence},
        assessments={"T1": [assessment]},
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    catalog = payload["evidence_catalog"]
    projected_metadata = [
        catalog[f"E{index}"]["metadata"] for index in range(1, 11)
    ]
    assert _compact_json_size(metadata_items) > MAX_TOTAL_METADATA_BYTES
    assert all(
        _compact_json_size(metadata) <= MAX_METADATA_BYTES_PER_EVIDENCE
        for metadata in projected_metadata
    )
    assert _compact_json_size(projected_metadata) <= MAX_TOTAL_METADATA_BYTES
    assert len(catalog["E1"]["metadata"]["long"]) <= MAX_METADATA_STRING_CHARS
    assert (
        catalog["E2"]["metadata"]["deep"]["l1"]["l2"]["l3"]
        == METADATA_OMISSION_MARKER
    )
    assert len(catalog["E3"]["metadata"]["wide"]) == (
        MAX_METADATA_COLLECTION_ITEMS
    )
    assert catalog["E4"]["metadata"] == metadata_items[3]
    for index in range(1, 11):
        item = catalog[f"E{index}"]
        assert item["evidence_id"] == f"E{index}"
        assert item["task_id"] == "T1"
        assert item["content"] == f"Body uniquely belonging to E{index}"
        assert item["metadata"]["filename"] == f"E{index}.md"
        assert item["origin_id"] == f"origin-E{index}"
        assert item["source"] == f"E{index}.md"
        assert item["retrieval_source"] == "local"
    assert [item.metadata for item in evidence] == original_metadata
    assert payload["evidence_view"]["max_metadata_bytes_per_evidence"] == 2048
    assert payload["evidence_view"]["max_total_metadata_bytes"] == 12000
    assert payload["evidence_view"]["metadata_size_unit"] == (
        "compact JSON UTF-8 bytes"
    )
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "Metadata may only assist with source identity" in prompt
    assert "cannot independently support a new or rewritten Claim" in prompt


def test_repairer_wraps_metadata_projection_errors_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, report, pool, assessments = _research_inputs()
    llm = FakeLLM(json.dumps(_repair_payload()))

    def fail_projection(metadata_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise ValueError("projection failed")

    monkeypatch.setattr(
        repair_module,
        "project_metadata_collection",
        fail_projection,
    )

    with pytest.raises(SelfCheckError, match="JSON-serializable"):
        ReportRepairer(llm=llm).repair(
            plan=plan,
            report=report,
            self_check_result=_revise_result(),
            evidence_pool=pool,
            assessments=assessments,
        )

    assert llm.calls == []


def test_repairer_rejects_unprovided_evidence_for_a_modified_claim() -> None:
    plan, report, _, assessments = _research_inputs()
    evidence = [_evidence(f"E{index}") for index in range(1, 12)]
    assessments["T1"][-1].evidence_judgments = [
        EvidenceJudgment(
            evidence_id=item.id,
            relevance=EvidenceRelevance.RELEVANT,
            quality=EvidenceQuality.STRONG,
            reason="Usable for repair.",
        )
        for item in evidence
    ]
    pool = {"T1": evidence}
    report = assemble_report(plan.objective, report.sections, pool)
    payload = _repair_payload()
    payload["sections"][0]["claims"][0] = {
        "text": "A newly rewritten factual Claim.",
        "evidence_ids": ["E11"],
    }

    with pytest.raises(ReportGenerationError, match="provided repair Evidence view"):
        ReportRepairer(llm=FakeLLM(json.dumps(payload))).repair(
            plan=plan,
            report=report,
            self_check_result=_revise_result(),
            evidence_pool=pool,
            assessments=assessments,
        )


def test_repairer_preserves_unchanged_claims_with_unprovided_legal_references() -> None:
    plan, _, _, _ = _research_inputs()
    evidence = [_evidence(f"E{index}") for index in range(1, 12)]
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id=item.id,
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Usable for the report.",
            )
            for item in evidence
        ],
        coverage=EvidenceCoverage.PARTIAL,
        sufficient=False,
        missing_information=["Long-term maintenance cost is unknown."],
        reason="One required comparison remains unavailable.",
    )
    claims = [
        Claim(
            id=f"T1-C{index}",
            task_id="T1",
            text=f"Claim {index}",
            evidence_ids=[f"E{index}"],
        )
        for index in range(1, 12)
    ]
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=plan.tasks[0].question,
                claims=claims,
                sufficient=False,
                missing_information=list(assessment.missing_information),
            )
        ],
        {"T1": evidence},
    )
    payload = {
        "objective": plan.objective,
        "sections": [
            {
                "task_id": "T1",
                "title": plan.tasks[0].question,
                "claims": [
                    {
                        "text": claim.text,
                        "evidence_ids": list(claim.evidence_ids),
                    }
                    for claim in claims
                ],
                "sufficient": False,
                "missing_information": list(assessment.missing_information),
            }
        ],
    }

    repaired = ReportRepairer(llm=FakeLLM(json.dumps(payload))).repair(
        plan=plan,
        report=report,
        self_check_result=_revise_result(claim_id="T1-C1"),
        evidence_pool={"T1": evidence},
        assessments={"T1": [assessment]},
    )

    assert repaired.sections[0].claims[-1].evidence_ids == ["E11"]


def _repair_payload() -> dict[str, Any]:
    return {
        "objective": "Assess method A",
        "sections": [
            {
                "task_id": "T1",
                "title": "How does method A perform?",
                "claims": [
                    {
                        "text": "Method A reduced tokens in two experiments.",
                        "evidence_ids": ["E1"],
                    }
                ],
                "sufficient": False,
                "missing_information": [
                    "Long-term maintenance cost is unknown."
                ],
            }
        ],
    }


def _revise_result(
    *,
    task_id: str | None = "T1",
    claim_id: str | None = "T1-C1",
) -> SelfCheckResult:
    return SelfCheckResult(
        status=SelfCheckStatus.REVISE,
        issues=[
            SelfCheckIssue(
                code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
                reason="Claim must be narrowed.",
                task_id=task_id,
                claim_id=claim_id,
            )
        ],
        summary="Repair required.",
    )


def test_repairer_rejects_unvalidated_issue_identity() -> None:
    plan, report, pool, assessments = _research_inputs()

    with pytest.raises(SelfCheckError, match="unknown Claim"):
        ReportRepairer(llm=FakeLLM(json.dumps(_repair_payload()))).repair(
            plan=plan,
            report=report,
            self_check_result=_revise_result(claim_id="T1-C99"),
            evidence_pool=pool,
            assessments=assessments,
        )


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("unknown_evidence", ReportGenerationError, "allowed candidate"),
        ("claim_extra_source", SelfCheckError, "field set"),
        ("changed_objective", SelfCheckError, "objective"),
        ("changed_sufficiency", SelfCheckError, "sufficiency"),
        ("changed_gaps", SelfCheckError, "missing_information"),
    ],
)
def test_repairer_rejects_changes_outside_controlled_boundary(
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    plan, report, pool, assessments = _research_inputs()
    payload = _repair_payload()
    section = payload["sections"][0]
    claim = section["claims"][0]
    if case == "unknown_evidence":
        claim["evidence_ids"] = ["invented"]
    elif case == "claim_extra_source":
        claim["source"] = "https://invented.invalid"
    elif case == "changed_objective":
        payload["objective"] = "A new objective"
    elif case == "changed_sufficiency":
        section["sufficient"] = True
    elif case == "changed_gaps":
        section["missing_information"] = []

    with pytest.raises(error, match=message):
        ReportRepairer(llm=FakeLLM(json.dumps(payload))).repair(
            plan=plan,
            report=report,
            self_check_result=_revise_result(),
            evidence_pool=pool,
            assessments=assessments,
        )


def test_repairer_rejects_pass_result_before_calling_llm() -> None:
    plan, report, pool, assessments = _research_inputs()
    llm = FakeLLM(json.dumps(_repair_payload()))

    with pytest.raises(SelfCheckError, match="revise"):
        ReportRepairer(llm=llm).repair(
            plan=plan,
            report=report,
            self_check_result=SelfCheckResult(
                status=SelfCheckStatus.PASS,
                issues=[],
                summary="No repair required.",
            ),
            evidence_pool=pool,
            assessments=assessments,
        )

    assert llm.calls == []
