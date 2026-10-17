"""Tests for validated, retrieval-independent Evidence grading."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.evidence import (
    EVIDENCE_GRADER_SYSTEM_PROMPT,
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceGrader,
    EvidenceGradingError,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
)
from insight_agent.evidence.grader import (
    MAX_METADATA_BYTES_PER_EVIDENCE,
    MAX_TOTAL_METADATA_BYTES,
    METADATA_PRIORITY_KEYS,
)
from insight_agent.planning import ResearchTask
from insight_agent.routing import RetrievalSource


class FakeLLM:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append(
            {"messages": [dict(message) for message in messages], "tools": tools}
        )
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedLLM:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append(
            {"messages": [dict(message) for message in messages], "tools": tools}
        )
        if not self.contents:
            raise AssertionError("ScriptedLLM ran out of responses")
        message = SimpleNamespace(content=self.contents.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _task() -> ResearchTask:
    return ResearchTask(id="T1", question="Compare memory mechanisms.")


def _evidence(
    evidence_id: str = "ev-1",
    *,
    content: str = "Vector memory uses embedding similarity.",
    origin_id: str = "chunk-1",
    source: str = "notes.md",
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="T1",
        retrieval_source=RetrievalSource.LOCAL,
        origin_id=origin_id,
        content=content,
        source=source,
        metadata=metadata
        or {"chunk_id": "chunk-1", "retrieval_score": 0.91},
    )


def _valid_payload(*evidence_ids: str) -> dict[str, Any]:
    return {
        "task_id": "T1",
        "evidence_judgments": [
            {
                "evidence_id": evidence_id,
                "relevance": "relevant",
                "quality": "strong",
                "reason": "Direct support with provenance.",
            }
            for evidence_id in evidence_ids
        ],
        "coverage": "complete",
        "sufficient": True,
        "missing_information": [],
        "reason": "All requested dimensions are covered.",
    }


def _grade_payload(
    payload: Any,
    *,
    evidence: list[Evidence] | None = None,
) -> EvidenceAssessment:
    grader = EvidenceGrader(llm=FakeLLM(json.dumps(payload)))  # type: ignore[arg-type]
    return grader.grade(
        task=_task(),
        objective="Compare memory systems",
        constraints=["Cover trade-offs"],
        evidence=evidence or [_evidence()],
    )


def test_grading_models_use_the_closed_vocabulary() -> None:
    judgment = EvidenceJudgment(
        evidence_id="ev-1",
        relevance=EvidenceRelevance.RELEVANT,
        quality=EvidenceQuality.STRONG,
        reason="Direct and traceable support.",
    )
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[judgment],
        coverage=EvidenceCoverage.COMPLETE,
        sufficient=True,
        missing_information=[],
        reason="The task is covered.",
    )

    assert [item.value for item in EvidenceRelevance] == [
        "relevant",
        "partial",
        "irrelevant",
    ]
    assert [item.value for item in EvidenceQuality] == ["strong", "usable", "weak"]
    assert [item.value for item in EvidenceCoverage] == [
        "complete",
        "partial",
        "insufficient",
    ]
    assert assessment.evidence_judgments == [judgment]
    assert issubclass(EvidenceGradingError, RuntimeError)


def test_assessment_defaults_to_no_missing_information_or_reason() -> None:
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[],
        coverage=EvidenceCoverage.COMPLETE,
        sufficient=True,
    )

    assert assessment.missing_information == []
    assert assessment.reason == ""


def test_grader_parses_a_valid_evidence_assessment() -> None:
    evidence = _evidence()
    llm = FakeLLM(
        json.dumps(
            {
                "task_id": "T1",
                "evidence_judgments": [
                    {
                        "evidence_id": evidence.id,
                        "relevance": "relevant",
                        "quality": "strong",
                        "reason": "Direct support with provenance.",
                    }
                ],
                "coverage": "complete",
                "sufficient": True,
                "missing_information": [],
                "reason": "All requested dimensions are covered.",
            }
        )
    )
    grader = EvidenceGrader(llm=llm)  # type: ignore[arg-type]

    assessment = grader.grade(
        task=_task(),
        objective="Compare memory systems",
        constraints=["Cover trade-offs"],
        evidence=[evidence],
    )

    assert assessment == EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id="ev-1",
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Direct support with provenance.",
            )
        ],
        coverage=EvidenceCoverage.COMPLETE,
        sufficient=True,
        missing_information=[],
        reason="All requested dimensions are covered.",
    )
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None


def test_grader_sends_a_bounded_excerpt_without_mutating_original_evidence() -> None:
    original_content = "HEAD:" + ("a" * 6990) + ":TAIL"
    evidence = _evidence(
        content=original_content,
        origin_id="https://example.test/article",
        source="https://example.test/article?ref=search",
        metadata={"title": "Traceable source", "search_rank": 1},
    )
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[evidence],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    view = payload["evidence"][0]
    assert len(view["content"]) <= 6000
    assert view["content"].startswith("HEAD:")
    assert view["content"].endswith(":TAIL")
    assert "CONTENT OMITTED" in view["content"]
    assert view["evidence_id"] == evidence.id
    assert view["origin_id"] == evidence.origin_id
    assert view["source"] == evidence.source
    assert view["metadata"] == evidence.metadata
    assert evidence.content == original_content


def test_grader_bounds_total_body_size_and_number_of_evidence_items() -> None:
    evidence = [
        _evidence(f"ev-{index}", content=str(index) * 7000)
        for index in range(1, 13)
    ]
    provided_ids = [f"ev-{index}" for index in range(1, 11)]
    llm = FakeLLM(json.dumps(_valid_payload(*provided_ids)))

    assessment = EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=evidence,
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    views = payload["evidence"]
    assert [item["evidence_id"] for item in views] == provided_ids
    assert len(views) == 10
    assert all(len(item["content"]) <= 6000 for item in views)
    assert sum(len(item["content"]) for item in views) <= 40000
    assert [item.evidence_id for item in assessment.evidence_judgments] == provided_ids


def test_grader_keeps_all_content_when_the_view_is_within_limits() -> None:
    evidence = [
        _evidence("ev-1", content="first complete body"),
        _evidence("ev-2", content="second complete body"),
    ]
    llm = FakeLLM(json.dumps(_valid_payload("ev-1", "ev-2")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=evidence,
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert [item["content"] for item in payload["evidence"]] == [
        "first complete body",
        "second complete body",
    ]


def _serialized_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_grader_projects_oversized_metadata_without_mutating_evidence() -> None:
    original_metadata = {
        **{f"auxiliary-{index:02d}": index for index in range(30)},
        "title": "Traceable source",
        "final_url": "https://example.test/final",
        "search_query": "memory mechanisms",
        "search_rank": 1,
        "search_score": 0.93,
        "search_snippet": "x" * (MAX_METADATA_BYTES_PER_EVIDENCE * 4),
    }
    evidence = _evidence(
        origin_id="https://example.test/final",
        source="https://example.test/original",
        metadata=original_metadata,
    )
    before = deepcopy(evidence.metadata)
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    assessment = EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[evidence],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    view = payload["evidence"][0]
    assert _serialized_size(view["metadata"]) <= MAX_METADATA_BYTES_PER_EVIDENCE
    assert view["metadata"]["title"] == original_metadata["title"]
    assert view["metadata"]["final_url"] == original_metadata["final_url"]
    assert len(view["metadata"]["search_snippet"]) < len(
        original_metadata["search_snippet"]
    )
    assert view["evidence_id"] == evidence.id
    assert view["retrieval_source"] == evidence.retrieval_source.value
    assert view["origin_id"] == evidence.origin_id
    assert view["source"] == evidence.source
    assert [item.evidence_id for item in assessment.evidence_judgments] == ["ev-1"]
    assert evidence.metadata == before


def test_grader_bounds_total_serialized_metadata_across_evidence() -> None:
    evidence = [
        _evidence(
            f"ev-{index}",
            metadata={
                "title": f"Source {index}",
                "search_rank": index,
                "payload_a": str(index) * MAX_METADATA_BYTES_PER_EVIDENCE,
                "payload_b": str(index) * MAX_METADATA_BYTES_PER_EVIDENCE,
            },
        )
        for index in range(1, 11)
    ]
    llm = FakeLLM(
        json.dumps(_valid_payload(*(item.id for item in evidence)))
    )

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=evidence,
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    projected = [item["metadata"] for item in payload["evidence"]]
    assert payload["evidence_view"]["max_metadata_bytes_per_evidence"] == (
        MAX_METADATA_BYTES_PER_EVIDENCE
    )
    assert payload["evidence_view"]["max_total_metadata_bytes"] == (
        MAX_TOTAL_METADATA_BYTES
    )
    assert payload["evidence_view"]["metadata_size_unit"] == (
        "compact JSON UTF-8 bytes"
    )
    assert all(
        _serialized_size(metadata) <= MAX_METADATA_BYTES_PER_EVIDENCE
        for metadata in projected
    )
    assert _serialized_size(projected) <= MAX_TOTAL_METADATA_BYTES


def test_grader_nested_metadata_cannot_bypass_serialized_budget() -> None:
    metadata = {
        "title": "Nested source",
        "nested": {
            f"section-{index}": [
                {"text": "多" * MAX_METADATA_BYTES_PER_EVIDENCE}
                for _ in range(20)
            ]
            for index in range(20)
        },
    }
    evidence = _evidence(metadata=metadata)
    before = deepcopy(metadata)
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[evidence],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    projected = payload["evidence"][0]["metadata"]
    assert _serialized_size(projected) <= MAX_METADATA_BYTES_PER_EVIDENCE
    assert projected["title"] == "Nested source"
    assert evidence.metadata == before


def test_grader_bounds_deep_metadata_before_serialization() -> None:
    metadata: dict[str, Any] = {}
    cursor = metadata
    for _ in range(1100):
        child: dict[str, Any] = {}
        cursor["level"] = child
        cursor = child
    cursor["leaf"] = "unchanged"
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[_evidence(metadata=metadata)],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    projected = payload["evidence"][0]["metadata"]
    assert _serialized_size(projected) <= MAX_METADATA_BYTES_PER_EVIDENCE
    original_cursor = metadata
    for _ in range(1100):
        original_cursor = original_cursor["level"]
    assert original_cursor == {"leaf": "unchanged"}


def test_grader_does_not_scan_an_entire_wide_metadata_dictionary() -> None:
    class BoundedIterationDict(dict[str, Any]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            for index, key in enumerate(super().__iter__()):
                if index >= MAX_METADATA_BYTES_PER_EVIDENCE:
                    raise AssertionError("metadata projection scanned too many keys")
                yield key

    metadata = BoundedIterationDict(
        [
            *[(key, f"value-{key}") for key in METADATA_PRIORITY_KEYS],
            *[
                (f"field-{index:05d}", index)
                for index in range(MAX_METADATA_BYTES_PER_EVIDENCE * 2)
            ],
        ]
    )
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[_evidence(metadata=metadata)],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    projected = payload["evidence"][0]["metadata"]
    assert list(projected) == list(METADATA_PRIORITY_KEYS)
    assert _serialized_size(projected) <= MAX_METADATA_BYTES_PER_EVIDENCE


def test_grader_prioritizes_stable_provenance_over_long_descriptions() -> None:
    long_text = "x" * MAX_METADATA_BYTES_PER_EVIDENCE
    metadata = {
        "title": long_text,
        "search_query": long_text,
        "search_title": long_text,
        "search_snippet": long_text,
        "final_url": "https://example.test/final",
        "original_url": "https://example.test/original",
        "filename": "paper.pdf",
        "page": 7,
        "retrieval_score": 0.91,
        "chunk_id": "chunk-7",
        "document_id": "document-1",
        "source_type": "pdf",
        "chunk_index": 6,
        "start_char": 1200,
        "end_char": 2400,
        "search_rank": 1,
        "search_score": 0.89,
    }
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[_evidence(metadata=metadata)],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    projected = payload["evidence"][0]["metadata"]
    for key in (
        "final_url",
        "original_url",
        "filename",
        "page",
        "retrieval_score",
        "chunk_id",
        "document_id",
        "source_type",
        "chunk_index",
        "start_char",
        "end_char",
        "search_rank",
        "search_score",
    ):
        assert projected[key] == metadata[key]


def test_grader_preserves_small_metadata_as_a_complete_value() -> None:
    metadata = {
        "title": "Small source",
        "final_url": "https://example.test/article",
        "content_type": "text/html; charset=utf-8",
        "search_query": "memory mechanisms",
        "search_rank": 2,
        "search_score": 0.82,
        "custom": {"tags": ["memory", "retrieval"], "reviewed": True},
    }
    evidence = _evidence(metadata=metadata)
    llm = FakeLLM(json.dumps(_valid_payload("ev-1")))

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[evidence],
    )

    payload = json.loads(llm.calls[0]["messages"][1]["content"])
    assert payload["evidence"][0]["metadata"] == metadata


def test_grader_input_remains_bounded_when_a_second_round_accumulates_evidence() -> None:
    first_round = [
        _evidence(f"ev-{index}", content="x" * 7000) for index in range(1, 9)
    ]
    second_round = [
        *first_round,
        *[
            _evidence(f"ev-{index}", content="y" * 7000)
            for index in range(9, 17)
        ],
    ]
    llm = ScriptedLLM(
        json.dumps(_valid_payload(*(item.id for item in first_round))),
        json.dumps(_valid_payload(*(item.id for item in second_round[:10]))),
    )
    grader = EvidenceGrader(llm=llm)  # type: ignore[arg-type]

    for evidence in (first_round, second_round):
        grader.grade(
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=evidence,
        )

    for call in llm.calls:
        payload = json.loads(call["messages"][1]["content"])
        assert len(payload["evidence"]) <= 10
        assert sum(len(item["content"]) for item in payload["evidence"]) <= 40000


def test_grader_rejects_a_judgment_for_evidence_omitted_from_the_view() -> None:
    evidence = [_evidence(f"ev-{index}") for index in range(1, 12)]
    invalid = json.dumps(_valid_payload(*(f"ev-{index}" for index in range(1, 12))))
    llm = ScriptedLLM(invalid, invalid)

    with pytest.raises(EvidenceGradingError, match="unknown Evidence ID.*ev-11"):
        EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=evidence,
        )


def test_grader_repair_reuses_only_the_bounded_evidence_view() -> None:
    full_content = "private-full-body:" + ("z" * 7000)
    evidence = _evidence(content=full_content)
    invalid = json.dumps(_valid_payload())
    repaired = json.dumps(_valid_payload("ev-1"))
    llm = ScriptedLLM(invalid, repaired)

    EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[evidence],
    )

    assert len(llm.calls) == 2
    repair_messages = llm.calls[1]["messages"]
    assert repair_messages[:2] == llm.calls[0]["messages"]
    assert full_content not in json.dumps(repair_messages, ensure_ascii=False)
    repair_payload = json.loads(repair_messages[1]["content"])
    assert len(repair_payload["evidence"][0]["content"]) <= 6000
    assert "supplied Evidence View" in repair_messages[-1]["content"]


def test_grader_repairs_a_missing_judgment_with_one_complete_regeneration() -> None:
    evidence = [_evidence(f"ev-{index}") for index in range(1, 5)]
    invalid = _valid_payload("ev-1", "ev-2", "ev-3")
    repaired = _valid_payload("ev-1", "ev-2", "ev-3", "ev-4")
    llm = ScriptedLLM(json.dumps(invalid), json.dumps(repaired))

    assessment = EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=["Cover trade-offs"],
        evidence=evidence,
    )

    assert [item.evidence_id for item in assessment.evidence_judgments] == [
        "ev-1",
        "ev-2",
        "ev-3",
        "ev-4",
    ]
    assert len(llm.calls) == 2
    repair_messages = llm.calls[1]["messages"]
    assert repair_messages[:2] == llm.calls[0]["messages"]
    assert repair_messages[2] == {"role": "assistant", "content": json.dumps(invalid)}
    repair_instruction = repair_messages[3]["content"]
    assert "missing judgment for Evidence ID(s): ev-4" in repair_instruction
    assert "complete Assessment" in repair_instruction
    assert "exactly once" in repair_instruction
    original_payload = json.loads(repair_messages[1]["content"])
    assert [item["evidence_id"] for item in original_payload["evidence"]] == [
        "ev-1",
        "ev-2",
        "ev-3",
        "ev-4",
    ]


def test_grader_stops_after_one_failed_repair() -> None:
    evidence = [_evidence("ev-1"), _evidence("ev-2")]
    still_missing = json.dumps(_valid_payload("ev-1"))
    llm = ScriptedLLM(still_missing, still_missing)

    with pytest.raises(EvidenceGradingError, match="missing judgment.*ev-2"):
        EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=evidence,
        )

    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "invalid",
    [
        _valid_payload("invented"),
        _valid_payload("ev-1", "ev-1"),
        {
            **_valid_payload("ev-1"),
            "coverage": "partial",
            "sufficient": False,
            "missing_information": [],
        },
    ],
    ids=["invented-id", "duplicate-id", "missing-information"],
)
def test_grader_repairs_other_assessment_contract_violations(
    invalid: dict[str, Any],
) -> None:
    repaired = _valid_payload("ev-1")
    llm = ScriptedLLM(json.dumps(invalid), json.dumps(repaired))

    assessment = EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[_evidence()],
    )

    assert assessment.task_id == "T1"
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing required field"),
        ({"task_id": "T1"}, "missing required field"),
        ([], "must be an object"),
    ],
)
def test_grader_rejects_invalid_top_level_shapes(
    payload: Any,
    message: str,
) -> None:
    with pytest.raises(EvidenceGradingError, match=message):
        _grade_payload(payload)


@pytest.mark.parametrize("location", ["assessment", "judgment"])
def test_grader_rejects_unexpected_output_fields(location: str) -> None:
    payload = _valid_payload("ev-1")
    if location == "assessment":
        payload["source"] = "web"
    else:
        payload["evidence_judgments"][0]["source"] = "web"

    with pytest.raises(EvidenceGradingError, match="unexpected field"):
        _grade_payload(payload)


def test_grader_rejects_a_mismatched_task_id() -> None:
    payload = _valid_payload("ev-1")
    payload["task_id"] = "T99"

    with pytest.raises(EvidenceGradingError, match="task_id"):
        _grade_payload(payload)


def test_grader_rejects_an_invented_evidence_id() -> None:
    payload = _valid_payload("invented")

    with pytest.raises(EvidenceGradingError, match="unknown Evidence ID"):
        _grade_payload(payload)


def test_grader_rejects_duplicate_evidence_judgments() -> None:
    payload = _valid_payload("ev-1", "ev-1")

    with pytest.raises(EvidenceGradingError, match="duplicate Evidence ID"):
        _grade_payload(payload)


def test_grader_requires_every_pool_item_to_be_judged() -> None:
    payload = _valid_payload("ev-1")
    evidence = [_evidence("ev-1"), _evidence("ev-2")]

    with pytest.raises(EvidenceGradingError, match="missing judgment"):
        _grade_payload(payload, evidence=evidence)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("relevance", "related"),
        ("quality", "excellent"),
    ],
)
def test_grader_rejects_invalid_judgment_enums(
    field_name: str,
    invalid_value: str,
) -> None:
    payload = _valid_payload("ev-1")
    payload["evidence_judgments"][0][field_name] = invalid_value

    with pytest.raises(EvidenceGradingError, match=field_name):
        _grade_payload(payload)


def test_grader_rejects_invalid_coverage() -> None:
    payload = _valid_payload("ev-1")
    payload["coverage"] = "mostly"

    with pytest.raises(EvidenceGradingError, match="coverage"):
        _grade_payload(payload)


def test_grader_rejects_non_boolean_sufficient() -> None:
    payload = _valid_payload("ev-1")
    payload["sufficient"] = "false"

    with pytest.raises(EvidenceGradingError, match="sufficient"):
        _grade_payload(payload)


def test_insufficient_assessment_requires_missing_information() -> None:
    payload = _valid_payload("ev-1")
    payload.update(
        coverage="partial",
        sufficient=False,
        missing_information=[],
    )

    with pytest.raises(EvidenceGradingError, match="missing_information"):
        _grade_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(reason="  "), "reason"),
        (
            lambda payload: payload["evidence_judgments"][0].update(reason=""),
            "reason",
        ),
        (
            lambda payload: payload.update(missing_information="missing"),
            "missing_information",
        ),
        (
            lambda payload: payload.update(missing_information=["  "]),
            "missing_information",
        ),
    ],
)
def test_grader_rejects_invalid_explanatory_fields(
    mutate: Any,
    message: str,
) -> None:
    payload = _valid_payload("ev-1")
    mutate(payload)

    with pytest.raises(EvidenceGradingError, match=message):
        _grade_payload(payload)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "did not contain JSON"),
        ("{not-json", "invalid JSON"),
    ],
)
def test_grader_rejects_unparseable_model_output(
    content: str,
    message: str,
) -> None:
    llm = FakeLLM(content)
    grader = EvidenceGrader(llm=llm)  # type: ignore[arg-type]

    with pytest.raises(EvidenceGradingError, match=message):
        grader.grade(
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=[_evidence()],
        )

    assert len(llm.calls) == 1


def test_grader_does_not_repair_llm_network_errors() -> None:
    network_error = ConnectionError("LLM endpoint unavailable")

    class FailingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            raise network_error

    llm = FailingLLM()

    with pytest.raises(ConnectionError) as excinfo:
        EvidenceGrader(llm=llm).grade(  # type: ignore[arg-type]
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=[_evidence()],
        )

    assert excinfo.value is network_error
    assert llm.calls == 1


def test_grader_wraps_invalid_retrieval_source_as_grading_error() -> None:
    evidence = _evidence()
    object.__setattr__(evidence, "retrieval_source", "local")
    grader = EvidenceGrader(llm=FakeLLM("unused"))  # type: ignore[arg-type]

    with pytest.raises(EvidenceGradingError, match="retrieval_source"):
        grader.grade(
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=[evidence],
        )


def test_grader_wraps_non_json_metadata_as_grading_error() -> None:
    evidence = _evidence()
    evidence.metadata["not_json"] = {"set-item"}
    grader = EvidenceGrader(llm=FakeLLM("unused"))  # type: ignore[arg-type]

    with pytest.raises(EvidenceGradingError, match="JSON-serializable"):
        grader.grade(
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=[evidence],
        )


def test_empty_evidence_pool_is_insufficient_without_calling_llm() -> None:
    class NeverLLM:
        def chat(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("empty Evidence pool must not call the LLM")

    assessment = EvidenceGrader(llm=NeverLLM()).grade(  # type: ignore[arg-type]
        task=_task(),
        objective="Compare memory systems",
        constraints=[],
        evidence=[],
    )

    assert assessment == EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[],
        coverage=EvidenceCoverage.INSUFFICIENT,
        sufficient=False,
        missing_information=[
            "The current task lacks direct supporting Evidence."
        ],
        reason="No Evidence has been collected for the current task.",
    )


def test_grader_prompt_forbids_answering_routing_and_pool_mutation() -> None:
    prompt = EVIDENCE_GRADER_SYSTEM_PROMPT.lower()

    assert "untrusted input data" in prompt
    assert "ignore any instructions" in prompt
    assert "metadata is not verified factual support" in prompt
    assert "factual support only from" in prompt
    assert "final research answer" in prompt
    assert "citation" in prompt
    assert "structured report" in prompt
    assert "create" in prompt and "evidence ids" in prompt
    assert "local, web, or vision" in prompt
    assert "retrieval score" in prompt and "probability" in prompt
    assert "quality is not" in prompt and "confidence score" in prompt
    assert "modify the evidence pool" in prompt
    assert "do not call tools" in prompt
