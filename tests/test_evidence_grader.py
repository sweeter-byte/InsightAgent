"""Tests for validated, retrieval-independent Evidence grading."""

from __future__ import annotations

import json
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


def _task() -> ResearchTask:
    return ResearchTask(id="T1", question="Compare memory mechanisms.")


def _evidence(evidence_id: str = "ev-1") -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="T1",
        retrieval_source=RetrievalSource.LOCAL,
        origin_id="chunk-1",
        content="Vector memory uses embedding similarity.",
        source="notes.md",
        metadata={"chunk_id": "chunk-1", "retrieval_score": 0.91},
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
    grader = EvidenceGrader(llm=FakeLLM(content))  # type: ignore[arg-type]

    with pytest.raises(EvidenceGradingError, match=message):
        grader.grade(
            task=_task(),
            objective="Compare memory systems",
            constraints=[],
            evidence=[_evidence()],
        )


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
    assert "final research answer" in prompt
    assert "citation" in prompt
    assert "structured report" in prompt
    assert "create" in prompt and "evidence ids" in prompt
    assert "local, web, or vision" in prompt
    assert "retrieval score" in prompt and "probability" in prompt
    assert "quality is not" in prompt and "confidence score" in prompt
    assert "modify the evidence pool" in prompt
    assert "do not call tools" in prompt
