from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any

import pytest

from evals import (
    EvalCase,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    JudgeResult,
)
from evals.judge import (
    EVALUATION_JUDGE_SYSTEM_PROMPT,
    JudgeOutputError,
    JudgeRequest,
    JudgeTask,
    JudgeTimeoutError,
    LLMEvaluationJudge,
    evaluate_with_judge,
)
from insight_agent.evidence.grader import EVIDENCE_GRADER_SYSTEM_PROMPT
from insight_agent.self_check.prompts import SELF_CHECK_SYSTEM_PROMPT


def _request() -> JudgeRequest:
    return JudgeRequest(
        task=JudgeTask.CLAIM_GROUNDEDNESS,
        case_id="case-1",
        query="What does the evidence support?",
        rubric=("Use only the bound Evidence.",),
        payload={
            "claim": {"id": "T1-C1", "text": "Supported fact."},
            "evidence": [{"id": "E1", "content": "Supported fact."}],
        },
    )


@pytest.mark.parametrize("label", list(JudgeLabel))
def test_evaluate_with_judge_preserves_valid_finite_label(label: JudgeLabel) -> None:
    class FakeJudge:
        def evaluate(self, request: JudgeRequest) -> JudgeResult:
            assert request == _request()
            return JudgeResult(label, "Short rubric-specific reason.")

    outcome = evaluate_with_judge(FakeJudge(), _request())

    assert outcome == JudgeEvaluation(
        status=JudgeExecutionStatus.COMPLETED,
        result=JudgeResult(label, "Short rubric-specific reason."),
    )


@pytest.mark.parametrize(
    ("returned", "reason_fragment"),
    [
        ({"label": "pass", "reason": "not typed"}, "JudgeResult"),
        (JudgeResult("unknown", "reason"), "label"),  # type: ignore[arg-type]
        (JudgeResult(JudgeLabel.PASS, " "), "reason"),
    ],
)
def test_evaluate_with_judge_marks_illegal_return_as_invalid_output(
    returned: object,
    reason_fragment: str,
) -> None:
    class FakeJudge:
        def evaluate(self, request: JudgeRequest) -> Any:
            return returned

    outcome = evaluate_with_judge(FakeJudge(), _request())

    assert outcome.status is JudgeExecutionStatus.INVALID_OUTPUT
    assert outcome.result is None
    assert reason_fragment in outcome.reason


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (TimeoutError("deadline"), JudgeExecutionStatus.TIMEOUT),
        (JudgeTimeoutError("sdk timeout"), JudgeExecutionStatus.TIMEOUT),
        (JudgeOutputError("bad JSON"), JudgeExecutionStatus.INVALID_OUTPUT),
        (RuntimeError("endpoint down"), JudgeExecutionStatus.BACKEND_ERROR),
    ],
)
def test_evaluate_with_judge_keeps_failures_out_of_semantic_labels(
    error: Exception,
    status: JudgeExecutionStatus,
) -> None:
    class FakeJudge:
        def evaluate(self, request: JudgeRequest) -> JudgeResult:
            raise error

    outcome = evaluate_with_judge(FakeJudge(), _request())

    assert outcome.status is status
    assert outcome.result is None
    assert outcome.reason


class FakeLLM:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: object = None,
        *,
        timeout: float | None = None,
    ) -> object:
        self.calls.append(
            {"messages": messages, "tools": tools, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def test_llm_judge_uses_independent_prompt_strict_payload_and_timeout() -> None:
    llm = FakeLLM('{"label":"partial","reason":"One point is missing."}')
    judge = LLMEvaluationJudge(llm=llm, timeout_seconds=2.5)

    result = judge.evaluate(_request())

    assert result == JudgeResult(JudgeLabel.PARTIAL, "One point is missing.")
    call = llm.calls[0]
    assert call["tools"] is None
    assert call["timeout"] == 2.5
    messages = call["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "system",
        "content": EVALUATION_JUDGE_SYSTEM_PROMPT,
    }
    assert EVALUATION_JUDGE_SYSTEM_PROMPT != EVIDENCE_GRADER_SYSTEM_PROMPT
    assert EVALUATION_JUDGE_SYSTEM_PROMPT != SELF_CHECK_SYSTEM_PROMPT
    payload = json.loads(messages[1]["content"])
    assert payload["task"] == "claim_groundedness"
    assert set(payload) == {"task", "case_id", "query", "rubric", "payload"}
    assert "assessment" not in messages[1]["content"].lower()
    assert "self_check" not in messages[1]["content"].lower()


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        "[]",
        '{"label":"pass"}',
        '{"label":"unknown","reason":"bad"}',
        '{"label":"pass","reason":" "}',
        '{"label":"pass","reason":"ok","extra":true}',
    ],
)
def test_llm_judge_rejects_illegal_model_output(content: str) -> None:
    judge = LLMEvaluationJudge(llm=FakeLLM(content), timeout_seconds=1.0)

    with pytest.raises(JudgeOutputError):
        judge.evaluate(_request())


def test_llm_judge_maps_client_timeout_without_calling_network() -> None:
    judge = LLMEvaluationJudge(
        llm=FakeLLM(error=TimeoutError("deadline")),
        timeout_seconds=1.0,
    )

    with pytest.raises(JudgeTimeoutError, match="deadline"):
        judge.evaluate(_request())
