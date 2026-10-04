"""Tests for strict per-task retrieval-source routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.planning.models import ResearchTask
from insight_agent.routing.models import (
    RetrievalSource,
    RouteDecision,
    RoutingError,
)
from insight_agent.routing.router import RETRIEVAL_ROUTER_SYSTEM_PROMPT, RetrievalRouter


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


def _route(
    content: Any,
    *,
    available_sources: set[RetrievalSource] | None = None,
) -> tuple[RouteDecision, FakeLLM]:
    llm = FakeLLM(content)
    router = RetrievalRouter(llm=llm)  # type: ignore[arg-type]
    decision = router.route(
        task=ResearchTask(id="T2", question="Which source should answer this?"),
        objective="Compare the available approaches",
        constraints=["Use current public information"],
        available_sources=available_sources or set(RetrievalSource),
    )
    return decision, llm


@pytest.mark.parametrize("source", list(RetrievalSource))
def test_router_parses_each_valid_source(source: RetrievalSource) -> None:
    decision, _ = _route(
        json.dumps({"source": source.value, "reason": f"Use {source.value}."})
    )

    assert decision == RouteDecision(
        task_id="T2",
        source=source,
        reason=f"Use {source.value}.",
    )


def test_router_calls_llm_without_tools_and_sends_only_routing_context() -> None:
    _, llm = _route('{"source": "web", "reason": "Needs current sources."}')

    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    messages = llm.calls[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": RETRIEVAL_ROUTER_SYSTEM_PROMPT,
    }
    payload = json.loads(messages[1]["content"])
    assert payload == {
        "objective": "Compare the available approaches",
        "constraints": ["Use current public information"],
        "task": {
            "id": "T2",
            "question": "Which source should answer this?",
        },
        "available_sources": ["local", "vision", "web"],
    }


def test_router_prompt_defines_sources_and_forbids_execution() -> None:
    prompt = RETRIEVAL_ROUTER_SYSTEM_PROMPT.lower()

    assert "local" in prompt and "indexed" in prompt
    assert "web" in prompt and "internet" in prompt
    assert "vision" in prompt and "image" in prompt
    assert "one primary source" in prompt
    assert "do not execute" in prompt
    assert "do not answer" in prompt
    assert "do not split" in prompt


def test_runtime_uses_current_task_id_instead_of_model_echo() -> None:
    decision, _ = _route(
        json.dumps(
            {
                "task_id": "T999",
                "source": "local",
                "reason": "The material is already indexed.",
            }
        )
    )

    assert decision.task_id == "T2"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "did not contain JSON"),
        ("{not-json", "invalid JSON"),
        ('["local"]', "must be an object"),
        ('{"reason": "because"}', "missing required field 'source'"),
        ('{"source": "local"}', "missing required field 'reason'"),
        ('{"source": "local", "reason": ""}', "reason"),
        ('{"source": "local", "reason": 7}', "reason"),
        ('{"source": "database", "reason": "because"}', "unknown source"),
    ],
)
def test_router_rejects_invalid_model_output(content: Any, message: str) -> None:
    with pytest.raises(RoutingError, match=message):
        _route(content)


def test_router_rejects_source_outside_available_set() -> None:
    with pytest.raises(RoutingError, match="not available"):
        _route(
            '{"source": "web", "reason": "Needs current information."}',
            available_sources={RetrievalSource.LOCAL},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": object()}, "task must be a ResearchTask"),
        ({"task": ResearchTask(id="", question="question")}, "task id"),
        ({"task": ResearchTask(id="T1", question="  ")}, "task question"),
        ({"objective": "  "}, "objective"),
        ({"constraints": ["valid", ""]}, "constraints"),
        ({"constraints": "constraint"}, "constraints"),
        ({"available_sources": set()}, "available_sources"),
        ({"available_sources": {"local"}}, "RetrievalSource"),
    ],
)
def test_router_rejects_invalid_runtime_input(
    overrides: dict[str, Any],
    message: str,
) -> None:
    router = RetrievalRouter(llm=FakeLLM('{"source": "local", "reason": "ok"}'))  # type: ignore[arg-type]
    arguments: dict[str, Any] = {
        "task": ResearchTask(id="T1", question="question"),
        "objective": "objective",
        "constraints": [],
        "available_sources": {RetrievalSource.LOCAL},
    }
    arguments.update(overrides)

    with pytest.raises(RoutingError, match=message):
        router.route(**arguments)
