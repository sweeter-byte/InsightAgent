"""Tests for the `ResearchAgent` loop.

These tests must NEVER hit a real OpenAI-compatible endpoint. We drive the
loop with a scripted `FakeLLMClient` that returns pre-baked responses shaped
like the SDK's `ChatCompletion` object.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from insight_agent.agent import AgentStepsExceeded, DEFAULT_SYSTEM_PROMPT, ResearchAgent
from insight_agent.ingestion import SourceType
from insight_agent.retrieval import KnowledgeSearchTool, RetrievalResult
from insight_agent.tools.file_tools import READ_FILE_SCHEMA
from insight_agent.tools.registry import (
    ToolRegistry,
    build_default_registry,
    default_tool_schemas,
)


# ---------------------------------------------------------------------------
# Fake LLM plumbing
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> Any:
    """Wrap a plain-text assistant reply in an SDK-shaped object."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_tool_call_response(calls: Iterable[tuple[str, str, dict[str, Any]]]) -> Any:
    """Build an SDK-shaped response containing one or more tool calls.

    Each tuple is ``(id, name, arguments_dict)``. ``arguments`` is serialized
    to a JSON string to match the OpenAI wire format.
    """
    tool_calls = [
        SimpleNamespace(
            id=cid,
            type="function",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for cid, name, args in calls
    ]
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLMClient:
    """A scripted stand-in for `LLMClient.chat`."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []  # messages/tools snapshots

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        # Store a shallow copy so assertions can inspect what the agent sent.
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_agent(
    llm: FakeLLMClient,
    registry: ToolRegistry | None = None,
    *,
    max_steps: int = 10,
) -> ResearchAgent:
    return ResearchAgent(
        llm=llm,  # type: ignore[arg-type] — duck-typed against `LLMClient.chat`
        registry=registry or build_default_registry(),
        tool_schemas=[READ_FILE_SCHEMA],
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# 1. Happy path: one tool call, then a final text answer
# ---------------------------------------------------------------------------


def test_agent_executes_tool_call_and_returns_final_answer(tmp_path) -> None:
    f = tmp_path / "result.txt"
    f.write_text("Method A: 81.3%\nMethod B: 87.6%\nMethod C: 84.1%\n", encoding="utf-8")

    llm = FakeLLMClient(
        [
            _make_tool_call_response([("call_1", "read_file", {"path": str(f)})]),
            _make_text_response("Method B 的准确率最高,为 87.6%。"),
        ]
    )
    agent = _build_agent(llm)

    answer = agent.run("请读取 result.txt 并告诉我准确率最高的方法")

    assert answer == "Method B 的准确率最高,为 87.6%。"
    # Two LLM turns happened.
    assert len(llm.calls) == 2

    second_turn = llm.calls[1]["messages"]
    # user turn is preserved
    assert second_turn[1]["role"] == "user"
    # assistant message with the tool_call was appended
    assistant_msg = next(m for m in second_turn if m["role"] == "assistant")
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"
    # the tool result message carries the right tool_call_id and file contents
    tool_msg = next(m for m in second_turn if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "87.6%" in tool_msg["content"]


# ---------------------------------------------------------------------------
# 2. Unknown tool name
# ---------------------------------------------------------------------------


def test_unknown_tool_produces_error_message_not_crash() -> None:
    llm = FakeLLMClient(
        [
            _make_tool_call_response([("call_x", "no_such_tool", {})]),
            _make_text_response("I cannot use that tool."),
        ]
    )
    agent = _build_agent(llm)

    answer = agent.run("do something")

    assert answer == "I cannot use that tool."
    tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("Error:")
    assert "Unknown tool" in tool_msg["content"]


# ---------------------------------------------------------------------------
# 3. Illegal JSON arguments
# ---------------------------------------------------------------------------


def test_invalid_json_arguments_are_reported_as_tool_error() -> None:
    bad_call = SimpleNamespace(
        id="call_bad",
        type="function",
        function=SimpleNamespace(name="read_file", arguments="not-a-json{"),
    )
    llm = FakeLLMClient(
        [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, tool_calls=[bad_call]
            ))]),
            _make_text_response("参数解析失败,无法读取。"),
        ]
    )
    agent = _build_agent(llm)

    answer = agent.run("read something")

    assert answer == "参数解析失败,无法读取。"
    tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_bad"
    assert "invalid arguments" in tool_msg["content"]


# ---------------------------------------------------------------------------
# 4. max_steps guard
# ---------------------------------------------------------------------------


def test_max_steps_terminates_loop_with_clear_error(tmp_path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")

    # Every turn produces another tool call; the loop should never terminate
    # on its own and must trip the max_steps guard.
    llm = FakeLLMClient(
        [_make_tool_call_response([(f"call_{i}", "read_file", {"path": str(f)})])
         for i in range(10)]
    )
    agent = _build_agent(llm, max_steps=3)

    with pytest.raises(AgentStepsExceeded, match="exceeded 3 steps"):
        agent.run("loop forever")

    # exactly `max_steps` LLM calls were made
    assert len(llm.calls) == 3


# ---------------------------------------------------------------------------
# 5. Multiple tool calls in a single assistant turn
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_in_one_turn_all_execute(tmp_path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A", encoding="utf-8")
    b.write_text("B", encoding="utf-8")

    llm = FakeLLMClient(
        [
            _make_tool_call_response(
                [("c1", "read_file", {"path": str(a)}),
                 ("c2", "read_file", {"path": str(b)})]
            ),
            _make_text_response("done"),
        ]
    )
    agent = _build_agent(llm)

    answer = agent.run("read both")

    assert answer == "done"
    tool_msgs = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}
    assert {m["content"] for m in tool_msgs} == {"A", "B"}


# ---------------------------------------------------------------------------
# 6. Empty assistant text without tool calls — must not crash
# ---------------------------------------------------------------------------


def test_empty_final_text_returns_empty_string() -> None:
    llm = FakeLLMClient([_make_text_response("")])
    agent = _build_agent(llm)

    assert agent.run("say nothing") == ""


# ---------------------------------------------------------------------------
# 7. Constructor guards
# ---------------------------------------------------------------------------


def test_max_steps_must_be_positive() -> None:
    llm = FakeLLMClient([])
    with pytest.raises(ValueError):
        ResearchAgent(llm=llm,  # type: ignore[arg-type]
                      registry=build_default_registry(),
                      tool_schemas=[READ_FILE_SCHEMA],
                      max_steps=0)


def test_agent_dispatches_knowledge_search_as_an_ordinary_tool_observation() -> None:
    class FixtureRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            assert (query, top_k) == ("Project Helios highest score", 2)
            return [
                RetrievalResult(
                    chunk_id="helios-beta",
                    score=0.99,
                    content="Project Helios Beta scored 48.9.",
                    document_id="helios-document",
                    source="helios.txt",
                    source_type=SourceType.TEXT,
                    chunk_index=0,
                    start_char=0,
                    end_char=32,
                    metadata={"method": "Beta"},
                )
            ]

    llm = FakeLLMClient(
        [
            _make_tool_call_response(
                [
                    (
                        "search_1",
                        "search_knowledge_base",
                        {"query": "Project Helios highest score", "top_k": 2},
                    )
                ]
            ),
            _make_text_response("Beta has the highest score: 48.9."),
        ]
    )
    registry = build_default_registry(
        knowledge_search_tool=KnowledgeSearchTool(FixtureRetriever())
    )
    agent = ResearchAgent(
        llm=llm,  # type: ignore[arg-type]
        registry=registry,
        tool_schemas=default_tool_schemas(),
    )

    agent.run("Which Project Helios method scored highest?")

    assert len(llm.calls) == 2
    tool_message = next(
        message for message in llm.calls[1]["messages"] if message["role"] == "tool"
    )
    assert tool_message["tool_call_id"] == "search_1"
    assert "source: helios.txt" in tool_message["content"]
    assert "Beta scored 48.9" in tool_message["content"]
    schema_names = {
        schema["function"]["name"] for schema in llm.calls[0]["tools"]
    }
    assert schema_names == {"read_file", "search_knowledge_base"}


def test_agent_prompt_requires_grounded_local_knowledge_answers() -> None:
    assert "search_knowledge_base" in DEFAULT_SYSTEM_PROMPT
    assert "local knowledge" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "insufficient" in DEFAULT_SYSTEM_PROMPT.lower()
    assert "not present" in DEFAULT_SYSTEM_PROMPT.lower()
