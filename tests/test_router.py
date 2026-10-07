"""Tests for the Intent Router.

These tests must NEVER hit a real API. We use a scripted FakeLLM that returns
pre-baked SDK-shaped responses to exercise all classification and fallback paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.router import Intent, IntentRouter, ROUTER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Fake LLM plumbing (mirrors the pattern in test_agent.py)
# ---------------------------------------------------------------------------


def _make_text_response(text: str) -> Any:
    """Wrap a plain-text assistant reply in an SDK-shaped object."""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeLLM:
    """A minimal stand-in for `LLMClient` that returns a single scripted response."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_router(response_text: str) -> tuple[IntentRouter, FakeLLM]:
    fake = FakeLLM(_make_text_response(response_text))
    router = IntentRouter(llm=fake)  # type: ignore[arg-type]
    return router, fake


# ---------------------------------------------------------------------------
# 1. Valid intent responses
# ---------------------------------------------------------------------------


def test_route_returns_direct() -> None:
    router, _ = _build_router("direct")
    assert router.route("Python 中 list 和 tuple 有什么区别？") == Intent.DIRECT


def test_route_returns_analyze() -> None:
    router, _ = _build_router("analyze")
    assert router.route("读取 notes.txt，告诉我准确率最高的方法") == Intent.ANALYZE


def test_route_returns_research() -> None:
    router, _ = _build_router("research")
    assert router.route("调研最近 Agent Memory 的主要研究方向并进行比较") == Intent.RESEARCH


# ---------------------------------------------------------------------------
# 2. Whitespace and case normalization
# ---------------------------------------------------------------------------


def test_route_handles_whitespace_padded_response() -> None:
    router, _ = _build_router("  analyze  ")
    assert router.route("summarize this file") == Intent.ANALYZE


def test_route_handles_mixed_case_response() -> None:
    router, _ = _build_router("Research")
    assert router.route("latest trends in LLM agents") == Intent.RESEARCH


def test_route_handles_uppercase_response() -> None:
    router, _ = _build_router("DIRECT")
    assert router.route("what is 2+2") == Intent.DIRECT


# ---------------------------------------------------------------------------
# 3. Fallback cases — invalid or empty output → Intent.RESEARCH
# ---------------------------------------------------------------------------


def test_route_fallback_on_invalid_text() -> None:
    router, _ = _build_router("I think this is research.")
    assert router.route("anything") == Intent.RESEARCH


def test_route_fallback_on_empty_string() -> None:
    router, _ = _build_router("")
    assert router.route("anything") == Intent.RESEARCH


def test_route_fallback_on_whitespace_only() -> None:
    router, _ = _build_router("   ")
    assert router.route("anything") == Intent.RESEARCH


def test_route_fallback_on_unrecognized_keyword() -> None:
    router, _ = _build_router("maybe")
    assert router.route("anything") == Intent.RESEARCH


# ---------------------------------------------------------------------------
# 4. Verify LLM is called without tools
# ---------------------------------------------------------------------------


def test_router_does_not_pass_tools_to_llm() -> None:
    router, fake = _build_router("direct")
    router.route("hello")

    assert len(fake.calls) == 1
    assert fake.calls[0]["tools"] is None


# ---------------------------------------------------------------------------
# 5. Verify message structure sent to LLM
# ---------------------------------------------------------------------------


def test_router_sends_system_and_user_messages() -> None:
    router, fake = _build_router("analyze")
    query = "读取 report.txt 并总结"
    router.route(query)

    messages = fake.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == ROUTER_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == query


# ---------------------------------------------------------------------------
# 6. Prompt contract
# ---------------------------------------------------------------------------


def test_prompt_prioritizes_explicit_research_over_named_materials() -> None:
    assert (
        "An explicit request to conduct research MUST be classified as research"
        in ROUTER_SYSTEM_PROMPT
    )
    assert "indexed local materials or images" in ROUTER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 7. Intent enum value checks
# ---------------------------------------------------------------------------


def test_intent_enum_values() -> None:
    assert Intent.DIRECT == "direct"
    assert Intent.ANALYZE == "analyze"
    assert Intent.RESEARCH == "research"


def test_intent_is_str_subclass() -> None:
    assert isinstance(Intent.DIRECT, str)
    assert isinstance(Intent.ANALYZE, str)
    assert isinstance(Intent.RESEARCH, str)
