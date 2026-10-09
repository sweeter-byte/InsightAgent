"""Tests for the `InsightAgent` façade — routing dispatch only.

The point of these tests is NOT answer quality; it is to verify that a query
lands on the correct execution path. Real LLM / real agent loops are never
invoked: fakes capture every call for inspection.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.app import DIRECT_SYSTEM_PROMPT, InsightAgent
from insight_agent.router import Intent, IntentRouter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRouter:
    """Pretends to be ``IntentRouter``; always returns the pre-set intent."""

    def __init__(self, intent: Intent) -> None:
        self._intent = intent
        self.calls: list[str] = []

    def route(self, query: str) -> Intent:
        self.calls.append(query)
        return self._intent


class FakeLLM:
    """Pretends to be ``LLMClient``; records every chat call and returns a
    scripted SDK-shaped response carrying ``reply_text`` as the assistant
    content."""

    def __init__(self, reply_text: str = "direct answer") -> None:
        self.reply_text = reply_text
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        message = SimpleNamespace(content=self.reply_text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeResearchAgent:
    """Pretends to be ``ResearchAgent``; records each ``run`` invocation."""

    def __init__(self, reply_text: str = "research answer") -> None:
        self.reply_text = reply_text
        self.calls: list[str] = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        return self.reply_text


class FakeResearchCoordinator:
    """Pretends to plan and then invoke the existing research path."""

    def __init__(self, reply_text: str = "planned research answer") -> None:
        self.reply_text = reply_text
        self.calls: list[str] = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        return self.reply_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(
    intent: Intent,
    *,
    direct_reply: str = "direct answer",
    research_reply: str = "research answer",
) -> tuple[
    InsightAgent,
    FakeRouter,
    FakeLLM,
    FakeResearchAgent,
    FakeResearchCoordinator,
]:
    router = FakeRouter(intent)
    llm = FakeLLM(direct_reply)
    research = FakeResearchAgent(research_reply)
    coordinator = FakeResearchCoordinator(research_reply)
    app = InsightAgent(
        router=router,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        research_agent=research,  # type: ignore[arg-type]
        research_coordinator=coordinator,  # type: ignore[arg-type]
    )
    return app, router, llm, research, coordinator


def test_close_releases_callbacks_once_in_reverse_registration_order() -> None:
    events: list[str] = []
    router = FakeRouter(Intent.DIRECT)
    app = InsightAgent(
        router=router,  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        research_agent=FakeResearchAgent(),  # type: ignore[arg-type]
        research_coordinator=FakeResearchCoordinator(),  # type: ignore[arg-type]
        close_callbacks=[
            lambda: events.append("first"),
            lambda: events.append("second"),
        ],
    )

    app.close()
    app.close()

    assert events == ["second", "first"]


def test_close_attempts_all_callbacks_before_propagating_cleanup_error() -> None:
    events: list[str] = []
    app, *_ = _build(Intent.DIRECT)
    owned_resources = ExitStack()
    owned_resources.callback(lambda: events.append("llm"))
    owned_resources.callback(lambda: events.append("checkpoint"))
    failure = RuntimeError("cleanup failed")

    def fail_cleanup() -> None:
        events.append("failing")
        raise failure

    app.add_close_callback(owned_resources.close)
    app.add_close_callback(lambda: events.append("middle"))
    app.add_close_callback(fail_cleanup)
    app.add_close_callback(lambda: events.append("last"))

    with pytest.raises(RuntimeError, match="cleanup failed") as raised:
        app.close()

    assert raised.value is failure
    assert events == ["last", "failing", "middle", "checkpoint", "llm"]
    app.close()
    assert events == ["last", "failing", "middle", "checkpoint", "llm"]


# ---------------------------------------------------------------------------
# 1. DIRECT path — plain chat, no tools, no ResearchAgent
# ---------------------------------------------------------------------------


def test_direct_intent_uses_llm_not_research_agent() -> None:
    app, router, llm, research, coordinator = _build(
        Intent.DIRECT, direct_reply="hello there"
    )

    answer = app.run("what is 2+2")

    # router was consulted exactly once with the raw query
    assert router.calls == ["what is 2+2"]
    # the LLM was hit once for the direct answer
    assert len(llm.calls) == 1
    # ResearchAgent was NOT invoked
    assert research.calls == []
    assert coordinator.calls == []
    # the returned text is the direct LLM reply
    assert answer == "hello there"


def test_direct_call_passes_no_tools() -> None:
    app, _, llm, _, _ = _build(Intent.DIRECT)

    app.run("anything")

    assert llm.calls[0]["tools"] is None


def test_direct_call_sends_system_and_user_messages() -> None:
    app, _, llm, _, _ = _build(Intent.DIRECT)

    app.run("tell me a joke")

    messages = llm.calls[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == DIRECT_SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "tell me a joke"


def test_direct_returns_empty_string_when_llm_has_no_choices() -> None:
    """A malformed SDK response should not crash the direct path."""

    class EmptyLLM:
        def chat(self, messages, tools=None):  # noqa: ANN001, ARG002
            return SimpleNamespace(choices=[])

    router = FakeRouter(Intent.DIRECT)
    research = FakeResearchAgent()
    coordinator = FakeResearchCoordinator()
    app = InsightAgent(
        router=router,  # type: ignore[arg-type]
        llm=EmptyLLM(),  # type: ignore[arg-type]
        research_agent=research,  # type: ignore[arg-type]
        research_coordinator=coordinator,  # type: ignore[arg-type]
    )

    assert app.run("boom") == ""
    assert research.calls == []
    assert coordinator.calls == []


# ---------------------------------------------------------------------------
# 2. ANALYZE path — ResearchAgent.run, direct LLM untouched
# ---------------------------------------------------------------------------


def test_analyze_intent_delegates_to_research_agent() -> None:
    app, router, llm, research, coordinator = _build(
        Intent.ANALYZE, research_reply="Method B wins"
    )

    answer = app.run("read notes.txt and summarize")

    assert router.calls == ["read notes.txt and summarize"]
    assert research.calls == ["read notes.txt and summarize"]
    assert coordinator.calls == []
    # the direct-answer LLM must not be invoked on the analyze path
    assert llm.calls == []
    assert answer == "Method B wins"


# ---------------------------------------------------------------------------
# 3. RESEARCH path — planner/coordinator only
# ---------------------------------------------------------------------------


def test_research_intent_delegates_to_research_coordinator() -> None:
    app, router, llm, research, coordinator = _build(
        Intent.RESEARCH, research_reply="survey complete"
    )

    answer = app.run("investigate recent Agent Memory papers")

    assert router.calls == ["investigate recent Agent Memory papers"]
    assert research.calls == []
    assert coordinator.calls == ["investigate recent Agent Memory papers"]
    assert llm.calls == []
    assert answer == "survey complete"


def test_analyze_and_research_use_distinct_execution_paths() -> None:
    app_a, router_a, _, research_a, coordinator_a = _build(Intent.ANALYZE)
    app_r, router_r, _, research_r, coordinator_r = _build(Intent.RESEARCH)

    app_a.run("q-analyze")
    app_r.run("q-research")

    assert router_a._intent is Intent.ANALYZE
    assert router_r._intent is Intent.RESEARCH
    assert research_a.calls == ["q-analyze"]
    assert coordinator_a.calls == []
    assert research_r.calls == []
    assert coordinator_r.calls == ["q-research"]


# ---------------------------------------------------------------------------
# 4. Wiring sanity — the app must not touch anything outside the injection
# ---------------------------------------------------------------------------


def test_run_does_not_construct_real_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``app.run`` internally tried to build a real LLMClient, we'd hit env
    validation. FakeRouter/FakeLLM/FakeResearchAgent being the only collaborators
    means run() should succeed with no env vars set at all.
    """
    for var in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    app, _, _, _, _ = _build(Intent.DIRECT, direct_reply="ok")
    assert app.run("hi") == "ok"


# ---------------------------------------------------------------------------
# 5. Direct system prompt is overridable but has a sensible default
# ---------------------------------------------------------------------------


def test_custom_direct_prompt_is_used() -> None:
    router = FakeRouter(Intent.DIRECT)
    llm = FakeLLM("resp")
    research = FakeResearchAgent()
    coordinator = FakeResearchCoordinator()
    app = InsightAgent(
        router=router,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        research_agent=research,  # type: ignore[arg-type]
        research_coordinator=coordinator,  # type: ignore[arg-type]
        direct_system_prompt="custom prompt",
    )

    app.run("hello")

    assert llm.calls[0]["messages"][0]["content"] == "custom prompt"


# ---------------------------------------------------------------------------
# 6. Router fallback end-to-end — invalid LLM output still executes
# ---------------------------------------------------------------------------


def test_router_fallback_still_executes_via_research_coordinator() -> None:
    """A real ``IntentRouter`` fed garbage falls back to RESEARCH, and the app
    still dispatches to ResearchAgent — no crash, empty answer never returned.
    """
    # FakeLLM returns prose instead of a single intent word → router falls back.
    routing_llm = FakeLLM("I am not sure which category this belongs to.")
    router = IntentRouter(llm=routing_llm)  # type: ignore[arg-type]
    answering_llm = FakeLLM("unused direct answer")
    research = FakeResearchAgent(reply_text="handled after fallback")
    coordinator = FakeResearchCoordinator(reply_text="handled after fallback")

    app = InsightAgent(
        router=router,
        llm=answering_llm,  # type: ignore[arg-type]
        research_agent=research,  # type: ignore[arg-type]
        research_coordinator=coordinator,  # type: ignore[arg-type]
    )

    answer = app.run("some ambiguous query")

    # the router consumed the fallback LLM (one classification call, no tools)
    assert len(routing_llm.calls) == 1
    assert routing_llm.calls[0]["tools"] is None
    # fallback intent is RESEARCH → planner/coordinator path handled it
    assert research.calls == []
    assert coordinator.calls == ["some ambiguous query"]
    # the direct-answer LLM was never used
    assert answering_llm.calls == []
    assert answer == "handled after fallback"
