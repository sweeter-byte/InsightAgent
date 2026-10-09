from __future__ import annotations

import asyncio
import threading

import pytest

from insight_agent.application.query import (
    QueryExecutionError,
    QueryRoutingError,
    QueryRuntimeError,
    QueryService,
)
from insight_agent.router import Intent
from insight_agent.runtime.errors import RuntimeUnavailable
from insight_agent.runtime.models import RunRecord


class FakeApplication:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent
        self.route_calls: list[str] = []
        self.direct_calls: list[str] = []
        self.analyze_calls: list[str] = []

    def route(self, query: str) -> Intent:
        self.route_calls.append(query)
        return self.intent

    def answer_direct(self, query: str) -> str:
        self.direct_calls.append(query)
        return "direct answer"

    def analyze(self, query: str) -> str:
        self.analyze_calls.append(query)
        return "analysis answer"


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def create_run(
        self,
        query: str,
        *,
        request_id: str | None = None,
    ) -> RunRecord:
        self.calls.append((query, request_id))
        return RunRecord.new(run_id="run-1", thread_id="thread-1")


def _service(
    intent: Intent,
) -> tuple[QueryService, FakeApplication, FakeRuntime]:
    application = FakeApplication(intent)
    runtime = FakeRuntime()
    service = QueryService(application, runtime)  # type: ignore[arg-type]
    return service, application, runtime


@pytest.mark.asyncio
async def test_direct_uses_only_direct_answer_branch() -> None:
    service, application, runtime = _service(Intent.DIRECT)

    result = await service.query("hello", request_id="request-1")

    assert result.model_dump(mode="json") == {
        "request_id": "request-1",
        "intent": "direct",
        "status": "completed",
        "answer": "direct answer",
        "run_id": None,
    }
    assert application.route_calls == ["hello"]
    assert application.direct_calls == ["hello"]
    assert application.analyze_calls == []
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_analyze_uses_only_research_agent_branch() -> None:
    service, application, runtime = _service(Intent.ANALYZE)

    result = await service.query("inspect notes", request_id="request-2")

    assert result.model_dump(mode="json") == {
        "request_id": "request-2",
        "intent": "analyze",
        "status": "completed",
        "answer": "analysis answer",
        "run_id": None,
    }
    assert application.route_calls == ["inspect notes"]
    assert application.direct_calls == []
    assert application.analyze_calls == ["inspect notes"]
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_research_submits_only_to_runtime_and_returns_run_id() -> None:
    service, application, runtime = _service(Intent.RESEARCH)

    result = await service.query("research agents", request_id="request-3")

    assert result.model_dump(mode="json") == {
        "request_id": "request-3",
        "intent": "research",
        "status": "queued",
        "answer": None,
        "run_id": "run-1",
    }
    assert application.route_calls == ["research agents"]
    assert application.direct_calls == []
    assert application.analyze_calls == []
    assert runtime.calls == [("research agents", "request-3")]


@pytest.mark.asyncio
async def test_all_intents_have_the_same_stable_response_fields() -> None:
    fields: list[set[str]] = []
    for intent in Intent:
        service, _, _ = _service(intent)
        result = await service.query("query", request_id="request")
        fields.append(set(result.model_dump()))

    assert fields == [
        {"request_id", "intent", "status", "answer", "run_id"}
    ] * len(Intent)


@pytest.mark.asyncio
async def test_routing_failure_is_sanitized_and_logged(caplog) -> None:
    service, application, _ = _service(Intent.DIRECT)
    query = "private query text"

    def fail_route(_query: str) -> Intent:
        raise RuntimeError("secret-key-value")

    application.route = fail_route  # type: ignore[method-assign]
    with caplog.at_level("ERROR"), pytest.raises(
        QueryRoutingError, match="unable to route query"
    ) as raised:
        await service.query(query, request_id="request-1")

    assert "secret-key-value" not in str(raised.value)
    record = caplog.records[-1]
    assert record.request_id == "request-1"  # type: ignore[attr-defined]
    assert record.stage == "routing"  # type: ignore[attr-defined]
    assert query not in record.getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", [Intent.DIRECT, Intent.ANALYZE])
async def test_synchronous_execution_failure_is_sanitized_and_logged(
    intent: Intent,
    caplog,
) -> None:
    service, application, _ = _service(intent)
    query = "private query text"

    def fail(_query: str) -> str:
        raise RuntimeError("secret-key-value")

    if intent is Intent.DIRECT:
        application.answer_direct = fail  # type: ignore[method-assign]
    else:
        application.analyze = fail  # type: ignore[method-assign]

    with caplog.at_level("ERROR"), pytest.raises(
        QueryExecutionError, match="unable to execute query"
    ) as raised:
        await service.query(query, request_id="request-2")

    assert "secret-key-value" not in str(raised.value)
    record = caplog.records[-1]
    assert record.request_id == "request-2"  # type: ignore[attr-defined]
    assert record.stage == "execution"  # type: ignore[attr-defined]
    assert record.intent == intent.value  # type: ignore[attr-defined]
    assert query not in record.getMessage()


@pytest.mark.asyncio
async def test_unexpected_runtime_failure_is_sanitized_and_logged(caplog) -> None:
    service, _, runtime = _service(Intent.RESEARCH)
    query = "private query text"

    async def fail_create_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("secret-key-value")

    runtime.create_run = fail_create_run  # type: ignore[method-assign]
    with caplog.at_level("ERROR"), pytest.raises(
        QueryRuntimeError, match="research runtime unavailable"
    ) as raised:
        await service.query(query, request_id="request-3")

    assert "secret-key-value" not in str(raised.value)
    record = caplog.records[-1]
    assert record.request_id == "request-3"  # type: ignore[attr-defined]
    assert record.stage == "runtime_submission"  # type: ignore[attr-defined]
    assert record.intent == "research"  # type: ignore[attr-defined]
    assert query not in record.getMessage()


@pytest.mark.asyncio
async def test_runtime_unavailable_type_is_preserved(caplog) -> None:
    service, _, runtime = _service(Intent.RESEARCH)
    unavailable = RuntimeUnavailable("secret-key-value")

    async def fail_create_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise unavailable

    runtime.create_run = fail_create_run  # type: ignore[method-assign]
    with caplog.at_level("ERROR"), pytest.raises(RuntimeUnavailable) as raised:
        await service.query("query", request_id="request-4")

    assert raised.value is unavailable
    record = caplog.records[-1]
    assert record.request_id == "request-4"  # type: ignore[attr-defined]
    assert record.stage == "runtime_submission"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_blocking_router_does_not_block_event_loop_scheduling() -> None:
    service, application, _ = _service(Intent.DIRECT)
    started = threading.Event()
    release = threading.Event()

    def blocking_route(_query: str) -> Intent:
        started.set()
        release.wait()
        return Intent.DIRECT

    application.route = blocking_route  # type: ignore[method-assign]
    fallback_release = threading.Timer(1, release.set)
    fallback_release.start()
    task = asyncio.create_task(service.query("query", request_id="request-5"))
    try:
        assert await asyncio.to_thread(started.wait, 0.5)
        scheduled = asyncio.Event()
        asyncio.get_running_loop().call_soon(scheduled.set)
        await asyncio.wait_for(scheduled.wait(), timeout=0.2)
        assert not task.done()
    finally:
        release.set()
        fallback_release.cancel()

    result = await asyncio.wait_for(task, timeout=1)
    assert result.answer == "direct answer"
