from __future__ import annotations

import asyncio
from fastapi.testclient import TestClient

from insight_agent.runtime.api import _event_stream, create_api
from insight_agent.runtime.events import InMemoryRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus, RuntimeEvent
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import InMemoryRunRegistry
from insight_agent.runtime.service import ProgressCallback, ResearchRuntimeService


class InstantRunner:
    async def run(
        self,
        *,
        thread_id: str,
        query: str | None,
        resume: bool,
        on_progress: ProgressCallback,
    ) -> str:
        del query, resume
        await on_progress("retrieval", {"task_id": "T1"})
        return f"output:{thread_id}"


def _runtime() -> tuple[
    ResearchRuntimeService, InMemoryRunRegistry, InMemoryRuntimeEventStore
]:
    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=InstantRunner(),
        policy=RuntimePolicy(infra_retry_backoff_seconds=0),
        run_id_factory=lambda: "run-1",
        thread_id_factory=lambda: "thread-1",
    )
    return service, registry, events


def test_create_returns_202_without_exposing_research_state() -> None:
    service, _, _ = _runtime()
    with TestClient(create_api(service)) as client:
        response = client.post("/v1/research/runs", json={"query": "research"})

    assert response.status_code == 202
    assert response.json()["run_id"] == "run-1"
    assert response.json()["thread_id"] == "thread-1"
    assert response.json()["status"] == "queued"
    assert "plan" not in response.json()
    assert response.headers["x-request-id"]


def test_blank_query_is_422_and_unknown_run_is_404() -> None:
    service, _, _ = _runtime()
    with TestClient(create_api(service)) as client:
        blank = client.post("/v1/research/runs", json={"query": "  "})
        missing = client.get("/v1/research/runs/missing")

    assert blank.status_code == 422
    assert missing.status_code == 404


def test_illegal_resume_is_409() -> None:
    service, registry, _ = _runtime()
    asyncio.run(
        registry.create(
            RunRecord.new(
                run_id="completed",
                thread_id="thread-completed",
                status=RunStatus.COMPLETED,
            )
        )
    )
    with TestClient(create_api(service)) as client:
        response = client.post("/v1/research/runs/completed/resume")

    assert response.status_code == 409


def test_legal_resume_returns_202_with_same_identifiers() -> None:
    service, registry, _ = _runtime()
    asyncio.run(
        registry.create(
            RunRecord.new(
                run_id="interrupted",
                thread_id="original-thread",
                status=RunStatus.INTERRUPTED,
            )
        )
    )
    with TestClient(create_api(service)) as client:
        response = client.post("/v1/research/runs/interrupted/resume")

    assert response.status_code == 202
    assert response.json()["run_id"] == "interrupted"
    assert response.json()["thread_id"] == "original-thread"
    assert response.json()["status"] == "queued"


def test_sse_emits_stable_frames_in_order_and_ends_at_terminal_event() -> None:
    service, registry, events = _runtime()
    asyncio.run(_seed_completed(registry, events))

    with TestClient(create_api(service)) as client:
        response = client.get("/v1/research/runs/streamed/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    assert [(frame["id"], frame["event"]) for frame in frames] == [
        ("1-0", "run.queued"),
        ("2-0", "run.started"),
        ("3-0", "workflow.progress"),
        ("4-0", "run.completed"),
    ]
    assert frames[2]["data"] == '{"stage":"retrieval","task_id":"T1"}'


def test_sse_last_event_id_resumes_exclusively_after_cursor() -> None:
    service, registry, events = _runtime()
    asyncio.run(_seed_completed(registry, events))

    with TestClient(create_api(service)) as client:
        response = client.get(
            "/v1/research/runs/streamed/events",
            headers={"Last-Event-ID": "2-0"},
        )

    assert [frame["id"] for frame in _parse_sse(response.text)] == ["3-0", "4-0"]


def test_closing_sse_observer_does_not_cancel_research_run() -> None:
    asyncio.run(_assert_disconnect_does_not_cancel())


def test_sse_waits_for_terminal_event_when_status_wins_race() -> None:
    asyncio.run(_assert_terminal_event_race_is_closed())


async def _assert_terminal_event_race_is_closed() -> None:
    class RacingService:
        def __init__(self) -> None:
            self.reads = 0

        async def read_events(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            self.reads += 1
            if self.reads == 1:
                return []
            return [
                RuntimeEvent(
                    event_id="9-0",
                    run_id="race",
                    type="run.completed",
                    data={},
                )
            ]

        async def get_run(self, run_id: str) -> RunRecord:
            return RunRecord.new(
                run_id=run_id,
                thread_id="thread-race",
                status=RunStatus.COMPLETED,
            )

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    frames = [
        frame
        async for frame in _event_stream(
            RacingService(),  # type: ignore[arg-type]
            ConnectedRequest(),  # type: ignore[arg-type]
            "race",
            None,
        )
    ]

    assert len(frames) == 1
    assert "event: run.completed" in frames[0]


async def _assert_disconnect_does_not_cancel() -> None:
    class BlockingRunner:
        def __init__(self) -> None:
            self.gate = asyncio.Event()

        async def run(
            self,
            *,
            thread_id: str,
            query: str | None,
            resume: bool,
            on_progress: ProgressCallback,
        ) -> str:
            del thread_id, query, resume, on_progress
            await self.gate.wait()
            return "done"

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    runner = BlockingRunner()
    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=runner,
        policy=RuntimePolicy(infra_retry_backoff_seconds=0),
        run_id_factory=lambda: "disconnect-run",
        thread_id_factory=lambda: "disconnect-thread",
    )
    await service.create_run("query")
    while (await registry.get("disconnect-run")).status is not RunStatus.RUNNING:  # type: ignore[union-attr]
        await asyncio.sleep(0)

    observed = [
        frame
        async for frame in _event_stream(
            service,
            DisconnectedRequest(),  # type: ignore[arg-type]
            "disconnect-run",
            None,
        )
    ]

    assert observed
    assert (await registry.get("disconnect-run")).status is RunStatus.RUNNING  # type: ignore[union-attr]
    runner.gate.set()
    async with asyncio.timeout(1):
        while (await registry.get("disconnect-run")).status is not RunStatus.COMPLETED:  # type: ignore[union-attr]
            await asyncio.sleep(0)
    await service.close()


async def _seed_completed(
    registry: InMemoryRunRegistry,
    events: InMemoryRuntimeEventStore,
) -> None:
    await registry.create(
        RunRecord.new(
            run_id="streamed",
            thread_id="thread-streamed",
            status=RunStatus.COMPLETED,
        )
    )
    for event_type, data in (
        ("run.queued", {}),
        ("run.started", {}),
        ("workflow.progress", {"stage": "retrieval", "task_id": "T1"}),
        ("run.completed", {}),
    ):
        await events.append(
            RuntimeEvent(run_id="streamed", type=event_type, data=data)
        )


def _parse_sse(payload: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for raw_frame in payload.strip().split("\n\n"):
        frame: dict[str, str] = {}
        for line in raw_frame.splitlines():
            name, value = line.split(": ", 1)
            frame[name] = value
        frames.append(frame)
    return frames
