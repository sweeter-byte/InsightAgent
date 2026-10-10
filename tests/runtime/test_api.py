from __future__ import annotations

import asyncio
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from insight_agent.application import (
    ComponentReadiness,
    ComponentStatus,
    MaterialImportResult,
    ImageIngestionUnavailableError,
    MaterialProcessingError,
    MaterialStatus,
    MaterialTooLargeError,
    MaterialValidationError,
    QueryExecutionError,
    QueryResponse,
    QueryRoutingError,
    QueryRuntimeError,
    QueryStatus,
    ReadinessResult,
    ReadinessStatus,
    StoredMaterial,
)
from insight_agent.router import Intent
from insight_agent.runtime.api import _event_stream, create_api
from insight_agent.runtime.errors import RuntimeUnavailable
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


class FakeQueryService:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent
        self.calls: list[tuple[str, str]] = []

    async def query(self, query: str, *, request_id: str) -> QueryResponse:
        self.calls.append((query, request_id))
        if self.intent is Intent.RESEARCH:
            return QueryResponse(
                request_id=request_id,
                intent=self.intent,
                status=QueryStatus.QUEUED,
                run_id="run-from-query",
            )
        return QueryResponse(
            request_id=request_id,
            intent=self.intent,
            status=QueryStatus.COMPLETED,
            answer=f"{self.intent.value} answer",
        )


class FailingQueryService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def query(self, query: str, *, request_id: str) -> QueryResponse:
        del query, request_id
        raise self.error


class FakeMaterialStorage:
    def __init__(self, *, error: Exception | None = None, duplicate: bool = False) -> None:
        self.error = error
        self.duplicate = duplicate
        self.filenames: list[str | None] = []

    async def store(self, upload):
        self.filenames.append(upload.filename)
        if self.error is not None:
            raise self.error
        return StoredMaterial("material-1", Path("/controlled/material-1.txt"), self.duplicate)


class FakeKnowledgeService:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.paths: list[Path] = []

    def import_material(self, material: StoredMaterial) -> MaterialImportResult:
        self.paths.append(material.path)
        if self.error is not None:
            raise self.error
        return MaterialImportResult(
            material_id=material.material_id,
            document_ids=["document-1"],
            status=MaterialStatus.INDEXED,
            chunk_count=2,
            deduplicated=material.deduplicated,
        )


class FakeReadiness:
    def __init__(self, ready: bool) -> None:
        status = ReadinessStatus.READY if ready else ReadinessStatus.NOT_READY
        component = ComponentReadiness(
            status=ComponentStatus.READY if ready else ComponentStatus.UNAVAILABLE
        )
        self.result = ReadinessResult(
            status=status,
            required={"redis": component},
            optional={"web": ComponentReadiness(status=ComponentStatus.UNCONFIGURED)},
        )

    async def check(self) -> ReadinessResult:
        return self.result


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


def test_health_is_liveness_only() -> None:
    with TestClient(create_api()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(("ready", "status_code"), [(True, 200), (False, 503)])
def test_ready_maps_required_component_state(ready: bool, status_code: int) -> None:
    with TestClient(create_api(readiness_service=FakeReadiness(ready))) as client:
        response = client.get("/ready")

    assert response.status_code == status_code
    assert response.json()["status"] == ("ready" if ready else "not_ready")


@pytest.mark.parametrize(("duplicate", "status_code"), [(False, 201), (True, 200)])
def test_material_upload_returns_traceable_result(
    duplicate: bool, status_code: int
) -> None:
    storage = FakeMaterialStorage(duplicate=duplicate)
    knowledge = FakeKnowledgeService()
    app = create_api(material_storage=storage, knowledge_service=knowledge)

    with TestClient(app) as client:
        response = client.post(
            "/v1/materials",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == status_code
    assert response.json() == {
        "material_id": "material-1",
        "document_ids": ["document-1"],
        "status": "indexed",
        "chunk_count": 2,
        "deduplicated": duplicate,
    }
    assert storage.filenames == ["notes.txt"]
    assert knowledge.paths == [Path("/controlled/material-1.txt")]
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (MaterialValidationError("unsafe private value"), 422, "invalid material upload"),
        (MaterialTooLargeError("unsafe private value"), 413, "material upload too large"),
        (
            ImageIngestionUnavailableError("unsafe private value"),
            422,
            "image ingestion unavailable; configure VISION_*",
        ),
        (
            MaterialProcessingError("ingestion", material_id="secret-id"),
            422,
            "unable to ingest material",
        ),
        (
            MaterialProcessingError("indexing", material_id="secret-id"),
            503,
            "knowledge service unavailable",
        ),
    ],
)
def test_material_upload_errors_are_sanitized(
    error: Exception, status_code: int, detail: str
) -> None:
    if isinstance(error, MaterialProcessingError):
        storage = FakeMaterialStorage()
        knowledge = FakeKnowledgeService(error=error)
    else:
        storage = FakeMaterialStorage(error=error)
        knowledge = FakeKnowledgeService()
    app = create_api(material_storage=storage, knowledge_service=knowledge)

    with TestClient(app) as client:
        response = client.post(
            "/v1/materials",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert "unsafe" not in response.text
    assert "secret-id" not in response.text


def test_material_endpoint_accepts_only_multipart_file() -> None:
    app = create_api(
        material_storage=FakeMaterialStorage(),
        knowledge_service=FakeKnowledgeService(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/materials",
            json={"url": "http://169.254.169.254", "path": "/etc/passwd"},
        )

    assert response.status_code == 422


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


@pytest.mark.parametrize(
    ("intent", "expected_status", "expected_answer", "expected_run_id"),
    [
        (Intent.DIRECT, 200, "direct answer", None),
        (Intent.ANALYZE, 200, "analyze answer", None),
        (Intent.RESEARCH, 202, None, "run-from-query"),
    ],
)
def test_unified_query_has_stable_response_contract(
    intent: Intent,
    expected_status: int,
    expected_answer: str | None,
    expected_run_id: str | None,
) -> None:
    runtime, _, _ = _runtime()
    query_service = FakeQueryService(intent)
    with TestClient(create_api(runtime, query_service=query_service)) as client:
        response = client.post("/v1/query", json={"query": "  hello  "})

    request_id = response.headers["x-request-id"]
    assert response.status_code == expected_status
    assert response.json() == {
        "request_id": request_id,
        "intent": intent.value,
        "status": "queued" if intent is Intent.RESEARCH else "completed",
        "answer": expected_answer,
        "run_id": expected_run_id,
    }
    assert query_service.calls == [("hello", request_id)]


def test_unified_query_rejects_blank_input_before_dispatch() -> None:
    runtime, _, _ = _runtime()
    query_service = FakeQueryService(Intent.DIRECT)
    with TestClient(create_api(runtime, query_service=query_service)) as client:
        response = client.post("/v1/query", json={"query": "  "})

    assert response.status_code == 422
    assert query_service.calls == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (QueryRoutingError("secret-key-value"), 502, "unable to route query"),
        (
            QueryExecutionError("secret-key-value"),
            502,
            "unable to execute query",
        ),
        (
            QueryRuntimeError("secret-key-value"),
            503,
            "research runtime unavailable",
        ),
        (
            RuntimeUnavailable("secret-key-value"),
            503,
            "research runtime unavailable",
        ),
    ],
)
def test_unified_query_errors_are_sanitized(
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    runtime, _, _ = _runtime()
    app = create_api(
        runtime,
        query_service=FailingQueryService(error),  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"query": "hello"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert response.headers["x-request-id"]
    assert "secret-key-value" not in response.text
    assert "Traceback" not in response.text


def test_unified_query_without_composed_service_returns_503() -> None:
    runtime, _, _ = _runtime()
    with TestClient(create_api(runtime)) as client:
        response = client.post("/v1/query", json={"query": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "research runtime unavailable"}
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
