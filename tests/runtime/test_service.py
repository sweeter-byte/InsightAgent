from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import threading

import pytest

from insight_agent.runtime.errors import InvalidRunState, TransientRuntimeError
from insight_agent.runtime.events import InMemoryRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import InMemoryRunRegistry
from insight_agent.runtime.runner import LangGraphResearchRunner
from insight_agent.runtime.service import ResearchRuntimeService


ProgressCallback = Callable[[str, dict[str, object]], Awaitable[None]]


class ControlledRunner:
    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.calls: list[tuple[str, str | None, bool]] = []
        self.active = 0
        self.max_active = 0
        self.failures: set[str] = set()

    async def run(
        self,
        *,
        thread_id: str,
        query: str | None,
        resume: bool,
        on_progress: ProgressCallback,
    ) -> str:
        self.calls.append((thread_id, query, resume))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await on_progress("retrieval", {"task_id": "T1"})
            await self.gates.setdefault(thread_id, asyncio.Event()).wait()
            if thread_id in self.failures:
                raise RuntimeError("secret internal failure details")
            return f"result for {thread_id}"
        finally:
            self.active -= 1


@dataclass
class _SyncState:
    final_output: str


@dataclass
class _SyncSnapshot:
    state: _SyncState
    completed: bool = True


class BlockingSyncCoordinator:
    """Expose the non-cancellable behavior of a real sync workflow thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gates: dict[tuple[str, int], threading.Event] = {}
        self._started: dict[tuple[str, int], threading.Event] = {}
        self._finished: dict[tuple[str, int], threading.Event] = {}
        self._counts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0

    def start_research(self, query: str, *, thread_id: str) -> _SyncSnapshot:
        del query
        return self._execute(thread_id)

    def resume_research(self, thread_id: str) -> _SyncSnapshot:
        return self._execute(thread_id)

    def _execute(self, thread_id: str) -> _SyncSnapshot:
        with self._lock:
            occurrence = self._counts.get(thread_id, 0) + 1
            self._counts[thread_id] = occurrence
            key = (thread_id, occurrence)
            gate = self._gates.setdefault(key, threading.Event())
            started = self._started.setdefault(key, threading.Event())
            finished = self._finished.setdefault(key, threading.Event())
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            started.set()
        try:
            gate.wait()
            return _SyncSnapshot(_SyncState(f"result:{thread_id}:{occurrence}"))
        finally:
            with self._lock:
                self.active -= 1
                finished.set()

    def release(self, thread_id: str, occurrence: int = 1) -> None:
        with self._lock:
            self._gates.setdefault(
                (thread_id, occurrence), threading.Event()
            ).set()

    def release_all(self) -> None:
        with self._lock:
            for gate in self._gates.values():
                gate.set()

    def has_started(self, thread_id: str, occurrence: int = 1) -> bool:
        with self._lock:
            event = self._started.get((thread_id, occurrence))
            return event is not None and event.is_set()

    def has_finished(self, thread_id: str, occurrence: int = 1) -> bool:
        with self._lock:
            event = self._finished.get((thread_id, occurrence))
            return event is not None and event.is_set()

    def call_count(self, thread_id: str) -> int:
        with self._lock:
            return self._counts.get(thread_id, 0)


async def _event_types(
    events: InMemoryRuntimeEventStore, run_id: str
) -> list[str]:
    return [event.type for event in await events.read_after(run_id, None)]


async def _wait_for_status(
    registry: InMemoryRunRegistry,
    run_id: str,
    expected: RunStatus,
) -> RunRecord:
    async with asyncio.timeout(1):
        while True:
            record = await registry.get(run_id)
            assert record is not None
            if record.status is expected:
                return record
            await asyncio.sleep(0)


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


def _service(
    runner: ControlledRunner,
    *,
    concurrency: int = 2,
    timeout: float = 1,
) -> tuple[ResearchRuntimeService, InMemoryRunRegistry, InMemoryRuntimeEventStore]:
    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    ids = iter(f"run-{index}" for index in range(10))
    threads = iter(f"thread-{index}" for index in range(10))
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=runner,
        policy=RuntimePolicy(
            max_concurrency=concurrency,
            run_timeout_seconds=timeout,
            infra_retry_backoff_seconds=0,
        ),
        run_id_factory=lambda: next(ids),
        thread_id_factory=lambda: next(threads),
    )
    return service, registry, events


@pytest.mark.asyncio
async def test_create_persists_queued_record_and_event_before_runner_starts() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)

    record = await service.create_run("query")

    assert record.status is RunStatus.QUEUED
    assert (await registry.get(record.run_id)) is not None
    assert await _event_types(events, record.run_id) == ["run.queued"]
    assert runner.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_run_completes_and_preserves_sanitized_progress() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)
    record = await service.create_run("query")
    await _wait_for_status(registry, record.run_id, RunStatus.RUNNING)
    await _wait_until(lambda: record.thread_id in runner.gates)
    runner.gates[record.thread_id].set()

    completed = await _wait_for_status(registry, record.run_id, RunStatus.COMPLETED)

    assert completed.final_output == f"result for {record.thread_id}"
    assert await _event_types(events, record.run_id) == [
        "run.queued",
        "run.started",
        "workflow.progress",
        "run.completed",
    ]
    await service.close()


@pytest.mark.asyncio
async def test_runner_failure_is_not_retried_and_error_is_safe() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)
    record = await service.create_run("query")
    await _wait_for_status(registry, record.run_id, RunStatus.RUNNING)
    await _wait_until(lambda: record.thread_id in runner.gates)
    runner.failures.add(record.thread_id)
    runner.gates[record.thread_id].set()

    failed = await _wait_for_status(registry, record.run_id, RunStatus.FAILED)

    assert failed.error == "RuntimeError"
    assert "secret" not in failed.error
    assert len(runner.calls) == 1
    assert (await _event_types(events, record.run_id))[-1] == "run.failed"
    await service.close()


@pytest.mark.asyncio
async def test_timeout_preserves_thread_and_releases_slot_after_execution_exits() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner, concurrency=1, timeout=0.02)
    first = await service.create_run("first")
    second = await service.create_run("second")

    timed_out = await _wait_for_status(registry, first.run_id, RunStatus.TIMED_OUT)
    await asyncio.sleep(0.01)
    queued = await registry.get(second.run_id)

    assert queued is not None and queued.status is RunStatus.QUEUED
    assert runner.active == 1
    runner.gates[first.thread_id].set()
    await _wait_for_status(registry, second.run_id, RunStatus.RUNNING)

    assert timed_out.thread_id == first.thread_id
    assert (await _event_types(events, first.run_id))[-1] == "run.timed_out"
    await _wait_until(lambda: second.thread_id in runner.gates)
    runner.gates[second.thread_id].set()
    await service.close()


@pytest.mark.asyncio
async def test_timed_out_sync_workflow_holds_real_concurrency_slot_until_exit() -> None:
    coordinator = BlockingSyncCoordinator()
    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    thread_ids = iter(["thread-a", "thread-b"])
    run_ids = iter(["run-a", "run-b"])
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=LangGraphResearchRunner(coordinator),  # type: ignore[arg-type]
        policy=RuntimePolicy(
            max_concurrency=1,
            run_timeout_seconds=0.02,
            infra_retry_backoff_seconds=0,
        ),
        run_id_factory=lambda: next(run_ids),
        thread_id_factory=lambda: next(thread_ids),
    )
    try:
        first = await service.create_run("first")
        await _wait_until(lambda: coordinator.has_started(first.thread_id))
        await _wait_for_status(registry, first.run_id, RunStatus.TIMED_OUT)

        second = await service.create_run("second")
        await asyncio.sleep(0.05)

        assert coordinator.has_started(second.thread_id) is False
        assert coordinator.max_active == 1

        coordinator.release(first.thread_id)
        await _wait_until(lambda: coordinator.has_started(second.thread_id))
        assert coordinator.max_active == 1
        coordinator.release(second.thread_id)
        await _wait_for_status(registry, second.run_id, RunStatus.COMPLETED)
    finally:
        coordinator.release_all()
        await service.close()


@pytest.mark.asyncio
async def test_resume_is_rejected_while_timed_out_sync_workflow_is_alive() -> None:
    coordinator = BlockingSyncCoordinator()
    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=LangGraphResearchRunner(coordinator),  # type: ignore[arg-type]
        policy=RuntimePolicy(
            max_concurrency=2,
            run_timeout_seconds=0.02,
            infra_retry_backoff_seconds=0,
        ),
        run_id_factory=lambda: "run-a",
        thread_id_factory=lambda: "thread-a",
    )
    try:
        record = await service.create_run("query")
        await _wait_until(lambda: coordinator.has_started(record.thread_id))
        await _wait_for_status(registry, record.run_id, RunStatus.TIMED_OUT)

        with pytest.raises(InvalidRunState, match="still executing locally"):
            await service.resume_run(record.run_id)

        assert coordinator.call_count(record.thread_id) == 1
        coordinator.release(record.thread_id, 1)
        await _wait_until(lambda: coordinator.has_finished(record.thread_id, 1))
        await asyncio.sleep(0)

        resumed = await service.resume_run(record.run_id)
        assert resumed.thread_id == record.thread_id
        await _wait_until(
            lambda: coordinator.has_started(record.thread_id, occurrence=2)
        )
        assert coordinator.max_active == 1
        coordinator.release(record.thread_id, 2)
        await _wait_for_status(registry, record.run_id, RunStatus.COMPLETED)
    finally:
        coordinator.release_all()
        await service.close()


@pytest.mark.asyncio
async def test_semaphore_limits_actual_concurrent_runner_calls() -> None:
    runner = ControlledRunner()
    service, registry, _ = _service(runner, concurrency=2)
    records = [await service.create_run(str(index)) for index in range(3)]

    await _wait_for_status(registry, records[0].run_id, RunStatus.RUNNING)
    await _wait_for_status(registry, records[1].run_id, RunStatus.RUNNING)
    await _wait_until(
        lambda: all(record.thread_id in runner.gates for record in records[:2])
    )
    third = await registry.get(records[2].run_id)

    assert third is not None and third.status is RunStatus.QUEUED
    assert runner.max_active == 2
    runner.gates[records[0].thread_id].set()
    await _wait_for_status(registry, records[2].run_id, RunStatus.RUNNING)
    await _wait_until(lambda: records[2].thread_id in runner.gates)
    assert runner.max_active == 2
    runner.gates[records[1].thread_id].set()
    runner.gates[records[2].thread_id].set()
    await service.close()


@pytest.mark.asyncio
async def test_parallel_runs_keep_status_errors_and_events_isolated() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)
    first = await service.create_run("first")
    second = await service.create_run("second")
    await _wait_for_status(registry, first.run_id, RunStatus.RUNNING)
    await _wait_for_status(registry, second.run_id, RunStatus.RUNNING)
    await _wait_until(
        lambda: first.thread_id in runner.gates
        and second.thread_id in runner.gates
    )
    runner.failures.add(first.thread_id)
    runner.gates[first.thread_id].set()
    runner.gates[second.thread_id].set()

    failed = await _wait_for_status(registry, first.run_id, RunStatus.FAILED)
    completed = await _wait_for_status(registry, second.run_id, RunStatus.COMPLETED)

    assert first.run_id != second.run_id
    assert first.thread_id != second.thread_id
    assert failed.error == "RuntimeError"
    assert completed.error is None
    assert "run.failed" in await _event_types(events, first.run_id)
    assert "run.failed" not in await _event_types(events, second.run_id)
    assert all(
        event.run_id == first.run_id
        for event in await events.read_after(first.run_id, None)
    )
    await service.close()


@pytest.mark.asyncio
async def test_resume_reuses_original_thread_without_initial_query() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)
    original = RunRecord.new(
        run_id="existing-run",
        thread_id="existing-thread",
        status=RunStatus.INTERRUPTED,
    )
    await registry.create(original)

    resumed = await service.resume_run(original.run_id)
    await _wait_for_status(registry, original.run_id, RunStatus.RUNNING)
    await _wait_until(lambda: bool(runner.calls))

    assert resumed.run_id == original.run_id
    assert resumed.thread_id == original.thread_id
    assert runner.calls == [("existing-thread", None, True)]
    assert (await _event_types(events, original.run_id))[:2] == [
        "run.resuming",
        "run.started",
    ]
    runner.gates[original.thread_id].set()
    await service.close()


@pytest.mark.asyncio
async def test_concurrent_resume_requests_schedule_workflow_only_once() -> None:
    runner = ControlledRunner()
    service, registry, _ = _service(runner)
    await registry.create(
        RunRecord.new(
            run_id="existing-run",
            thread_id="existing-thread",
            status=RunStatus.INTERRUPTED,
        )
    )

    results = await asyncio.gather(
        service.resume_run("existing-run"),
        service.resume_run("existing-run"),
        return_exceptions=True,
    )
    await _wait_for_status(registry, "existing-run", RunStatus.RUNNING)

    assert sum(isinstance(result, InvalidRunState) for result in results) == 1
    assert runner.calls == [("existing-thread", None, True)]
    runner.gates["existing-thread"].set()
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [RunStatus.COMPLETED, RunStatus.RUNNING, RunStatus.QUEUED, RunStatus.FAILED]
)
async def test_resume_rejects_non_resumable_states(status: RunStatus) -> None:
    runner = ControlledRunner()
    service, registry, _ = _service(runner)
    await registry.create(
        RunRecord.new(run_id="existing", thread_id="thread", status=status)
    )

    with pytest.raises(InvalidRunState):
        await service.resume_run("existing")

    assert runner.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_startup_marks_only_running_records_interrupted() -> None:
    runner = ControlledRunner()
    service, registry, events = _service(runner)
    for run_id, status in (
        ("r1", RunStatus.RUNNING),
        ("r2", RunStatus.COMPLETED),
        ("r3", RunStatus.RUNNING),
    ):
        await registry.create(
            RunRecord.new(run_id=run_id, thread_id=f"t-{run_id}", status=status)
        )

    interrupted = await service.reconcile_interrupted_runs()

    assert interrupted == 2
    assert (await registry.get("r1")).status is RunStatus.INTERRUPTED  # type: ignore[union-attr]
    assert (await registry.get("r2")).status is RunStatus.COMPLETED  # type: ignore[union-attr]
    assert (await registry.get("r3")).status is RunStatus.INTERRUPTED  # type: ignore[union-attr]
    assert await _event_types(events, "r1") == ["run.interrupted"]
    assert runner.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_transient_registry_failure_is_retried_without_retrying_runner() -> None:
    class FlakyRegistry(InMemoryRunRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.create_attempts = 0

        async def create(self, record: RunRecord) -> None:
            self.create_attempts += 1
            if self.create_attempts == 1:
                raise TransientRuntimeError("temporary")
            await super().create(record)

    runner = ControlledRunner()
    registry = FlakyRegistry()
    events = InMemoryRuntimeEventStore()
    service = ResearchRuntimeService(
        registry=registry,
        events=events,
        runner=runner,
        policy=RuntimePolicy(infra_retry_attempts=2, infra_retry_backoff_seconds=0),
        run_id_factory=lambda: "run",
        thread_id_factory=lambda: "thread",
    )

    await service.create_run("query")

    assert registry.create_attempts == 2
    assert runner.calls == []
    await service.close()


@pytest.mark.asyncio
async def test_lifecycle_logs_have_structured_identifiers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = ControlledRunner()
    service, registry, _ = _service(runner)
    caplog.set_level("INFO", logger="insight_agent.runtime.service")

    record = await service.create_run("query", request_id="request-1")
    await _wait_for_status(registry, record.run_id, RunStatus.RUNNING)
    await _wait_until(lambda: record.thread_id in runner.gates)

    assert {item.msg for item in caplog.records} >= {
        "run created",
        "run queued",
        "run started",
    }

    queued_log = next(item for item in caplog.records if item.msg == "run queued")
    assert queued_log.request_id == "request-1"  # type: ignore[attr-defined]
    assert queued_log.run_id == record.run_id  # type: ignore[attr-defined]
    assert queued_log.thread_id == record.thread_id  # type: ignore[attr-defined]
    assert queued_log.status == "queued"  # type: ignore[attr-defined]
    runner.gates[record.thread_id].set()
    await _wait_for_status(registry, record.run_id, RunStatus.COMPLETED)
    assert "run completed" in {item.msg for item in caplog.records}
    await service.close()
