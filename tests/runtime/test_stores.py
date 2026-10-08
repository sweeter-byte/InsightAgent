from __future__ import annotations

from datetime import UTC, datetime

import pytest

from insight_agent.runtime.events import InMemoryRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus, RuntimeEvent
from insight_agent.runtime.registry import InMemoryRunRegistry


def _record(run_id: str = "run-1") -> RunRecord:
    now = datetime.now(UTC)
    return RunRecord(
        run_id=run_id,
        thread_id=f"thread-{run_id}",
        status=RunStatus.QUEUED,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_in_memory_registry_creates_reads_and_updates_records() -> None:
    registry = InMemoryRunRegistry()
    original = _record()

    await registry.create(original)
    updated = await registry.update_status(
        original.run_id,
        RunStatus.COMPLETED,
        final_output="# result",
    )

    assert await registry.get(original.run_id) == updated
    assert updated.status is RunStatus.COMPLETED
    assert updated.final_output == "# result"
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at
    assert await registry.list_by_status({RunStatus.COMPLETED}) == [updated]


@pytest.mark.asyncio
async def test_in_memory_event_store_reads_exclusively_after_event_id() -> None:
    store = InMemoryRuntimeEventStore()
    first = RuntimeEvent(run_id="run-1", type="run.queued", data={})
    second = RuntimeEvent(run_id="run-1", type="run.started", data={})

    first_id = await store.append(first)
    second_id = await store.append(second)

    assert [event.event_id for event in await store.read_after("run-1", None)] == [
        first_id,
        second_id,
    ]
    assert [
        event.event_id
        for event in await store.read_after("run-1", first_id)
    ] == [second_id]


@pytest.mark.asyncio
async def test_in_memory_event_streams_are_isolated() -> None:
    store = InMemoryRuntimeEventStore()
    await store.append(RuntimeEvent(run_id="run-a", type="run.queued", data={}))
    await store.append(RuntimeEvent(run_id="run-b", type="run.failed", data={}))

    events = await store.read_after("run-a", None)

    assert [event.type for event in events] == ["run.queued"]
