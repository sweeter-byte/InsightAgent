from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from insight_agent.runtime.events import RedisRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus, RuntimeEvent
from insight_agent.runtime.registry import RedisRunRegistry


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.streams: dict[str, list[tuple[bytes, dict[bytes, bytes]]]] = {}
        self.expirations: list[tuple[str, int]] = []
        self.xread_calls: list[tuple[dict[str, str], int | None]] = []

    async def set(self, key: str, value: str, *, nx: bool = False) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value.encode()
        return True

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def xadd(
        self,
        key: str,
        fields: dict[str, str],
        *,
        maxlen: int,
        approximate: bool,
    ) -> bytes:
        del maxlen, approximate
        event_id = f"{len(self.streams.get(key, [])) + 1}-0".encode()
        encoded = {name.encode(): value.encode() for name, value in fields.items()}
        self.streams.setdefault(key, []).append((event_id, encoded))
        return event_id

    async def expire(self, key: str, ttl: int) -> bool:
        self.expirations.append((key, ttl))
        return True

    async def xrange(
        self, key: str, *, min: str, max: str
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        del max
        events = self.streams.get(key, [])
        if min == "-":
            return list(events)
        baseline = min.removeprefix("(")
        return [row for row in events if _id(row[0]) > _id(baseline)]

    async def xread(
        self,
        streams: dict[str, str],
        *,
        block: int | None,
    ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]:
        self.xread_calls.append((streams, block))
        key, baseline = next(iter(streams.items()))
        rows = [
            row
            for row in self.streams.get(key, [])
            if _id(row[0]) > _id(baseline)
        ]
        return [(key.encode(), rows)] if rows else []

    async def scan_iter(self, *, match: str) -> AsyncIterator[bytes]:
        prefix = match.removesuffix("*")
        for key in self.values:
            if key.startswith(prefix):
                yield key.encode()


def _id(value: bytes | str) -> tuple[int, int]:
    if isinstance(value, bytes):
        value = value.decode()
    return tuple(int(part) for part in value.split("-", 1))  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_redis_registry_round_trips_records_and_filters_status() -> None:
    redis = FakeRedis()
    registry = RedisRunRegistry(redis)  # type: ignore[arg-type]
    queued = RunRecord.new(run_id="r1", thread_id="t1")
    completed = RunRecord.new(
        run_id="r2", thread_id="t2", status=RunStatus.COMPLETED
    )
    await registry.create(queued)
    await registry.create(completed)

    updated = await registry.update_status(
        "r1", RunStatus.COMPLETED, final_output="result"
    )

    assert await registry.get("r1") == updated
    assert updated.final_output == "result"
    assert {record.run_id for record in await registry.list_by_status({RunStatus.COMPLETED})} == {
        "r1",
        "r2",
    }
    assert set(redis.values) == {"runtime:run:r1", "runtime:run:r2"}


@pytest.mark.asyncio
async def test_redis_event_store_uses_stream_id_and_ttl() -> None:
    redis = FakeRedis()
    store = RedisRuntimeEventStore(redis, event_ttl_seconds=123)  # type: ignore[arg-type]

    first_id = await store.append(
        RuntimeEvent(run_id="r1", type="run.queued", data={"safe": True})
    )
    second_id = await store.append(
        RuntimeEvent(run_id="r1", type="run.completed", data={})
    )
    events = await store.read_after("r1", first_id)

    assert first_id == "1-0"
    assert second_id == "2-0"
    assert [(event.event_id, event.type) for event in events] == [
        ("2-0", "run.completed")
    ]
    assert redis.expirations == [
        ("runtime:events:r1", 123),
        ("runtime:events:r1", 123),
    ]


@pytest.mark.asyncio
async def test_redis_event_store_uses_xread_for_blocking_reconnect() -> None:
    redis = FakeRedis()
    store = RedisRuntimeEventStore(redis, event_ttl_seconds=60)  # type: ignore[arg-type]
    await store.append(RuntimeEvent(run_id="r1", type="run.queued", data={}))

    events = await store.read_after("r1", "0-0", block_ms=250)

    assert [event.type for event in events] == ["run.queued"]
    assert redis.xread_calls == [({"runtime:events:r1": "0-0"}, 250)]
