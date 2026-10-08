"""Runtime event-store contracts and deterministic in-memory behavior."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from insight_agent.runtime.errors import TransientRuntimeError
from insight_agent.runtime.models import RuntimeEvent


class RuntimeEventStore(Protocol):
    async def append(self, event: RuntimeEvent) -> str: ...

    async def read_after(
        self,
        run_id: str,
        event_id: str | None,
        *,
        block_ms: int = 0,
    ) -> list[RuntimeEvent]: ...


class InMemoryRuntimeEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[RuntimeEvent]] = {}
        self._counter = 0
        self._condition = asyncio.Condition()

    async def append(self, event: RuntimeEvent) -> str:
        async with self._condition:
            self._counter += 1
            event_id = f"{self._counter}-0"
            stored = event.model_copy(update={"event_id": event_id})
            self._events.setdefault(event.run_id, []).append(stored)
            self._condition.notify_all()
            return event_id

    async def read_after(
        self,
        run_id: str,
        event_id: str | None,
        *,
        block_ms: int = 0,
    ) -> list[RuntimeEvent]:
        async with self._condition:
            events = self._read(run_id, event_id)
            if events or block_ms <= 0:
                return events
            try:
                async with asyncio.timeout(block_ms / 1000):
                    await self._condition.wait()
            except TimeoutError:
                return []
            return self._read(run_id, event_id)

    def _read(self, run_id: str, event_id: str | None) -> list[RuntimeEvent]:
        events = list(self._events.get(run_id, []))
        if event_id is None:
            return events
        return [event for event in events if _stream_id_gt(event.event_id, event_id)]


def _stream_id_gt(candidate: str | None, baseline: str) -> bool:
    if candidate is None:
        return False
    candidate_parts = tuple(int(part) for part in candidate.split("-", 1))
    baseline_parts = tuple(int(part) for part in baseline.split("-", 1))
    return candidate_parts > baseline_parts


class RedisRuntimeEventStore:
    """One durable Redis Stream per public research run."""

    def __init__(
        self,
        redis_client: Any,
        *,
        event_ttl_seconds: int,
        stream_maxlen: int = 1_000,
        key_prefix: str = "runtime:events:",
    ) -> None:
        self._redis = redis_client
        self._event_ttl_seconds = event_ttl_seconds
        self._stream_maxlen = stream_maxlen
        self._key_prefix = key_prefix

    async def append(self, event: RuntimeEvent) -> str:
        key = self._key(event.run_id)
        payload = event.model_dump_json(exclude={"event_id"})
        try:
            event_id = await self._redis.xadd(
                key,
                {"event": payload},
                maxlen=self._stream_maxlen,
                approximate=True,
            )
            await self._redis.expire(key, self._event_ttl_seconds)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis event store unavailable") from exc
        return _text(event_id)

    async def read_after(
        self,
        run_id: str,
        event_id: str | None,
        *,
        block_ms: int = 0,
    ) -> list[RuntimeEvent]:
        key = self._key(run_id)
        try:
            if block_ms > 0:
                response = await self._redis.xread(
                    {key: event_id or "0-0"},
                    block=block_ms,
                )
                rows = response[0][1] if response else []
            else:
                rows = await self._redis.xrange(
                    key,
                    min=f"({event_id}" if event_id else "-",
                    max="+",
                )
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis event store unavailable") from exc
        return [_event_from_stream(row) for row in rows]

    def _key(self, run_id: str) -> str:
        return f"{self._key_prefix}{run_id}"


def _event_from_stream(row: tuple[Any, dict[Any, Any]]) -> RuntimeEvent:
    event_id, fields = row
    payload = fields.get(b"event", fields.get("event"))
    if payload is None:
        raise ValueError("runtime event stream entry is missing event payload")
    event = RuntimeEvent.model_validate_json(payload)
    return event.model_copy(update={"event_id": _text(event_id)})


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value
