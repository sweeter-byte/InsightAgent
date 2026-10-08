"""Run registry contracts and an in-memory implementation for tests."""

from __future__ import annotations

import asyncio
from collections.abc import Set
from typing import Any, Protocol

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from insight_agent.runtime.errors import (
    DuplicateRun,
    RunNotFound,
    TransientRuntimeError,
)
from insight_agent.runtime.models import RunRecord, RunStatus, utc_now


class RunRegistry(Protocol):
    async def create(self, record: RunRecord) -> None: ...

    async def get(self, run_id: str) -> RunRecord | None: ...

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        final_output: str | None = None,
    ) -> RunRecord: ...

    async def list_by_status(self, statuses: Set[RunStatus]) -> list[RunRecord]: ...


class InMemoryRunRegistry:
    """Concurrency-safe process-local registry used by deterministic tests."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: RunRecord) -> None:
        async with self._lock:
            if record.run_id in self._records:
                raise DuplicateRun(f"run already exists: {record.run_id}")
            self._records[record.run_id] = record

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            return self._records.get(run_id)

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        final_output: str | None = None,
    ) -> RunRecord:
        async with self._lock:
            current = self._records.get(run_id)
            if current is None:
                raise RunNotFound(f"unknown run: {run_id}")
            updated = current.model_copy(
                update={
                    "status": status,
                    "updated_at": utc_now(),
                    "error": error,
                    "final_output": final_output,
                }
            )
            self._records[run_id] = updated
            return updated

    async def list_by_status(self, statuses: Set[RunStatus]) -> list[RunRecord]:
        async with self._lock:
            return [
                record
                for record in self._records.values()
                if record.status in statuses
            ]


class RedisRunRegistry:
    """Redis-backed runtime metadata without exposing Redis above this layer."""

    def __init__(self, redis_client: Any, *, key_prefix: str = "runtime:run:") -> None:
        self._redis = redis_client
        self._key_prefix = key_prefix

    async def create(self, record: RunRecord) -> None:
        try:
            created = await self._redis.set(
                self._key(record.run_id),
                record.model_dump_json(),
                nx=True,
            )
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis run registry unavailable") from exc
        if not created:
            raise DuplicateRun(f"run already exists: {record.run_id}")

    async def get(self, run_id: str) -> RunRecord | None:
        try:
            payload = await self._redis.get(self._key(run_id))
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis run registry unavailable") from exc
        if payload is None:
            return None
        return RunRecord.model_validate_json(payload)

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        final_output: str | None = None,
    ) -> RunRecord:
        current = await self.get(run_id)
        if current is None:
            raise RunNotFound(f"unknown run: {run_id}")
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": utc_now(),
                "error": error,
                "final_output": final_output,
            }
        )
        try:
            await self._redis.set(self._key(run_id), updated.model_dump_json())
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis run registry unavailable") from exc
        return updated

    async def list_by_status(self, statuses: Set[RunStatus]) -> list[RunRecord]:
        records: list[RunRecord] = []
        try:
            async for key in self._redis.scan_iter(match=f"{self._key_prefix}*"):
                normalized_key = key.decode() if isinstance(key, bytes) else key
                payload = await self._redis.get(normalized_key)
                if payload is None:
                    continue
                record = RunRecord.model_validate_json(payload)
                if record.status in statuses:
                    records.append(record)
        except (RedisConnectionError, RedisTimeoutError) as exc:
            raise TransientRuntimeError("Redis run registry unavailable") from exc
        return records

    def _key(self, run_id: str) -> str:
        return f"{self._key_prefix}{run_id}"
