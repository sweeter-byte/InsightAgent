from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from insight_agent.runtime.events import RedisRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus, RuntimeEvent
from insight_agent.runtime.registry import RedisRunRegistry


@pytest.mark.asyncio
async def test_real_redis_registry_and_stream_round_trip() -> None:
    url = os.getenv("RUNTIME_TEST_REDIS_URL")
    if not url:
        pytest.skip("RUNTIME_TEST_REDIS_URL is not configured")
    namespace = f"insight-agent-test:{uuid4()}:"
    client = Redis.from_url(url)
    try:
        await client.ping()
        registry = RedisRunRegistry(client, key_prefix=f"{namespace}run:")
        events = RedisRuntimeEventStore(
            client,
            event_ttl_seconds=60,
            key_prefix=f"{namespace}events:",
        )
        record = RunRecord.new(run_id="run", thread_id="thread")

        await registry.create(record)
        started = await registry.update_status("run", RunStatus.RUNNING)
        first_id = await events.append(
            RuntimeEvent(run_id="run", type="run.started", data={})
        )
        second_id = await events.append(
            RuntimeEvent(run_id="run", type="run.completed", data={})
        )

        assert (await registry.get("run")) == started
        assert [event.event_id for event in await events.read_after("run", first_id)] == [
            second_id
        ]
        assert await registry.list_by_status({RunStatus.RUNNING}) == [started]
    finally:
        await client.aclose()
