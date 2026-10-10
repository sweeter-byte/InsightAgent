from __future__ import annotations

import pytest

from insight_agent.application.readiness import ReadinessService


class Redis:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def ping(self) -> bool:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return True


class Probe:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def check_ready(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
async def test_readiness_reports_required_and_optional_components() -> None:
    redis = Redis()
    qdrant = Probe()
    checkpoint = Probe()
    service = ReadinessService(
        redis=redis,
        qdrant=qdrant,
        checkpoint=checkpoint,
        web_available=False,
        vision_available=True,
    )

    result = await service.check()

    assert result.status.value == "ready"
    assert {name: item.status.value for name, item in result.required.items()} == {
        "redis": "ready",
        "qdrant": "ready",
        "checkpoint": "ready",
        "config": "ready",
    }
    assert {name: item.status.value for name, item in result.optional.items()} == {
        "web": "unconfigured",
        "vision": "available",
    }
    assert (redis.calls, qdrant.calls, checkpoint.calls) == (1, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", ["redis", "qdrant", "checkpoint"])
async def test_readiness_is_not_ready_when_required_probe_fails(failed: str) -> None:
    redis = Redis(RuntimeError("secret redis detail")) if failed == "redis" else Redis()
    qdrant = Probe(RuntimeError("secret qdrant detail")) if failed == "qdrant" else Probe()
    checkpoint = (
        Probe(RuntimeError("secret checkpoint detail"))
        if failed == "checkpoint"
        else Probe()
    )
    service = ReadinessService(
        redis=redis,
        qdrant=qdrant,
        checkpoint=checkpoint,
        web_available=False,
        vision_available=False,
    )

    result = await service.check()

    assert result.status.value == "not_ready"
    assert result.required[failed].status.value == "unavailable"
    assert result.required[failed].detail == f"{failed} unavailable"
    assert "secret" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_readiness_optional_status_is_configuration_only() -> None:
    service = ReadinessService(
        redis=Redis(),
        qdrant=Probe(),
        checkpoint=Probe(),
        web_available=True,
        vision_available=True,
    )

    result = await service.check()

    assert result.status.value == "ready"
    assert result.optional["web"].status.value == "available"
    assert result.optional["vision"].status.value == "available"
