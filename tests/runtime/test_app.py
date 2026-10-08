from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

from insight_agent.runtime.app import create_runtime_app
from insight_agent.runtime.policies import RuntimePolicy


class FakeRedis:
    def __init__(self, *, fail_ping: bool = False) -> None:
        self.ping_calls = 0
        self.close_calls = 0
        self.fail_ping = fail_ping

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self.fail_ping:
            raise RuntimeError("redis unavailable")
        return True

    async def scan_iter(self, *, match: str) -> AsyncIterator[bytes]:
        del match
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1


class FakeResearchApp:
    def __init__(self) -> None:
        self.research_coordinator = object()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_runtime_policy_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("RUNTIME_RUN_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("RUNTIME_EVENT_TTL_SECONDS", "123")

    policy = RuntimePolicy.from_env()

    assert policy.max_concurrency == 4
    assert policy.run_timeout_seconds == 12.5
    assert policy.event_ttl_seconds == 123


def test_lifespan_builds_shared_resources_and_closes_them() -> None:
    redis = FakeRedis()
    research_app = FakeResearchApp()
    urls: list[str] = []

    app = create_runtime_app(
        research_app_factory=lambda: research_app,  # type: ignore[arg-type]
        redis_client_factory=lambda url: (urls.append(url), redis)[1],  # type: ignore[arg-type]
        redis_url="redis://runtime.example/4",
        policy=RuntimePolicy(infra_retry_backoff_seconds=0),
    )

    with TestClient(app):
        assert app.state.runtime_service is not None
        assert app.state.research_app is research_app
        assert redis.ping_calls == 1

    assert urls == ["redis://runtime.example/4"]
    assert redis.close_calls == 1
    assert research_app.close_calls == 1


def test_lifespan_closes_resources_when_startup_fails() -> None:
    redis = FakeRedis(fail_ping=True)
    research_app = FakeResearchApp()
    app = create_runtime_app(
        research_app_factory=lambda: research_app,  # type: ignore[arg-type]
        redis_client_factory=lambda _url: redis,  # type: ignore[arg-type]
        policy=RuntimePolicy(infra_retry_backoff_seconds=0),
    )

    with pytest.raises(RuntimeError, match="redis unavailable"):
        with TestClient(app):
            pass

    assert redis.close_calls == 1
    assert research_app.close_calls == 1
