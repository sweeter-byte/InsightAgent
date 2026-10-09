from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

from insight_agent.runtime.app import create_runtime_app
from insight_agent.runtime.policies import RuntimeConfig, RuntimePolicy


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


def test_runtime_policy_reads_only_explicit_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_MAX_CONCURRENCY", "99")
    monkeypatch.setenv("RUNTIME_EVENT_TTL_SECONDS", "999")

    policy = RuntimePolicy.from_env(
        {
            "RUNTIME_MAX_CONCURRENCY": "4",
            "RUNTIME_RUN_TIMEOUT_SECONDS": "12.5",
            "RUNTIME_INFRA_RETRY_ATTEMPTS": "5",
            "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS": "0.75",
            "RUNTIME_EVENT_TTL_SECONDS": "123",
        }
    )

    assert policy == RuntimePolicy(
        max_concurrency=4,
        run_timeout_seconds=12.5,
        infra_retry_attempts=5,
        infra_retry_backoff_seconds=0.75,
        event_ttl_seconds=123,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RUNTIME_RUN_TIMEOUT_SECONDS", "nan"),
        ("RUNTIME_RUN_TIMEOUT_SECONDS", "inf"),
        ("RUNTIME_INFRA_RETRY_BACKOFF_SECONDS", "nan"),
        ("RUNTIME_INFRA_RETRY_BACKOFF_SECONDS", "inf"),
    ],
)
def test_runtime_policy_mapping_rejects_non_finite_floats(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=name):
        RuntimePolicy.from_env({name: value})


@pytest.mark.parametrize(
    "name",
    [
        "RUNTIME_MAX_CONCURRENCY",
        "RUNTIME_RUN_TIMEOUT_SECONDS",
        "RUNTIME_INFRA_RETRY_ATTEMPTS",
        "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS",
        "RUNTIME_EVENT_TTL_SECONDS",
    ],
)
def test_runtime_policy_mapping_rejects_boolean_numeric_values(
    name: str,
) -> None:
    with pytest.raises(ValueError, match=name):
        RuntimePolicy.from_env({name: True})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_timeout_seconds", float("nan")),
        ("run_timeout_seconds", float("inf")),
        ("infra_retry_backoff_seconds", float("nan")),
        ("infra_retry_backoff_seconds", float("inf")),
    ],
)
def test_runtime_policy_direct_construction_rejects_non_finite_floats(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        RuntimePolicy(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "max_concurrency",
        "run_timeout_seconds",
        "infra_retry_attempts",
        "infra_retry_backoff_seconds",
        "event_ttl_seconds",
    ],
)
def test_runtime_policy_direct_construction_rejects_booleans(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RuntimePolicy(**{field: True})


def test_runtime_policy_accepts_valid_numeric_values_and_zero_backoff() -> None:
    policy = RuntimePolicy(
        max_concurrency=1,
        run_timeout_seconds=1,
        infra_retry_attempts=1,
        infra_retry_backoff_seconds=0,
        event_ttl_seconds=1,
    )

    assert policy.run_timeout_seconds == 1
    assert policy.infra_retry_backoff_seconds == 0


def test_runtime_config_reads_explicit_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_REDIS_URL", "redis://ignored.example/0")
    monkeypatch.setenv("RUNTIME_MAX_CONCURRENCY", "99")

    config = RuntimeConfig.from_env(
        {
            "RUNTIME_REDIS_URL": " redis://runtime.example/8 ",
            "RUNTIME_MAX_CONCURRENCY": "4",
            "RUNTIME_RUN_TIMEOUT_SECONDS": "12.5",
        }
    )

    assert config.redis_url == "redis://runtime.example/8"
    assert config.policy.max_concurrency == 4
    assert config.policy.run_timeout_seconds == 12.5


def test_runtime_config_rejects_empty_direct_redis_url() -> None:
    with pytest.raises(ValueError, match="redis_url"):
        RuntimeConfig(redis_url="   ")


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
