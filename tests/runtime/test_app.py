from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
import runpy

import pytest
from starlette.requests import Request

from insight_agent.application import AppConfig
import insight_agent.runtime.app as runtime_app
from insight_agent.runtime.api import _service
from insight_agent.runtime.app import create_runtime_app
from insight_agent.runtime.policies import RuntimeConfig, RuntimePolicy
from insight_agent.runtime.service import ResearchRuntimeService


class FakeRedis:
    def __init__(self, close_order: list[str]) -> None:
        self.ping_calls = 0
        self.close_calls = 0
        self.close_order = close_order
        self.fail_ping = False
        self.fail_scan = False
        self.fail_close = False

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self.fail_ping:
            raise RuntimeError("redis unavailable")
        return True

    async def scan_iter(self, *, match: str) -> AsyncIterator[bytes]:
        del match
        if self.fail_scan:
            raise RuntimeError("reconciliation failed")
        if False:
            yield b""

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_order.append("redis")
        if self.fail_close:
            raise RuntimeError("redis close failed")


class FakeResearchApp:
    def __init__(self, close_order: list[str]) -> None:
        self.research_coordinator = object()
        self.close_calls = 0
        self.close_order = close_order

    def close(self) -> None:
        self.close_calls += 1
        self.close_order.append("application")


def _config() -> AppConfig:
    return AppConfig.from_env(
        {
            "LLM_API_KEY": "test-secret",
            "LLM_BASE_URL": "https://llm.example/v1",
            "LLM_MODEL": "test-model",
            "RERANKER_MODEL": "test/reranker",
            "RUNTIME_REDIS_URL": "redis://runtime.example/4",
            "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS": "0",
        }
    )


class StartupResources:
    def __init__(self) -> None:
        self.config = _config()
        self.calls: list[tuple[str, object]] = []
        self.close_order: list[str] = []
        self.redis = FakeRedis(self.close_order)
        self.research_app = FakeResearchApp(self.close_order)

    def resolve_config(self) -> AppConfig:
        self.calls.append(("config", None))
        return self.config

    def build_application(self, config: AppConfig) -> FakeResearchApp:
        self.calls.append(("application", config))
        return self.research_app

    def create_redis(self, url: str) -> FakeRedis:
        self.calls.append(("redis", url))
        return self.redis

    def create_app(self, **overrides):
        return create_runtime_app(
            config_factory=self.resolve_config,
            application_factory=self.build_application,
            redis_client_factory=self.create_redis,
            **overrides,
        )


def _track_service_close(
    monkeypatch: pytest.MonkeyPatch,
    close_order: list[str],
    *,
    fail: bool = False,
) -> None:
    original_close = ResearchRuntimeService.close

    async def close(service: ResearchRuntimeService) -> None:
        close_order.append("service")
        await original_close(service)
        if fail:
            raise RuntimeError("service close failed")

    monkeypatch.setattr(ResearchRuntimeService, "close", close)


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


@pytest.mark.parametrize("module_level", [False, True])
def test_runtime_creation_defers_environment_and_resources(
    monkeypatch: pytest.MonkeyPatch,
    module_level: bool,
) -> None:
    def unexpected_call(*_args, **_kwargs):
        pytest.fail("runtime construction must defer config and resource calls")

    monkeypatch.setattr(AppConfig, "from_env", unexpected_call)
    monkeypatch.setattr(RuntimePolicy, "from_env", unexpected_call)
    monkeypatch.setattr("insight_agent.application.build_application", unexpected_call)
    monkeypatch.setattr(runtime_app.Redis, "from_url", unexpected_call)

    if module_level:
        namespace = runpy.run_path(runtime_app.__file__)
        assert namespace["app"] is not None
    else:
        assert create_runtime_app() is not None


@pytest.mark.asyncio
async def test_lifespan_builds_shared_resources_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = StartupResources()
    _track_service_close(monkeypatch, resources.close_order)
    app = resources.create_app()

    assert resources.calls == []
    assert resources.redis.ping_calls == 0
    assert not hasattr(app.state, "runtime_service")
    async with app.router.lifespan_context(app):
        assert resources.calls == [
            ("config", None),
            ("application", resources.config),
            ("redis", resources.config.runtime.redis_url),
        ]
        assert resources.calls[1][1] is resources.config
        assert app.state.redis is resources.redis
        assert app.state.research_app is resources.research_app
        service = app.state.runtime_service
        assert isinstance(service, ResearchRuntimeService)
        assert service.policy is resources.config.runtime.policy
        assert service.registry._redis is resources.redis
        assert service.events._redis is resources.redis
        assert service.events._event_ttl_seconds == service.policy.event_ttl_seconds
        assert service.runner.coordinator is resources.research_app.research_coordinator
        assert resources.redis.ping_calls == 1
        assert resources.close_order == []
        request = Request({"type": "http", "app": app})
        assert _service(request) is service
        assert _service(request) is service
        assert len(resources.calls) == 3

    assert resources.close_order == ["service", "redis", "application"]
    assert resources.redis.close_calls == 1
    assert resources.research_app.close_calls == 1


@pytest.mark.asyncio
async def test_explicit_config_skips_config_factory() -> None:
    resources = StartupResources()
    explicit_config = replace(resources.config, embedding_model="explicit/embedder")
    app = resources.create_app(config=explicit_config)

    assert resources.calls == []
    async with app.router.lifespan_context(app):
        assert resources.calls == [
            ("application", explicit_config),
            ("redis", explicit_config.runtime.redis_url),
        ]
        assert resources.calls[0][1] is explicit_config
        assert app.state.runtime_service.policy is explicit_config.runtime.policy


@pytest.mark.asyncio
async def test_explicit_runtime_config_overrides_url_policy_and_event_ttl() -> None:
    resources = StartupResources()
    runtime = RuntimeConfig(
        redis_url="redis://override.example/8",
        policy=RuntimePolicy(max_concurrency=7, event_ttl_seconds=321),
    )
    app = resources.create_app(runtime_config=runtime)

    assert resources.calls == []
    async with app.router.lifespan_context(app):
        assert resources.calls[1][1] is resources.config
        assert resources.calls[2] == ("redis", runtime.redis_url)
        service = app.state.runtime_service
        assert service.policy is runtime.policy
        assert service.events._event_ttl_seconds == 321


@pytest.mark.asyncio
async def test_config_factory_failure_acquires_no_resources() -> None:
    resources = StartupResources()

    def resolve_config() -> AppConfig:
        resources.calls.append(("config", None))
        raise ValueError("invalid configuration")

    app = create_runtime_app(
        config_factory=resolve_config,
        application_factory=resources.build_application,
        redis_client_factory=resources.create_redis,
    )
    assert resources.calls == []
    with pytest.raises(ValueError, match="invalid configuration"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert resources.calls == [("config", None)]
    assert resources.close_order == []


@pytest.mark.asyncio
async def test_application_factory_failure_leaves_partial_cleanup_to_bootstrap() -> None:
    resources = StartupResources()

    def build_application(config: AppConfig) -> FakeResearchApp:
        resources.calls.append(("application", config))
        raise RuntimeError("application startup failed")

    app = create_runtime_app(
        config_factory=resources.resolve_config,
        application_factory=build_application,
        redis_client_factory=resources.create_redis,
    )
    assert resources.calls == []
    with pytest.raises(RuntimeError, match="application startup failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert resources.calls == [("config", None), ("application", resources.config)]
    assert resources.close_order == []
    assert resources.research_app.close_calls == 0


@pytest.mark.asyncio
async def test_redis_factory_failure_closes_acquired_application() -> None:
    resources = StartupResources()

    def create_redis(url: str) -> FakeRedis:
        resources.calls.append(("redis", url))
        raise RuntimeError("redis construction failed")

    app = create_runtime_app(
        config_factory=resources.resolve_config,
        application_factory=resources.build_application,
        redis_client_factory=create_redis,
    )
    assert resources.calls == []
    with pytest.raises(RuntimeError, match="redis construction failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert len(resources.calls) == 3
    assert resources.close_order == ["application"]
    assert resources.research_app.close_calls == 1
    assert resources.redis.close_calls == 0


@pytest.mark.asyncio
async def test_redis_ping_failure_closes_only_acquired_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = StartupResources()
    resources.redis.fail_ping = True
    _track_service_close(monkeypatch, resources.close_order)
    app = resources.create_app()

    with pytest.raises(RuntimeError, match="redis unavailable"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert resources.redis.ping_calls == 1
    assert resources.close_order == ["redis", "application"]
    assert resources.redis.close_calls == 1
    assert resources.research_app.close_calls == 1
    assert not hasattr(app.state, "runtime_service")


@pytest.mark.asyncio
async def test_reconciliation_failure_closes_service_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = StartupResources()
    resources.redis.fail_scan = True
    _track_service_close(monkeypatch, resources.close_order)
    app = resources.create_app()

    with pytest.raises(RuntimeError, match="reconciliation failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup must fail")

    assert isinstance(app.state.runtime_service, ResearchRuntimeService)
    assert resources.close_order == ["service", "redis", "application"]
    assert resources.redis.close_calls == 1
    assert resources.research_app.close_calls == 1


@pytest.mark.asyncio
async def test_redis_close_failure_still_closes_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = StartupResources()
    resources.redis.fail_close = True
    _track_service_close(monkeypatch, resources.close_order)
    app = resources.create_app()

    with pytest.raises(RuntimeError, match="redis close failed"):
        async with app.router.lifespan_context(app):
            pass

    assert resources.close_order == ["service", "redis", "application"]
    assert resources.redis.close_calls == 1
    assert resources.research_app.close_calls == 1


@pytest.mark.asyncio
async def test_service_close_failure_still_closes_redis_and_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = StartupResources()
    _track_service_close(monkeypatch, resources.close_order, fail=True)
    app = resources.create_app()

    with pytest.raises(RuntimeError, match="service close failed"):
        async with app.router.lifespan_context(app):
            pass

    assert resources.close_order == ["service", "redis", "application"]
    assert resources.redis.close_calls == 1
    assert resources.research_app.close_calls == 1


@pytest.mark.asyncio
async def test_each_lifespan_builds_one_distinct_shared_resource_set() -> None:
    lifetimes: list[StartupResources] = []
    services: list[ResearchRuntimeService] = []

    def resolve_config() -> AppConfig:
        resources = StartupResources()
        lifetimes.append(resources)
        return resources.resolve_config()

    app = create_runtime_app(
        config_factory=resolve_config,
        application_factory=lambda config: lifetimes[-1].build_application(config),
        redis_client_factory=lambda url: lifetimes[-1].create_redis(url),
    )
    assert lifetimes == []
    for _ in range(2):
        async with app.router.lifespan_context(app):
            resources = lifetimes[-1]
            services.append(app.state.runtime_service)
            assert app.state.research_app is resources.research_app
            assert app.state.redis is resources.redis
            request = Request({"type": "http", "app": app})
            assert _service(request) is services[-1]
            assert _service(request) is services[-1]
            assert len(resources.calls) == 3
        assert resources.redis.ping_calls == 1
        assert resources.redis.close_calls == 1
        assert resources.research_app.close_calls == 1

    assert len(lifetimes) == 2
    assert services[0] is not services[1]
    assert lifetimes[0].research_app is not lifetimes[1].research_app
    assert lifetimes[0].redis is not lifetimes[1].redis
