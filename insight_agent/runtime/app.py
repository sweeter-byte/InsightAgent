"""HTTP lifespan adapter for the shared application composition root."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI
from redis.asyncio import Redis

from insight_agent.application import AppConfig, QueryService, build_application
from insight_agent.runtime.api import create_api
from insight_agent.runtime.events import RedisRuntimeEventStore
from insight_agent.runtime.policies import RuntimeConfig
from insight_agent.runtime.registry import RedisRunRegistry
from insight_agent.runtime.runner import LangGraphResearchRunner
from insight_agent.runtime.service import ResearchRuntimeService


class ComposedResearchApp(Protocol):
    research_coordinator: Any

    def close(self) -> None: ...


def _redis_from_url(url: str) -> Redis:
    return Redis.from_url(url)


def create_runtime_app(
    *,
    config: AppConfig | None = None,
    config_factory: Callable[[], AppConfig] = AppConfig.from_env,
    application_factory: Callable[[AppConfig], ComposedResearchApp] = build_application,
    redis_client_factory: Callable[[str], Any] = _redis_from_url,
    runtime_config: RuntimeConfig | None = None,
) -> FastAPI:
    """Create the HTTP app without acquiring resources until lifespan startup."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        resolved_config = config or config_factory()
        resolved_runtime = runtime_config or resolved_config.runtime
        research_app = application_factory(resolved_config)
        try:
            redis_client = redis_client_factory(resolved_runtime.redis_url)
        except BaseException:
            research_app.close()
            raise
        service: ResearchRuntimeService | None = None
        try:
            await redis_client.ping()
            registry = RedisRunRegistry(redis_client)
            events = RedisRuntimeEventStore(
                redis_client,
                event_ttl_seconds=resolved_runtime.policy.event_ttl_seconds,
            )
            service = ResearchRuntimeService(
                registry=registry,
                events=events,
                runner=LangGraphResearchRunner(
                    research_app.research_coordinator
                ),
                policy=resolved_runtime.policy,
            )
            query_service = QueryService(research_app, service)  # type: ignore[arg-type]
            application.state.redis = redis_client
            application.state.research_app = research_app
            application.state.runtime_service = service
            application.state.query_service = query_service
            await service.reconcile_interrupted_runs()
            yield
        finally:
            try:
                if service is not None:
                    await service.close()
            finally:
                try:
                    await redis_client.aclose()
                finally:
                    research_app.close()

    return create_api(lifespan=lifespan)


app = create_runtime_app()
