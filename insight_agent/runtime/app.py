"""Production composition root for ``uvicorn insight_agent.runtime.app:app``."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import os
from typing import Any, Protocol

from fastapi import FastAPI
from redis.asyncio import Redis

from insight_agent.runtime.api import create_api
from insight_agent.runtime.events import RedisRuntimeEventStore
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import RedisRunRegistry
from insight_agent.runtime.runner import LangGraphResearchRunner
from insight_agent.runtime.service import ResearchRuntimeService


DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class ComposedResearchApp(Protocol):
    research_coordinator: Any

    def close(self) -> None: ...


def _build_research_app() -> ComposedResearchApp:
    from insight_agent.__main__ import _build_app

    return _build_app()


def _redis_from_url(url: str) -> Redis:
    return Redis.from_url(url)


def create_runtime_app(
    *,
    research_app_factory: Callable[[], ComposedResearchApp] = _build_research_app,
    redis_client_factory: Callable[[str], Any] = _redis_from_url,
    redis_url: str | None = None,
    policy: RuntimePolicy | None = None,
) -> FastAPI:
    """Create the HTTP app without acquiring resources until lifespan startup."""
    runtime_policy = policy or RuntimePolicy.from_env()
    configured_redis_url = redis_url or os.getenv(
        "RUNTIME_REDIS_URL", DEFAULT_REDIS_URL
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        research_app = research_app_factory()
        try:
            redis_client = redis_client_factory(configured_redis_url)
        except BaseException:
            research_app.close()
            raise
        service: ResearchRuntimeService | None = None
        try:
            await redis_client.ping()
            registry = RedisRunRegistry(redis_client)
            events = RedisRuntimeEventStore(
                redis_client,
                event_ttl_seconds=runtime_policy.event_ttl_seconds,
            )
            service = ResearchRuntimeService(
                registry=registry,
                events=events,
                runner=LangGraphResearchRunner(
                    research_app.research_coordinator
                ),
                policy=runtime_policy,
            )
            application.state.redis = redis_client
            application.state.research_app = research_app
            application.state.runtime_service = service
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
