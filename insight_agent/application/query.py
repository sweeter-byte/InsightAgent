"""Asynchronous application facade for unified query dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import Enum
import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from insight_agent.router import Intent
from insight_agent.runtime.errors import RuntimeUnavailable
from insight_agent.runtime.models import RunRecord


logger = logging.getLogger(__name__)


class QueryStatus(str, Enum):
    COMPLETED = "completed"
    QUEUED = "queued"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    intent: Intent
    status: QueryStatus
    answer: str | None = None
    run_id: str | None = None


class QueryRoutingError(RuntimeError):
    """The request could not be classified safely."""


class QueryExecutionError(RuntimeError):
    """A synchronous direct or analyze branch failed."""


class QueryRuntimeError(RuntimeError):
    """The request could not be submitted to the research runtime."""


class QueryApplication(Protocol):
    def route(self, query: str) -> Intent: ...

    def answer_direct(self, query: str) -> str: ...

    def analyze(self, query: str) -> str: ...


class QueryRuntime(Protocol):
    async def create_run(
        self,
        query: str,
        *,
        request_id: str | None = None,
    ) -> RunRecord: ...


class QueryService:
    """Route a query without blocking the HTTP event loop."""

    def __init__(
        self,
        application: QueryApplication,
        runtime: QueryRuntime,
    ) -> None:
        self.application = application
        self.runtime = runtime

    async def query(self, query: str, *, request_id: str) -> QueryResponse:
        try:
            intent = await asyncio.to_thread(self.application.route, query)
        except Exception as exc:
            logger.exception(
                "query routing failed",
                extra={"request_id": request_id, "stage": "routing"},
            )
            raise QueryRoutingError("unable to route query") from exc

        if intent is Intent.DIRECT:
            answer = await self._execute_sync(
                self.application.answer_direct,
                query,
                request_id=request_id,
                intent=intent,
            )
            return QueryResponse(
                request_id=request_id,
                intent=intent,
                status=QueryStatus.COMPLETED,
                answer=answer,
            )

        if intent is Intent.ANALYZE:
            answer = await self._execute_sync(
                self.application.analyze,
                query,
                request_id=request_id,
                intent=intent,
            )
            return QueryResponse(
                request_id=request_id,
                intent=intent,
                status=QueryStatus.COMPLETED,
                answer=answer,
            )

        try:
            run = await self.runtime.create_run(query, request_id=request_id)
        except RuntimeUnavailable:
            logger.exception(
                "research runtime unavailable",
                extra={
                    "request_id": request_id,
                    "stage": "runtime_submission",
                    "intent": intent.value,
                },
            )
            raise
        except Exception as exc:
            logger.exception(
                "research runtime submission failed",
                extra={
                    "request_id": request_id,
                    "stage": "runtime_submission",
                    "intent": intent.value,
                },
            )
            raise QueryRuntimeError("research runtime unavailable") from exc
        return QueryResponse(
            request_id=request_id,
            intent=intent,
            status=QueryStatus(run.status.value),
            run_id=run.run_id,
        )

    async def _execute_sync(
        self,
        operation: Callable[[str], str],
        query: str,
        *,
        request_id: str,
        intent: Intent,
    ) -> str:
        try:
            return await asyncio.to_thread(operation, query)
        except Exception as exc:
            logger.exception(
                "synchronous query execution failed",
                extra={
                    "request_id": request_id,
                    "stage": "execution",
                    "intent": intent.value,
                },
            )
            raise QueryExecutionError("unable to execute query") from exc
