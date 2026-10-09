"""FastAPI protocol adapter for the research runtime service."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Header, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from insight_agent.application.query import (
    QueryExecutionError,
    QueryRequest,
    QueryResponse,
    QueryRoutingError,
    QueryRuntimeError,
    QueryService,
)
from insight_agent.router import Intent
from insight_agent.runtime.errors import InvalidRunState, RunNotFound, RuntimeUnavailable
from insight_agent.runtime.models import (
    CreateResearchRunRequest,
    RunRecord,
    TERMINAL_STATUSES,
    RuntimeEvent,
)
from insight_agent.runtime.service import ResearchRuntimeService


TERMINAL_EVENT_TYPES = frozenset(
    {"run.completed", "run.failed", "run.timed_out", "run.interrupted"}
)


def create_api(
    service: ResearchRuntimeService | None = None,
    *,
    query_service: QueryService | None = None,
    lifespan: Any = None,
) -> FastAPI:
    """Build a thin HTTP application around one shared runtime service."""
    app = FastAPI(title="InsightAgent Research Runtime", lifespan=lifespan)
    if service is not None:
        app.state.runtime_service = service
    if query_service is not None:
        app.state.query_service = query_service
    app.include_router(_query_router())
    app.include_router(_router())

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RunNotFound)
    async def run_not_found_handler(
        _request: Request, exc: RunNotFound
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidRunState)
    async def invalid_state_handler(
        _request: Request, exc: InvalidRunState
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(QueryRoutingError)
    async def query_routing_error_handler(
        _request: Request,
        _exc: QueryRoutingError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "unable to route query"},
        )

    @app.exception_handler(QueryExecutionError)
    async def query_execution_error_handler(
        _request: Request,
        _exc: QueryExecutionError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "unable to execute query"},
        )

    @app.exception_handler(QueryRuntimeError)
    @app.exception_handler(RuntimeUnavailable)
    async def query_runtime_error_handler(
        _request: Request,
        _exc: QueryRuntimeError | RuntimeUnavailable,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "research runtime unavailable"},
        )

    return app


def _query_router() -> APIRouter:
    router = APIRouter(tags=["application"])

    @router.post("/v1/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest,
        request: Request,
        response: Response,
    ) -> QueryResponse:
        result = await _query_service(request).query(
            payload.query,
            request_id=request.state.request_id,
        )
        if result.intent is Intent.RESEARCH:
            response.status_code = status.HTTP_202_ACCEPTED
        return result

    return router


def _router() -> APIRouter:
    router = APIRouter(prefix="/v1/research/runs", tags=["research-runtime"])

    @router.post("", response_model=RunRecord, status_code=status.HTTP_202_ACCEPTED)
    async def create_run(
        payload: CreateResearchRunRequest,
        request: Request,
    ) -> RunRecord:
        return await _service(request).create_run(
            payload.query,
            request_id=request.state.request_id,
        )

    @router.get("/{run_id}", response_model=RunRecord)
    async def get_run(run_id: str, request: Request) -> RunRecord:
        return await _service(request).get_run(run_id)

    @router.post(
        "/{run_id}/resume",
        response_model=RunRecord,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def resume_run(run_id: str, request: Request) -> RunRecord:
        return await _service(request).resume_run(
            run_id,
            request_id=request.state.request_id,
        )

    @router.get("/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        service = _service(request)
        await service.get_run(run_id)
        return StreamingResponse(
            _event_stream(service, request, run_id, last_event_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def _service(request: Request) -> ResearchRuntimeService:
    return request.app.state.runtime_service


def _query_service(request: Request) -> QueryService:
    service = getattr(request.app.state, "query_service", None)
    if service is None:
        raise RuntimeUnavailable("query service unavailable")
    return service


async def _event_stream(
    service: ResearchRuntimeService,
    request: Request,
    run_id: str,
    last_event_id: str | None,
) -> AsyncIterator[str]:
    cursor = last_event_id
    while True:
        events = await service.read_events(run_id, cursor, block_ms=1_000)
        for event in events:
            assert event.event_id is not None
            cursor = event.event_id
            yield _encode_sse(event)
            if event.type in TERMINAL_EVENT_TYPES:
                return
        if await request.is_disconnected():
            return
        if not events and (await service.get_run(run_id)).status in TERMINAL_STATUSES:
            late_events = await service.read_events(
                run_id,
                cursor,
                block_ms=1_000,
            )
            for event in late_events:
                assert event.event_id is not None
                cursor = event.event_id
                yield _encode_sse(event)
                if event.type in TERMINAL_EVENT_TYPES:
                    return
            return


def _encode_sse(event: RuntimeEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.type}\ndata: {data}\n\n"
