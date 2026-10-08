"""Lifecycle orchestration for externally visible research runs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging
from typing import Protocol, TypeVar
from uuid import uuid4

from insight_agent.runtime.errors import (
    InvalidRunState,
    RunNotFound,
    RuntimeUnavailable,
    TransientRuntimeError,
)
from insight_agent.runtime.events import RuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus, RuntimeEvent
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import RunRegistry


ProgressCallback = Callable[[str, dict[str, object]], Awaitable[None]]


class ResearchWorkflowRunner(Protocol):
    async def run(
        self,
        *,
        thread_id: str,
        query: str | None,
        resume: bool,
        on_progress: ProgressCallback,
    ) -> str: ...


T = TypeVar("T")
logger = logging.getLogger(__name__)


class ResearchRuntimeService:
    def __init__(
        self,
        *,
        registry: RunRegistry,
        events: RuntimeEventStore,
        runner: ResearchWorkflowRunner,
        policy: RuntimePolicy,
        run_id_factory: Callable[[], str] | None = None,
        thread_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.registry = registry
        self.events = events
        self.runner = runner
        self.policy = policy
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))
        self._thread_id_factory = thread_id_factory or (lambda: str(uuid4()))
        self._semaphore = asyncio.Semaphore(policy.max_concurrency)
        self._resume_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._execution_owners: dict[str, str] = {}
        self._thread_owners: dict[str, str] = {}
        self._accepting = True

    async def create_run(
        self,
        query: str,
        *,
        request_id: str | None = None,
    ) -> RunRecord:
        if not self._accepting:
            raise RuntimeUnavailable("runtime is shutting down")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        record = RunRecord.new(
            run_id=self._run_id_factory(),
            thread_id=self._thread_id_factory(),
        )
        await self._infra(lambda: self.registry.create(record))
        self._log("run created", record, request_id=request_id)
        await self._append(record, "run.queued", {})
        self._log("run queued", record, request_id=request_id)
        self._spawn(record, query=query, resume=False, request_id=request_id)
        return record

    async def get_run(self, run_id: str) -> RunRecord:
        record = await self._infra(lambda: self.registry.get(run_id))
        if record is None:
            raise RunNotFound(f"unknown run: {run_id}")
        return record

    async def resume_run(
        self,
        run_id: str,
        *,
        request_id: str | None = None,
    ) -> RunRecord:
        if not self._accepting:
            raise RuntimeUnavailable("runtime is shutting down")
        async with self._resume_lock:
            current = await self.get_run(run_id)
            if current.status not in {RunStatus.INTERRUPTED, RunStatus.TIMED_OUT}:
                raise InvalidRunState(
                    f"run {run_id} cannot resume from {current.status.value}"
                )
            if self._has_local_execution(current):
                raise InvalidRunState(
                    f"run {run_id} is still executing locally"
                )
            queued = await self._update(run_id, RunStatus.QUEUED)
            await self._append(queued, "run.resuming", {})
            self._log("run resume requested", queued, request_id=request_id)
            self._spawn(queued, query=None, resume=True, request_id=request_id)
            return queued

    async def read_events(
        self,
        run_id: str,
        event_id: str | None,
        *,
        block_ms: int = 0,
    ) -> list[RuntimeEvent]:
        await self.get_run(run_id)
        return await self._infra(
            lambda: self.events.read_after(run_id, event_id, block_ms=block_ms)
        )

    async def reconcile_interrupted_runs(self) -> int:
        records = await self._infra(
            lambda: self.registry.list_by_status({RunStatus.RUNNING})
        )
        for record in records:
            interrupted = await self._update(record.run_id, RunStatus.INTERRUPTED)
            await self._append(interrupted, "run.interrupted", {})
            self._log("run interrupted", interrupted)
        return len(records)

    async def close(self) -> None:
        self._accepting = False
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn(
        self,
        record: RunRecord,
        *,
        query: str | None,
        resume: bool,
        request_id: str | None,
    ) -> None:
        task = asyncio.create_task(
            self._execute(
                record,
                query=query,
                resume=resume,
                request_id=request_id,
            ),
            name=f"research-run:{record.run_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(
        self,
        record: RunRecord,
        *,
        query: str | None,
        resume: bool,
        request_id: str | None,
    ) -> None:
        execution_task: asyncio.Task[str] | None = None
        interrupted_recorded = False
        try:
            async with self._semaphore:
                running = await self._update(record.run_id, RunStatus.RUNNING)
                await self._append(running, "run.started", {"resume": resume})
                self._log("run started", running, request_id=request_id)
                if resume:
                    self._log("run resumed", running, request_id=request_id)

                progress_enabled = True

                async def on_progress(
                    stage: str, data: dict[str, object]
                ) -> None:
                    if not progress_enabled:
                        return
                    safe_data = {"stage": stage, **data}
                    await self._append(running, "workflow.progress", safe_data)

                self._claim_execution(record)
                try:
                    execution_task = asyncio.create_task(
                        self.runner.run(
                            thread_id=record.thread_id,
                            query=query,
                            resume=resume,
                            on_progress=on_progress,
                        ),
                        name=f"research-workflow:{record.run_id}",
                    )
                    try:
                        async with asyncio.timeout(
                            self.policy.run_timeout_seconds
                        ):
                            final_output = await asyncio.shield(execution_task)
                    except TimeoutError:
                        progress_enabled = False
                        try:
                            timed_out = await self._update(
                                record.run_id, RunStatus.TIMED_OUT
                            )
                            await self._append(timed_out, "run.timed_out", {})
                            self._log(
                                "run timed out",
                                timed_out,
                                request_id=request_id,
                            )
                        finally:
                            await self._drain_execution(execution_task, record)
                        return
                    except asyncio.CancelledError:
                        progress_enabled = False
                        interrupted_recorded = True
                        try:
                            interrupted = await self._update(
                                record.run_id, RunStatus.INTERRUPTED
                            )
                            await self._append(
                                interrupted, "run.interrupted", {}
                            )
                            self._log(
                                "run interrupted",
                                interrupted,
                                request_id=request_id,
                            )
                        finally:
                            await self._drain_execution(execution_task, record)
                        raise
                finally:
                    self._release_execution(record)

                completed = await self._update(
                    record.run_id,
                    RunStatus.COMPLETED,
                    final_output=final_output,
                )
                await self._append(
                    completed,
                    "run.completed",
                    {"final_output": final_output},
                )
                self._log("run completed", completed, request_id=request_id)
        except asyncio.CancelledError:
            if not interrupted_recorded:
                interrupted = await self._update(
                    record.run_id, RunStatus.INTERRUPTED
                )
                await self._append(interrupted, "run.interrupted", {})
                self._log(
                    "run interrupted", interrupted, request_id=request_id
                )
            raise
        except Exception as exc:
            logger.exception(
                "run failed",
                extra={
                    "request_id": request_id,
                    "run_id": record.run_id,
                    "thread_id": record.thread_id,
                    "status": RunStatus.FAILED.value,
                },
            )
            failed = await self._update(
                record.run_id,
                RunStatus.FAILED,
                error=type(exc).__name__,
            )
            await self._append(
                failed,
                "run.failed",
                {"error": type(exc).__name__},
            )

    async def _drain_execution(
        self,
        task: asyncio.Task[str],
        record: RunRecord,
    ) -> None:
        """Keep local ownership until the underlying runner actually exits."""
        while True:
            try:
                await asyncio.shield(task)
                return
            except asyncio.CancelledError:
                if task.cancelled():
                    return
                continue
            except Exception:
                logger.exception(
                    "workflow exited after runtime execution boundary",
                    extra={
                        "run_id": record.run_id,
                        "thread_id": record.thread_id,
                    },
                )
                return

    def _claim_execution(self, record: RunRecord) -> None:
        if self._has_local_execution(record):
            raise InvalidRunState(
                f"run {record.run_id} is still executing locally"
            )
        self._execution_owners[record.run_id] = record.thread_id
        self._thread_owners[record.thread_id] = record.run_id

    def _release_execution(self, record: RunRecord) -> None:
        if self._execution_owners.get(record.run_id) == record.thread_id:
            self._execution_owners.pop(record.run_id, None)
        if self._thread_owners.get(record.thread_id) == record.run_id:
            self._thread_owners.pop(record.thread_id, None)

    def _has_local_execution(self, record: RunRecord) -> bool:
        return (
            record.run_id in self._execution_owners
            or record.thread_id in self._thread_owners
        )

    async def _update(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
        final_output: str | None = None,
    ) -> RunRecord:
        return await self._infra(
            lambda: self.registry.update_status(
                run_id,
                status,
                error=error,
                final_output=final_output,
            )
        )

    async def _append(
        self,
        record: RunRecord,
        event_type: str,
        data: dict[str, object],
    ) -> str:
        event = RuntimeEvent(run_id=record.run_id, type=event_type, data=data)
        return await self._infra(lambda: self.events.append(event))

    async def _infra(self, operation: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(1, self.policy.infra_retry_attempts + 1):
            try:
                return await operation()
            except TransientRuntimeError:
                if attempt == self.policy.infra_retry_attempts:
                    raise
                await asyncio.sleep(
                    self.policy.infra_retry_backoff_seconds * attempt
                )
        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _log(
        message: str,
        record: RunRecord,
        *,
        request_id: str | None = None,
    ) -> None:
        logger.info(
            message,
            extra={
                "request_id": request_id,
                "run_id": record.run_id,
                "thread_id": record.thread_id,
                "status": record.status.value,
            },
        )
