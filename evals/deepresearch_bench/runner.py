"""Thin execution adapter over the production ResearchCoordinator."""

from __future__ import annotations

import math
import os
import signal
import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, Sequence

from openai import APIStatusError, BadRequestError

from evals.deepresearch_bench.artifacts import (
    BENCHMARK_CHECKPOINT_RELATIVE_PATH,
    RunStore,
    utc_now,
)
from evals.deepresearch_bench.dataset import BenchmarkTask
from insight_agent.llm import ContextBudgetExceeded
from insight_agent.planning.coordinator import UnknownResearchThread


class Coordinator(Protocol):
    def start_research(self, query: str, *, thread_id: str) -> Any: ...

    def resume_research(self, thread_id: str) -> Any: ...


class TaskDeadlineExceeded(TimeoutError):
    pass


def _is_retryable_error(error: BaseException) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (BadRequestError, ContextBudgetExceeded)):
            return False
        if isinstance(current, APIStatusError) and current.status_code == 400:
            return False
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return True


@contextmanager
def task_deadline(seconds: float) -> Iterator[None]:
    if seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("per-task timeout requires SIGALRM support")
    previous = signal.getsignal(signal.SIGALRM)

    def expire(signum: int, frame: Any) -> None:
        del signum, frame
        raise TaskDeadlineExceeded(f"task exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class BenchmarkRunner:
    def __init__(
        self,
        coordinator: Coordinator,
        store: RunStore,
        *,
        timeout_seconds: float,
        max_retries: int,
        usage_meter: Any | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.coordinator = coordinator
        self.store = store
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.usage_meter = usage_meter

    def run(
        self,
        tasks: Sequence[BenchmarkTask],
        *,
        retry_failed: bool = False,
    ) -> None:
        for position, task in enumerate(tasks, start=1):
            prior = self.store.task_status(task.id)
            if prior == "succeeded" and self.store.has_complete_report(task.id):
                self.store.log(f"skip task={task.id} prior_status={prior}")
                continue
            if prior in {"failed", "timeout"} and not retry_failed:
                self.store.log(f"skip task={task.id} prior_status={prior}")
                continue
            self.store.log(f"start task={task.id} progress={position}/{len(tasks)}")
            self._run_task(task)
        self.store.finish()

    def _run_task(self, task: BenchmarkTask) -> None:
        started_at = utc_now()
        started = time.monotonic()
        usage_before = self._usage_snapshot()
        last_error: Exception | None = None
        previous_attempts = 0
        previous_elapsed = 0.0
        previous_usage: dict[str, int | None] = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        resume_thread_id: str | None = None
        try:
            prior = self.store.read_execution(task.id)
            previous_attempts = int(prior.get("attempts", 0))
            previous_elapsed = float(prior.get("elapsed_seconds", 0.0))
            if prior.get("status") in {"interrupted", "timeout", "running"}:
                resume_thread_id = str(prior["thread_id"])
            # A process killed while a request is in flight cannot persist a complete
            # usage delta. Preserve that uncertainty instead of under-reporting it.
            if prior.get("status") == "running":
                previous_usage = {name: None for name in previous_usage}
            else:
                previous_usage = {
                    name: prior.get(name)
                    for name in previous_usage
                }
        except FileNotFoundError:
            pass
        for attempt in range(1, self.max_retries + 2):
            absolute_attempt = previous_attempts + attempt
            resume = getattr(self.coordinator, "resume_research", None)
            resuming = resume_thread_id is not None and callable(resume)
            thread_id = resume_thread_id or (
                f"drb-{self.store.directory.name}-task-{task.id}-attempt-{absolute_attempt}"
            )
            if not resuming:
                thread_id = (
                    f"drb-{self.store.directory.name}-task-{task.id}-attempt-{absolute_attempt}"
                )
            self.store.write_case(
                task,
                self._execution(
                    task,
                    thread_id=thread_id,
                    started_at=started_at,
                    started=started,
                    status="running",
                    report_generated=False,
                    attempts=absolute_attempt,
                    elapsed_offset=previous_elapsed,
                    llm_calls=previous_usage["llm_calls"],
                    input_tokens=previous_usage["input_tokens"],
                    output_tokens=previous_usage["output_tokens"],
                ),
            )
            try:
                with task_deadline(self.timeout_seconds):
                    if resuming:
                        try:
                            snapshot = resume(thread_id)
                        except UnknownResearchThread:
                            thread_id = (
                                f"drb-{self.store.directory.name}-task-{task.id}"
                                f"-attempt-{absolute_attempt}"
                            )
                            self.store.write_case(
                                task,
                                self._execution(
                                    task,
                                    thread_id=thread_id,
                                    started_at=started_at,
                                    started=started,
                                    status="running",
                                    report_generated=False,
                                    attempts=absolute_attempt,
                                    elapsed_offset=previous_elapsed,
                                    llm_calls=previous_usage["llm_calls"],
                                    input_tokens=previous_usage["input_tokens"],
                                    output_tokens=previous_usage["output_tokens"],
                                ),
                            )
                            snapshot = self.coordinator.start_research(
                                task.prompt, thread_id=thread_id
                            )
                    else:
                        snapshot = self.coordinator.start_research(
                            task.prompt, thread_id=thread_id
                        )
                if not getattr(snapshot, "completed", False):
                    raise RuntimeError("research workflow returned with pending nodes")
                output = getattr(snapshot.state, "final_output", None)
                if not isinstance(output, str) or not output.strip():
                    raise RuntimeError("research workflow produced an empty report")
                usage = self._combine_usage(
                    previous_usage,
                    self._usage_delta(usage_before, self._usage_snapshot()),
                )
                execution = self._execution(
                    task,
                    thread_id=thread_id,
                    started_at=started_at,
                    started=started,
                    status="succeeded",
                    report_generated=True,
                    attempts=absolute_attempt,
                    elapsed_offset=previous_elapsed,
                    llm_calls=usage["llm_calls"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    tool_calls=len(getattr(snapshot.state, "route_decisions", [])),
                )
                self.store.write_case(task, execution, report=output)
                self.store.log(f"complete task={task.id} status=succeeded")
                return
            except KeyboardInterrupt as exc:
                usage = self._combine_usage(
                    previous_usage,
                    self._usage_delta(usage_before, self._usage_snapshot()),
                )
                execution = self._execution(
                    task,
                    thread_id=thread_id,
                    started_at=started_at,
                    started=started,
                    status="interrupted",
                    report_generated=False,
                    attempts=absolute_attempt,
                    elapsed_offset=previous_elapsed,
                    error=exc,
                    llm_calls=usage["llm_calls"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                )
                self.store.write_case(task, execution)
                self.store.log(f"complete task={task.id} status=interrupted")
                raise
            except Exception as exc:
                last_error = exc
                if attempt <= self.max_retries and _is_retryable_error(exc):
                    self.store.log(
                        f"retry task={task.id} attempt={attempt} error={type(exc).__name__}"
                    )
                    continue
                status = "timeout" if isinstance(exc, TaskDeadlineExceeded) else "failed"
                usage = self._combine_usage(
                    previous_usage,
                    self._usage_delta(usage_before, self._usage_snapshot()),
                )
                execution = self._execution(
                    task,
                    thread_id=thread_id,
                    started_at=started_at,
                    started=started,
                    status=status,
                    report_generated=False,
                    attempts=absolute_attempt,
                    elapsed_offset=previous_elapsed,
                    error=exc,
                    llm_calls=usage["llm_calls"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                )
                self.store.write_case(task, execution)
                self.store.log(
                    f"complete task={task.id} status={status} error={type(exc).__name__}"
                )
                return
        raise AssertionError(f"unreachable after {last_error!r}")

    def _execution(
        self,
        task: BenchmarkTask,
        *,
        thread_id: str,
        started_at: str,
        started: float,
        status: str,
        report_generated: bool,
        attempts: int,
        elapsed_offset: float = 0.0,
        error: BaseException | None = None,
        llm_calls: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tool_calls: int | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": task.id,
            "prompt": task.prompt,
            "language": task.language,
            "topic": task.topic,
            "thread_id": thread_id,
            "start_time": started_at,
            "end_time": utc_now(),
            "status": status,
            "report_generated": report_generated,
            "elapsed_seconds": round(elapsed_offset + time.monotonic() - started, 6),
            "attempts": attempts,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": _safe_error_message(error),
            "checkpoint": {
                "thread_id": thread_id,
                "database": str(BENCHMARK_CHECKPOINT_RELATIVE_PATH),
            },
        }

    def _usage_snapshot(self) -> dict[str, int] | None:
        snapshot = getattr(self.usage_meter, "usage_snapshot", None)
        return snapshot() if callable(snapshot) else None

    @staticmethod
    def _usage_delta(
        before: dict[str, int] | None, after: dict[str, int] | None
    ) -> dict[str, int | None]:
        if before is None or after is None:
            return {"llm_calls": None, "input_tokens": None, "output_tokens": None}
        calls = after["calls"] - before["calls"]
        unmetered = after["unmetered_calls"] - before["unmetered_calls"]
        return {
            "llm_calls": calls,
            "input_tokens": None if unmetered else after["input_tokens"] - before["input_tokens"],
            "output_tokens": None if unmetered else after["output_tokens"] - before["output_tokens"],
        }

    @staticmethod
    def _combine_usage(
        previous: dict[str, int | None], current: dict[str, int | None]
    ) -> dict[str, int | None]:
        return {
            name: None
            if previous.get(name) is None or current.get(name) is None
            else int(previous[name]) + int(current[name])  # type: ignore[arg-type]
            for name in ("llm_calls", "input_tokens", "output_tokens")
        }


def _safe_error_message(error: BaseException | None) -> str | None:
    if error is None:
        return None
    message = str(error)
    for name in (
        "LLM_API_KEY",
        "TAVILY_API_KEY",
        "VISION_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "JINA_API_KEY",
    ):
        secret = os.environ.get(name, "")
        if secret:
            message = message.replace(secret, "<redacted>")
    return message
