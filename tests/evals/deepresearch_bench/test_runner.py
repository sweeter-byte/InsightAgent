from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, BadRequestError

from evals.deepresearch_bench.artifacts import RunStore
from evals.deepresearch_bench.dataset import BenchmarkTask
from evals.deepresearch_bench.runner import BenchmarkRunner, TaskDeadlineExceeded, task_deadline
from insight_agent.planning.coordinator import UnknownResearchThread


def _task(task_id: int) -> BenchmarkTask:
    return BenchmarkTask(task_id, "topic", "zh" if task_id < 50 else "en", f"prompt {task_id}")


class FakeCoordinator:
    def __init__(self, *, fail_ids: set[int] | None = None, empty_ids: set[int] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.empty_ids = empty_ids or set()
        self.calls: list[tuple[str, str]] = []

    def start_research(self, query: str, *, thread_id: str):
        task_id = int(query.rsplit(" ", 1)[-1])
        self.calls.append((query, thread_id))
        if task_id in self.fail_ids:
            raise RuntimeError("provider failed")
        output = "" if task_id in self.empty_ids else f"# Report {task_id}\n\nFact [{task_id}](https://example.com/{task_id})."
        state = SimpleNamespace(final_output=output, route_decisions=[object(), object()])
        return SimpleNamespace(completed=True, state=state)


def _status_error(
    status_code: int, error_type: type[APIStatusError] = APIStatusError
) -> APIStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return error_type(
        "maximum context length exceeded",
        response=response,
        body={"error": {"message": "maximum context length exceeded"}},
    )


def _store(tmp_path: Path, *, fingerprint: str = "fp") -> RunStore:
    return RunStore.create_or_resume(
        tmp_path,
        "run-1",
        manifest={"run_id": "run-1", "task_count": 2},
        config_fingerprint=fingerprint,
    )


def test_runner_uses_independent_threads_and_persists_reports(tmp_path: Path) -> None:
    coordinator = FakeCoordinator()
    store = _store(tmp_path)

    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=0).run(
        [_task(1), _task(51)]
    )

    assert len({thread for _, thread in coordinator.calls}) == 2
    assert all(thread.startswith("drb-run-1-task-") for _, thread in coordinator.calls)
    case = tmp_path / "run-1" / "cases" / "1"
    assert "https://example.com/1" in (case / "report.md").read_text(encoding="utf-8")
    assert (case / "report.raw.md").read_text(encoding="utf-8") == (
        case / "report.md"
    ).read_text(encoding="utf-8")
    execution = json.loads((case / "execution.json").read_text(encoding="utf-8"))
    assert execution["status"] == "succeeded"
    assert execution["report_generated"] is True
    assert execution["tool_calls"] == 2
    assert execution["input_tokens"] is None


def test_runner_marks_failures_and_empty_reports_without_fake_success(tmp_path: Path) -> None:
    coordinator = FakeCoordinator(fail_ids={1}, empty_ids={51})
    store = _store(tmp_path)

    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=0).run(
        [_task(1), _task(51)]
    )

    one = store.read_execution(1)
    fifty_one = store.read_execution(51)
    assert one["status"] == "failed"
    assert one["error_type"] == "RuntimeError"
    assert fifty_one["status"] == "failed"
    assert fifty_one["report_generated"] is False
    assert "empty" in fifty_one["error_message"].lower()


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (_status_error(400, BadRequestError), "BadRequestError"),
        (_status_error(400), "APIStatusError"),
    ],
)
def test_runner_does_not_retry_http_400_errors(
    tmp_path: Path, error: APIStatusError, expected_type: str
) -> None:
    class BadRequestCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            self.calls.append((query, thread_id))
            raise error

    coordinator = BadRequestCoordinator()
    store = _store(tmp_path)

    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=2).run(
        [_task(1)]
    )

    execution = store.read_execution(1)
    assert len(coordinator.calls) == 1
    assert execution["status"] == "failed"
    assert execution["attempts"] == 1
    assert execution["error_type"] == expected_type
    assert execution["error_message"] == "maximum context length exceeded"


@pytest.mark.parametrize("chain_attribute", ["__cause__", "__context__"])
def test_runner_does_not_retry_wrapped_http_400_errors(
    tmp_path: Path, chain_attribute: str
) -> None:
    wrapped = RuntimeError("research failed")
    setattr(wrapped, chain_attribute, _status_error(400, BadRequestError))

    class WrappedBadRequestCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            self.calls.append((query, thread_id))
            raise wrapped

    coordinator = WrappedBadRequestCoordinator()
    store = _store(tmp_path)

    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=2).run(
        [_task(1)]
    )

    execution = store.read_execution(1)
    assert len(coordinator.calls) == 1
    assert execution["status"] == "failed"
    assert execution["attempts"] == 1
    assert execution["error_type"] == "RuntimeError"
    assert execution["error_message"] == "research failed"


def test_runner_retries_other_errors_up_to_max_retries(tmp_path: Path) -> None:
    coordinator = FakeCoordinator(fail_ids={1})
    store = _store(tmp_path)

    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=2).run(
        [_task(1)]
    )

    execution = store.read_execution(1)
    assert len(coordinator.calls) == 3
    assert execution["status"] == "failed"
    assert execution["attempts"] == 3
    assert execution["error_type"] == "RuntimeError"


def test_runner_redacts_api_keys_from_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-secret-value"
    monkeypatch.setenv("LLM_API_KEY", secret)

    class LeakyCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            raise RuntimeError(f"provider rejected {secret}")

    store = _store(tmp_path)
    BenchmarkRunner(
        LeakyCoordinator(), store, timeout_seconds=10, max_retries=0
    ).run([_task(1)])

    execution_text = (
        store.directory / "cases" / "1" / "execution.json"
    ).read_text(encoding="utf-8")
    assert secret not in execution_text
    assert "<redacted>" in execution_text


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan])
def test_runner_rejects_invalid_timeout(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        BenchmarkRunner(FakeCoordinator(), _store(tmp_path), timeout_seconds=timeout, max_retries=0)


def test_resume_skips_success_and_retries_failure_only_when_requested(tmp_path: Path) -> None:
    first = FakeCoordinator(fail_ids={51})
    store = _store(tmp_path)
    tasks = [_task(1), _task(51)]
    BenchmarkRunner(first, store, timeout_seconds=10, max_retries=0).run(tasks)

    second = FakeCoordinator()
    BenchmarkRunner(second, store, timeout_seconds=10, max_retries=0).run(tasks)
    assert second.calls == []

    BenchmarkRunner(second, store, timeout_seconds=10, max_retries=0).run(
        tasks, retry_failed=True
    )
    assert [query for query, _ in second.calls] == ["prompt 51"]


def test_resume_reruns_success_record_with_missing_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    task = _task(1)
    BenchmarkRunner(
        FakeCoordinator(), store, timeout_seconds=10, max_retries=0
    ).run([task])
    (store.directory / "cases" / "1" / "report.md").unlink()

    resumed = FakeCoordinator()
    BenchmarkRunner(
        resumed, store, timeout_seconds=10, max_retries=0
    ).run([task])

    assert [query for query, _ in resumed.calls] == ["prompt 1"]
    assert (store.directory / "cases" / "1" / "report.md").is_file()


def test_run_refuses_silent_resume_with_different_config(tmp_path: Path) -> None:
    _store(tmp_path, fingerprint="first")

    with pytest.raises(ValueError, match="configuration"):
        _store(tmp_path, fingerprint="second")


def test_keyboard_interrupt_is_persisted_then_reraised(tmp_path: Path) -> None:
    class InterruptedCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            raise KeyboardInterrupt

    store = _store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        BenchmarkRunner(
            InterruptedCoordinator(), store, timeout_seconds=10, max_retries=0
        ).run([_task(1)])

    assert store.read_execution(1)["status"] == "interrupted"


def test_resume_continues_interrupted_langgraph_thread(tmp_path: Path) -> None:
    class InterruptedCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            raise KeyboardInterrupt

    store = _store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        BenchmarkRunner(
            InterruptedCoordinator(), store, timeout_seconds=10, max_retries=0
        ).run([_task(1)])
    interrupted_thread = store.read_execution(1)["thread_id"]

    class ResumingCoordinator(FakeCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.resume_calls: list[str] = []

        def resume_research(self, thread_id: str):
            self.resume_calls.append(thread_id)
            return SimpleNamespace(
                completed=True,
                state=SimpleNamespace(
                    final_output="# Resumed\n\nFact [1](https://example.com/1).",
                    route_decisions=[object()],
                ),
            )

    coordinator = ResumingCoordinator()
    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=0).run(
        [_task(1)]
    )

    assert coordinator.resume_calls == [interrupted_thread]
    assert coordinator.calls == []
    assert store.read_execution(1)["status"] == "succeeded"


def test_resume_without_checkpoint_starts_a_fresh_thread(tmp_path: Path) -> None:
    class InterruptedBeforeCheckpoint(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            raise KeyboardInterrupt

    store = _store(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        BenchmarkRunner(
            InterruptedBeforeCheckpoint(), store, timeout_seconds=10, max_retries=0
        ).run([_task(1)])
    old_thread = store.read_execution(1)["thread_id"]

    class MissingCheckpointCoordinator(FakeCoordinator):
        def __init__(self) -> None:
            super().__init__()
            self.resume_calls: list[str] = []

        def resume_research(self, thread_id: str):
            self.resume_calls.append(thread_id)
            raise UnknownResearchThread(thread_id)

    coordinator = MissingCheckpointCoordinator()
    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=0).run(
        [_task(1)]
    )

    assert coordinator.resume_calls == [old_thread]
    assert len(coordinator.calls) == 1
    assert coordinator.calls[0][1] != old_thread
    assert store.read_execution(1)["status"] == "succeeded"


def test_task_deadline_raises_timeout() -> None:
    with pytest.raises(TaskDeadlineExceeded):
        with task_deadline(0.01):
            time.sleep(0.1)


def test_runner_persists_timeout_status(tmp_path: Path) -> None:
    class SlowCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            time.sleep(0.1)
            return super().start_research(query, thread_id=thread_id)

    store = _store(tmp_path)
    BenchmarkRunner(
        SlowCoordinator(), store, timeout_seconds=0.01, max_retries=0
    ).run([_task(1)])

    execution = store.read_execution(1)
    assert execution["status"] == "timeout"
    assert execution["report_generated"] is False


def test_export_contains_only_successful_nonempty_reports(tmp_path: Path) -> None:
    coordinator = FakeCoordinator(fail_ids={51})
    store = _store(tmp_path)
    BenchmarkRunner(coordinator, store, timeout_seconds=10, max_retries=0).run(
        [_task(1), _task(51)]
    )

    output = store.export_jsonl()
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert rows == [
        {
            "id": 1,
            "prompt": "prompt 1",
            "article": "# Report 1\n\nFact [1](https://example.com/1).",
        }
    ]
    assert "https://example.com/1" in rows[0]["article"]


def test_export_does_not_invent_citations(tmp_path: Path) -> None:
    class NoCitationCoordinator(FakeCoordinator):
        def start_research(self, query: str, *, thread_id: str):
            return SimpleNamespace(
                completed=True,
                state=SimpleNamespace(final_output="# Report\n\nNo source.", route_decisions=[]),
            )

    store = _store(tmp_path)
    BenchmarkRunner(
        NoCitationCoordinator(), store, timeout_seconds=10, max_retries=0
    ).run([_task(1)])

    article = json.loads(store.export_jsonl().read_text(encoding="utf-8"))["article"]
    assert article == "# Report\n\nNo source."
    assert "http" not in article
