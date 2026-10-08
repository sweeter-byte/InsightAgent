from __future__ import annotations

from dataclasses import dataclass

import pytest

from insight_agent.runtime.runner import LangGraphResearchRunner


@dataclass
class _State:
    final_output: str | None


@dataclass
class _Snapshot:
    state: _State
    completed: bool = True


class FakeCoordinator:
    def __init__(self) -> None:
        self.start_calls: list[tuple[str, str]] = []
        self.resume_calls: list[str] = []

    def start_research(self, query: str, *, thread_id: str) -> _Snapshot:
        self.start_calls.append((query, thread_id))
        return _Snapshot(_State("new output"))

    def resume_research(self, thread_id: str) -> _Snapshot:
        self.resume_calls.append(thread_id)
        return _Snapshot(_State("resumed output"))


@pytest.mark.asyncio
async def test_runner_starts_existing_workflow_with_runtime_thread_id() -> None:
    coordinator = FakeCoordinator()
    runner = LangGraphResearchRunner(coordinator)  # type: ignore[arg-type]
    progress: list[tuple[str, dict[str, object]]] = []

    output = await runner.run(
        thread_id="thread-1",
        query="query",
        resume=False,
        on_progress=lambda stage, data: _capture(progress, stage, data),
    )

    assert output == "new output"
    assert coordinator.start_calls == [("query", "thread-1")]
    assert coordinator.resume_calls == []
    assert [stage for stage, _ in progress] == ["planning", "completed"]


@pytest.mark.asyncio
async def test_runner_resumes_checkpoint_without_new_query() -> None:
    coordinator = FakeCoordinator()
    runner = LangGraphResearchRunner(coordinator)  # type: ignore[arg-type]
    progress: list[tuple[str, dict[str, object]]] = []

    output = await runner.run(
        thread_id="thread-1",
        query=None,
        resume=True,
        on_progress=lambda stage, data: _capture(progress, stage, data),
    )

    assert output == "resumed output"
    assert coordinator.start_calls == []
    assert coordinator.resume_calls == ["thread-1"]
    assert [stage for stage, _ in progress] == ["resuming", "completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume", "query"),
    [(False, None), (True, "must-not-be-supplied")],
)
async def test_runner_rejects_inconsistent_execution_mode(
    resume: bool, query: str | None
) -> None:
    runner = LangGraphResearchRunner(FakeCoordinator())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await runner.run(
            thread_id="thread-1",
            query=query,
            resume=resume,
            on_progress=lambda stage, data: _capture([], stage, data),
        )


async def _capture(
    target: list[tuple[str, dict[str, object]]],
    stage: str,
    data: dict[str, object],
) -> None:
    target.append((stage, data))
