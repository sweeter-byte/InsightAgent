from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from insight_agent.planning import ResearchCoordinator
from insight_agent.research import SQLiteCheckpointStore
from insight_agent.routing import RetrievalSource
from insight_agent.runtime.events import InMemoryRuntimeEventStore
from insight_agent.runtime.models import RunRecord, RunStatus
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import InMemoryRunRegistry
from insight_agent.runtime.runner import LangGraphResearchRunner
from insight_agent.runtime.service import ResearchRuntimeService
from tests.test_research_persistence import (
    _CountingPlanner,
    _FailOnceRetriever,
    _persistent_workflow,
)


@pytest.mark.asyncio
async def test_runtime_resumes_original_sqlite_checkpoint_without_replanning(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "runtime-checkpoints.sqlite"
    thread_id = "persisted-thread"
    run_id = "public-run"
    planner = _CountingPlanner()
    retriever = _FailOnceRetriever()

    with SQLiteCheckpointStore(checkpoint_path) as first_store:
        first_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=_persistent_workflow(
                first_store.checkpointer,
                retriever=retriever,
            ),
            available_sources={RetrievalSource.LOCAL},
        )
        with pytest.raises(RuntimeError, match="simulated crash"):
            first_coordinator.start_research("query", thread_id=thread_id)

    registry = InMemoryRunRegistry()
    events = InMemoryRuntimeEventStore()
    await registry.create(
        RunRecord.new(
            run_id=run_id,
            thread_id=thread_id,
            status=RunStatus.RUNNING,
        )
    )

    with SQLiteCheckpointStore(checkpoint_path) as second_store:
        rebuilt_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=_persistent_workflow(
                second_store.checkpointer,
                retriever=retriever,
            ),
            available_sources={RetrievalSource.LOCAL},
        )
        service = ResearchRuntimeService(
            registry=registry,
            events=events,
            runner=LangGraphResearchRunner(rebuilt_coordinator),
            policy=RuntimePolicy(infra_retry_backoff_seconds=0),
        )

        assert await service.reconcile_interrupted_runs() == 1
        interrupted = await service.get_run(run_id)
        assert interrupted.status is RunStatus.INTERRUPTED

        resumed = await service.resume_run(run_id)
        assert resumed.thread_id == thread_id
        async with asyncio.timeout(5):
            while (await service.get_run(run_id)).status is not RunStatus.COMPLETED:
                await asyncio.sleep(0.01)

        completed = await service.get_run(run_id)
        assert completed.thread_id == thread_id
        assert completed.final_output is not None
        assert planner.calls == ["query"]
        assert retriever.calls == 2
        await service.close()
