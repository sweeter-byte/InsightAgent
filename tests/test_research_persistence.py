"""Persistence tests for Chapter 15 research threads."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from insight_agent.evidence import Evidence
from insight_agent.ingestion import SourceType
from insight_agent.planning import ResearchCoordinator, ResearchPlan
from insight_agent.research import (
    DEFAULT_CHECKPOINT_PATH,
    ResearchRoutingWorkflow,
    SQLiteCheckpointStore,
    thread_config,
)
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource
from insight_agent.self_check import SelfCheckResult
from tests.test_self_check_workflow import (
    AlwaysPassChecker,
    FakeReportGenerator,
    FakeRetriever,
    FakeRouter,
    InsufficientGrader,
    NeverRepairer,
    _state,
)


def _persistent_workflow(
    checkpointer: object,
    *,
    retriever: object | None = None,
    report_generator: object | None = None,
    self_checker: object | None = None,
) -> ResearchRoutingWorkflow:
    return ResearchRoutingWorkflow(
        router=FakeRouter(),  # type: ignore[arg-type]
        local_retriever=retriever or FakeRetriever(),  # type: ignore[arg-type]
        grader=InsufficientGrader(),  # type: ignore[arg-type]
        report_generator=(
            report_generator or FakeReportGenerator()
        ),  # type: ignore[arg-type]
        self_checker=self_checker or AlwaysPassChecker(),  # type: ignore[arg-type]
        report_repairer=NeverRepairer(),  # type: ignore[arg-type]
        max_retrieval_rounds=1,
        checkpointer=checkpointer,  # type: ignore[arg-type]
    )


def test_thread_config_requires_and_preserves_non_empty_id() -> None:
    assert DEFAULT_CHECKPOINT_PATH == Path(".insight_agent/checkpoints.sqlite")
    assert thread_config("research-1") == {
        "configurable": {"thread_id": "research-1"}
    }

    with pytest.raises(ValueError, match="thread_id"):
        thread_config("  ")


def test_sqlite_store_creates_parent_and_closes_connection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "checkpoints.sqlite"

    store = SQLiteCheckpointStore(path)

    assert path.parent.is_dir()
    assert store.checkpointer is not None
    store.close()
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.connection.execute("SELECT 1")


def test_sqlite_store_context_manager_closes_connection(tmp_path: Path) -> None:
    with SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite") as store:
        connection = store.connection
        assert connection.execute("SELECT 1").fetchone() == (1,)

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_research_graph_persists_and_reads_completed_state(tmp_path: Path) -> None:
    config = thread_config("thread-a")
    with SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite") as store:
        workflow = _persistent_workflow(store.checkpointer)

        final_state = workflow.run(_state(), config=config)
        snapshot = workflow.get_state(config)

        assert snapshot.metadata is not None
        assert snapshot.next == ()
        assert snapshot.values["query"] == "Assess method A"
        restored = workflow.state_from_snapshot(snapshot)
        assert restored == final_state
        assert restored.final_output is not None


def test_research_graph_returns_empty_snapshot_for_unknown_thread(
    tmp_path: Path,
) -> None:
    with SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite") as store:
        workflow = _persistent_workflow(store.checkpointer)

        snapshot = workflow.get_state(thread_config("does-not-exist"))

        assert snapshot.metadata is None
        assert snapshot.created_at is None
        assert snapshot.values == {}
        assert snapshot.next == ()


class _TaggedRetriever:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"chunk-{self.tag}",
                score=0.9,
                content=f"Evidence unique to {self.tag}.",
                document_id=f"document-{self.tag}",
                source=f"{self.tag}.md",
                source_type=SourceType.MARKDOWN,
                chunk_index=0,
                start_char=0,
                end_char=20,
                metadata={"filename": f"{self.tag}.md"},
            )
        ]


def test_threads_survive_store_and_graph_rebuild_without_state_mixing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints.sqlite"
    config_a = thread_config("thread-a")
    config_b = thread_config("thread-b")

    with SQLiteCheckpointStore(path) as first_store:
        workflow_a = _persistent_workflow(
            first_store.checkpointer,
            retriever=_TaggedRetriever("A"),
        )
        workflow_b = _persistent_workflow(
            first_store.checkpointer,
            retriever=_TaggedRetriever("B"),
        )
        state_a = _state()
        state_b = _state()
        state_a.query = state_b.query = "same query"
        workflow_a.run(state_a, config=config_a)
        workflow_b.run(state_b, config=config_b)

    with SQLiteCheckpointStore(path) as second_store:
        rebuilt_workflow = _persistent_workflow(second_store.checkpointer)
        restored_a = rebuilt_workflow.state_from_snapshot(
            rebuilt_workflow.get_state(config_a)
        )
        restored_b = rebuilt_workflow.state_from_snapshot(
            rebuilt_workflow.get_state(config_b)
        )

        assert restored_a.query == restored_b.query == "same query"
        assert restored_a.evidence_pool != restored_b.evidence_pool
        assert restored_a.evidence_assessments != restored_b.evidence_assessments
        assert restored_a.report != restored_b.report
        assert restored_a.final_output != restored_b.final_output
        assert isinstance(restored_a.plan, ResearchPlan)
        assert isinstance(restored_a.evidence_pool["T1"][0], Evidence)
        assert (
            restored_a.evidence_pool["T1"][0].retrieval_source
            is RetrievalSource.LOCAL
        )
        assert isinstance(restored_a.self_check_result, SelfCheckResult)


class _CountingPlanner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def plan(self, query: str) -> ResearchPlan:
        self.calls.append(query)
        plan = _state().plan
        assert plan is not None
        return plan


class _FailOnceRetriever(_TaggedRetriever):
    def __init__(self) -> None:
        super().__init__("replayed")
        self.calls = 0

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated crash after external effect")
        return super().retrieve(query, top_k)


def test_failed_node_replays_from_latest_checkpoint_without_duplicate_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints.sqlite"
    thread_id = "replay-thread"
    config = thread_config(thread_id)
    planner = _CountingPlanner()
    retriever = _FailOnceRetriever()

    with SQLiteCheckpointStore(path) as first_store:
        first_workflow = _persistent_workflow(
            first_store.checkpointer,
            retriever=retriever,
        )
        first_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=first_workflow,
            available_sources={RetrievalSource.LOCAL},
            thread_id_factory=lambda: thread_id,
        )

        with pytest.raises(RuntimeError, match="simulated crash"):
            first_coordinator.start_research("query")

        failed_snapshot = first_workflow.get_state(config)
        assert failed_snapshot.next == ("local_entry",)
        assert failed_snapshot.values["task_index"] == 0
        assert failed_snapshot.values["route_decisions"] == []
        pending = first_coordinator.get_research_state(thread_id)
        assert pending.completed is False
        assert pending.next_nodes == ("local_entry",)

    with SQLiteCheckpointStore(path) as second_store:
        rebuilt_workflow = _persistent_workflow(
            second_store.checkpointer,
            retriever=retriever,
        )
        rebuilt_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=rebuilt_workflow,
            available_sources={RetrievalSource.LOCAL},
        )

        resumed = rebuilt_coordinator.resume_research(thread_id)

        assert resumed.completed is True
        assert planner.calls == ["query"]
        assert retriever.calls == 2
        assert resumed.state.plan is not None
        assert resumed.state.task_index == len(resumed.state.plan.tasks)
        assert len(resumed.state.route_decisions) == 1
        assert len(resumed.state.evidence_pool["T1"]) == 1


class _NeverReportGenerator:
    def generate_section(self, **kwargs: object) -> object:
        raise AssertionError("completed resume must not generate a report")


class _NeverSelfChecker:
    def check(self, **kwargs: object) -> object:
        raise AssertionError("completed resume must not run self-check")


def test_completed_thread_resume_is_stable_read_after_rebuild(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoints.sqlite"
    thread_id = "complete-thread"
    planner = _CountingPlanner()

    with SQLiteCheckpointStore(path) as first_store:
        first_workflow = _persistent_workflow(first_store.checkpointer)
        first_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=first_workflow,
            available_sources={RetrievalSource.LOCAL},
            thread_id_factory=lambda: thread_id,
        )
        completed = first_coordinator.start_research("query")

    with SQLiteCheckpointStore(path) as second_store:
        rebuilt_workflow = _persistent_workflow(
            second_store.checkpointer,
            report_generator=_NeverReportGenerator(),
            self_checker=_NeverSelfChecker(),
        )
        rebuilt_coordinator = ResearchCoordinator(
            planner=planner,
            workflow=rebuilt_workflow,
            available_sources={RetrievalSource.LOCAL},
        )

        resumed = rebuilt_coordinator.resume_research(thread_id)

        assert resumed.completed is True
        assert resumed.state == completed.state
        assert planner.calls == ["query"]


def test_checkpoint_can_resume_in_a_new_python_process(tmp_path: Path) -> None:
    path = tmp_path / "subprocess-checkpoints.sqlite"
    thread_id = "cross-process-thread"
    repository_root = Path(__file__).resolve().parents[1]
    process_a = textwrap.dedent(
        """
        import sys
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from insight_agent.research import SQLiteCheckpointStore, thread_config


        class State(TypedDict):
            value: int


        def prepare(state: State) -> dict[str, int]:
            return {"value": state["value"] + 1}


        def finish(state: State) -> dict[str, int]:
            return {"value": state["value"] + 1}


        builder = StateGraph(State)
        builder.add_node("prepare", prepare)
        builder.add_node("finish", finish)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "finish")
        builder.add_edge("finish", END)
        config = thread_config(sys.argv[2])
        with SQLiteCheckpointStore(sys.argv[1]) as store:
            graph = builder.compile(
                checkpointer=store.checkpointer,
                interrupt_before=["finish"],
            )
            result = graph.invoke(
                {"value": 0},
                config=config,
                durability="sync",
            )
            assert result == {"value": 1}
            assert graph.get_state(config).next == ("finish",)
        """
    )
    process_b = textwrap.dedent(
        """
        import sys
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from insight_agent.research import SQLiteCheckpointStore, thread_config


        class State(TypedDict):
            value: int


        def prepare(state: State) -> dict[str, int]:
            raise AssertionError("prepare must not replay after its checkpoint")


        def finish(state: State) -> dict[str, int]:
            return {"value": state["value"] + 1}


        builder = StateGraph(State)
        builder.add_node("prepare", prepare)
        builder.add_node("finish", finish)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "finish")
        builder.add_edge("finish", END)
        config = thread_config(sys.argv[2])
        with SQLiteCheckpointStore(sys.argv[1]) as store:
            graph = builder.compile(checkpointer=store.checkpointer)
            before = graph.get_state(config)
            assert before.values == {"value": 1}
            assert before.next == ("finish",)
            result = graph.invoke(None, config=config, durability="sync")
            assert result == {"value": 2}
            assert graph.get_state(config).next == ()
        """
    )

    first = subprocess.run(
        [sys.executable, "-c", process_a, str(path), thread_id],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr

    second = subprocess.run(
        [sys.executable, "-c", process_b, str(path), thread_id],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr


def test_checkpoint_readiness_uses_existing_connection(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "ready.sqlite")
    try:
        assert store.check_ready() is None
    finally:
        store.close()
