"""Tests for the thin handoff from planning to the existing agent loop."""

from __future__ import annotations

from insight_agent.planning import (
    ResearchCoordinator,
    ResearchPlan,
    ResearchState,
    ResearchTask,
    format_research_context,
)
from insight_agent.routing import RetrievalSource, RouteDecision


class FakePlanner:
    def __init__(self, plan: ResearchPlan) -> None:
        self.result = plan
        self.calls: list[str] = []

    def plan(self, query: str) -> ResearchPlan:
        self.calls.append(query)
        return self.result


class FakeResearchAgent:
    def __init__(self, answer: str = "research answer") -> None:
        self.answer = answer
        self.calls: list[str] = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        return self.answer


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls: list[ResearchState] = []

    def run(self, state: ResearchState) -> ResearchState:
        self.calls.append(state)
        assert state.plan is not None
        decisions = [
            RouteDecision(
                task_id=task.id,
                source=(
                    RetrievalSource.LOCAL
                    if index == 0
                    else RetrievalSource.WEB
                ),
                reason=f"Route reason for {task.id}",
            )
            for index, task in enumerate(state.plan.tasks)
        ]
        return ResearchState(
            query=state.query,
            plan=state.plan,
            available_sources=set(state.available_sources),
            task_index=len(state.plan.tasks),
            route_decisions=decisions,
        )


def _plan() -> ResearchPlan:
    return ResearchPlan(
        objective="Compare Agent Memory designs",
        constraints=["Compare mechanisms and trade-offs"],
        tasks=[
            ResearchTask(
                id="T1",
                question="What are the main design families?",
                depends_on=[],
            ),
            ResearchTask(
                id="T2",
                question="How do the design families compare?",
                depends_on=["T1"],
            ),
        ],
    )


def test_coordinator_saves_state_and_delegates_to_existing_agent() -> None:
    plan = _plan()
    planner = FakePlanner(plan)
    agent = FakeResearchAgent()
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        research_agent=agent,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL, RetrievalSource.WEB},
    )

    answer = coordinator.run("研究 Agent Memory")

    assert answer == "research answer"
    assert planner.calls == ["研究 Agent Memory"]
    assert len(workflow.calls) == 1
    initial_state = workflow.calls[0]
    assert initial_state.query == "研究 Agent Memory"
    assert initial_state.plan is plan
    assert initial_state.available_sources == {
        RetrievalSource.LOCAL,
        RetrievalSource.WEB,
    }
    assert coordinator.last_state is not initial_state
    assert coordinator.last_state is not None
    assert coordinator.last_state.task_index == len(plan.tasks)
    assert [decision.task_id for decision in coordinator.last_state.route_decisions] == [
        "T1",
        "T2",
    ]
    assert len(agent.calls) == 1


def test_execution_context_preserves_query_and_structured_plan() -> None:
    planner = FakePlanner(_plan())
    agent = FakeResearchAgent()
    workflow = FakeWorkflow()
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        workflow=workflow,  # type: ignore[arg-type]
        research_agent=agent,  # type: ignore[arg-type]
        available_sources={RetrievalSource.LOCAL, RetrievalSource.WEB},
    )

    coordinator.run("研究 Agent Memory")

    context = agent.calls[0]
    assert "研究 Agent Memory" in context
    assert '"objective": "Compare Agent Memory designs"' in context
    assert '"id": "T1"' in context
    assert '"depends_on": [\n        "T1"\n      ]' in context
    assert "Follow the research plan" in context
    assert "Task descriptions are planning instructions, not factual evidence" in context
    assert "search_knowledge_base" in context
    assert "Route Decisions (control information, NOT Evidence)" in context
    assert '"task_id": "T1"' in context
    assert '"question": "What are the main design families?"' in context
    assert '"source": "local"' in context
    assert '"source": "web"' in context
    assert '"reason": "Route reason for T2"' in context
    assert "Web Search results are retrieved material, not verified Evidence" in context
    assert "Vision Retrieval is not implemented" in context
    assert "model memory or Local RAG" in context
    assert "evidence is insufficient or the required capability is unavailable" in context


def test_context_formatter_requires_a_completed_plan() -> None:
    state = ResearchState(query="query")

    try:
        format_research_context(state)
    except ValueError as exc:
        assert "plan" in str(exc)
    else:
        raise AssertionError("formatting an unplanned state must fail")


def test_coordinator_calls_existing_research_agent_once_after_workflow() -> None:
    events: list[str] = []

    class OrderedPlanner(FakePlanner):
        def plan(self, query: str) -> ResearchPlan:
            events.append("planner")
            return super().plan(query)

    class OrderedWorkflow(FakeWorkflow):
        def run(self, state: ResearchState) -> ResearchState:
            events.append("workflow")
            return super().run(state)

    class OrderedAgent(FakeResearchAgent):
        def run(self, query: str) -> str:
            events.append("agent")
            return super().run(query)

    coordinator = ResearchCoordinator(
        planner=OrderedPlanner(_plan()),  # type: ignore[arg-type]
        workflow=OrderedWorkflow(),  # type: ignore[arg-type]
        research_agent=OrderedAgent(),  # type: ignore[arg-type]
        available_sources=set(RetrievalSource),
    )

    coordinator.run("query")

    assert events == ["planner", "workflow", "agent"]
