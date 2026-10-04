"""Tests for the thin handoff from planning to the existing agent loop."""

from __future__ import annotations

from insight_agent.planning import (
    ResearchCoordinator,
    ResearchPlan,
    ResearchState,
    ResearchTask,
    format_research_context,
)


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
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        research_agent=agent,  # type: ignore[arg-type]
    )

    answer = coordinator.run("研究 Agent Memory")

    assert answer == "research answer"
    assert planner.calls == ["研究 Agent Memory"]
    assert coordinator.last_state == ResearchState(
        query="研究 Agent Memory",
        plan=plan,
    )
    assert len(agent.calls) == 1


def test_execution_context_preserves_query_and_structured_plan() -> None:
    planner = FakePlanner(_plan())
    agent = FakeResearchAgent()
    coordinator = ResearchCoordinator(
        planner=planner,  # type: ignore[arg-type]
        research_agent=agent,  # type: ignore[arg-type]
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


def test_context_formatter_requires_a_completed_plan() -> None:
    state = ResearchState(query="query")

    try:
        format_research_context(state)
    except ValueError as exc:
        assert "plan" in str(exc)
    else:
        raise AssertionError("formatting an unplanned state must fail")
