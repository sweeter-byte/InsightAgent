from __future__ import annotations

from evals import DirectResearchWorkflowRunner, EvalCase
from insight_agent.evidence import Evidence
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import Claim, ReportSection, StructuredReport
from insight_agent.routing import RetrievalSource, RouteDecision
from insight_agent.web_search import WebFetchFailure, WebRetrievalResult


class FakePlanner:
    def plan(self, query: str) -> ResearchPlan:
        return ResearchPlan(
            objective=query,
            tasks=[ResearchTask(id="T1", question=query)],
        )


class FakeWorkflow:
    def __init__(self) -> None:
        self.states: list[ResearchState] = []

    def run(self, state: ResearchState) -> ResearchState:
        self.states.append(state)
        evidence = Evidence(
            id="E1",
            task_id="T1",
            retrieval_source=RetrievalSource.WEB,
            origin_id="https://fixture.invalid/page",
            content="Recorded evidence.",
            source="https://fixture.invalid/page",
            metadata={"final_url": "https://fixture.invalid/page"},
        )
        state.route_decisions.extend(
            [
                RouteDecision("T1", RetrievalSource.WEB, "first attempt"),
                RouteDecision("T1", RetrievalSource.WEB, "retry"),
            ]
        )
        state.web_results["T1"] = WebRetrievalResult(
            query=state.query,
            hits=[],
            documents=[],
            failures=[
                WebFetchFailure(
                    url="https://fixture.invalid/missing",
                    reason="recorded failure",
                )
            ],
        )
        state.evidence_pool["T1"] = [evidence]
        state.report = StructuredReport(
            objective=state.query,
            sections=[
                ReportSection(
                    task_id="T1",
                    title="Result",
                    claims=[Claim("C1", "T1", "Recorded evidence.", ["E1"])],
                )
            ],
        )
        state.final_output = "# Result\n\nRecorded evidence."
        return state


def test_direct_workflow_runner_reuses_workflow_without_runtime_infrastructure() -> None:
    workflow = FakeWorkflow()
    ticks = iter((10.000, 10.025))
    runner = DirectResearchWorkflowRunner(
        planner=FakePlanner(),
        workflow=workflow,
        available_sources={RetrievalSource.WEB},
        clock=lambda: next(ticks),
    )

    output = runner.run(EvalCase(case_id="web-1", query="fixed query"))

    assert workflow.states[0].available_sources == {RetrievalSource.WEB}
    assert output.state is workflow.states[0]
    assert output.report is output.state.report
    assert output.evidence_by_id == {"E1": output.state.evidence_pool["T1"][0]}
    assert output.runtime.completed is True
    assert output.runtime.latency_ms == 25
    assert output.runtime.prompt_tokens is None
    assert output.runtime.completion_tokens is None
    assert output.runtime.retrieval_calls == 2
    assert output.runtime.failed_calls == 1
    assert output.artifacts["final_output"] == output.state.final_output
    assert "report" in output.artifacts
    assert "evidence_pool" in output.artifacts


def test_direct_workflow_runner_rejects_incomplete_output() -> None:
    class IncompleteWorkflow:
        def run(self, state: ResearchState) -> ResearchState:
            return state

    runner = DirectResearchWorkflowRunner(
        planner=FakePlanner(),
        workflow=IncompleteWorkflow(),
        available_sources={RetrievalSource.LOCAL},
    )

    try:
        runner.run(EvalCase(case_id="broken", query="query"))
    except RuntimeError as exc:
        assert "report" in str(exc)
    else:
        raise AssertionError("incomplete workflow output must fail")
