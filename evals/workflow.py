"""Direct adapters from existing research components to Evaluation artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from evals.models import EvalCase, RuntimeMetrics
from insight_agent.evidence import Evidence
from insight_agent.planning import ResearchPlan, ResearchState
from insight_agent.reporting import StructuredReport
from insight_agent.routing import RetrievalSource


class EvaluationPlanner(Protocol):
    def plan(self, query: str) -> ResearchPlan: ...


class DirectResearchWorkflow(Protocol):
    def run(self, state: ResearchState) -> ResearchState: ...


class EvaluationWorkflowRunner(Protocol):
    """Evaluation-owned workflow boundary, independent of HTTP and Redis."""

    def run(self, case: EvalCase) -> WorkflowCaseOutput: ...


@dataclass(frozen=True, slots=True)
class WorkflowCaseOutput:
    """Observable complete-research output consumed by layered evaluators."""

    state: ResearchState
    report: StructuredReport
    evidence_by_id: Mapping[str, Evidence]
    runtime: RuntimeMetrics
    artifacts: dict[str, object]


class DirectResearchWorkflowRunner:
    """Call Planner and Research Workflow directly, without Runtime services."""

    def __init__(
        self,
        *,
        planner: EvaluationPlanner,
        workflow: DirectResearchWorkflow,
        available_sources: set[RetrievalSource],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not available_sources or any(
            not isinstance(source, RetrievalSource) for source in available_sources
        ):
            raise ValueError(
                "available_sources must contain at least one RetrievalSource"
            )
        self.planner = planner
        self.workflow = workflow
        self.available_sources = set(available_sources)
        self.clock = clock

    def run(self, case: EvalCase) -> WorkflowCaseOutput:
        if not isinstance(case, EvalCase):
            raise TypeError("case must be an EvalCase")
        started = self.clock()
        state = ResearchState(
            query=case.query,
            plan=self.planner.plan(case.query),
            available_sources=set(self.available_sources),
        )
        final_state = self.workflow.run(state)
        elapsed_ms = max(0, round((self.clock() - started) * 1000))
        if not isinstance(final_state, ResearchState):
            raise TypeError("workflow must return a ResearchState")
        if final_state.report is None:
            raise RuntimeError("research workflow did not produce a structured report")
        if not isinstance(final_state.final_output, str):
            raise RuntimeError("research workflow did not produce final_output")

        evidence_by_id = _flatten_evidence(final_state)
        return WorkflowCaseOutput(
            state=final_state,
            report=final_state.report,
            evidence_by_id=evidence_by_id,
            runtime=RuntimeMetrics(
                completed=True,
                latency_ms=elapsed_ms,
                prompt_tokens=None,
                completion_tokens=None,
                retrieval_calls=len(final_state.route_decisions),
                failed_calls=_failed_retrieval_calls(final_state),
            ),
            artifacts=_artifact_payload(final_state),
        )


def _flatten_evidence(state: ResearchState) -> dict[str, Evidence]:
    evidence_by_id: dict[str, Evidence] = {}
    for task_evidence in state.evidence_pool.values():
        for evidence in task_evidence:
            if evidence.id in evidence_by_id:
                raise RuntimeError(f"duplicate Evidence ID: {evidence.id!r}")
            evidence_by_id[evidence.id] = evidence
    return evidence_by_id


def _failed_retrieval_calls(state: ResearchState) -> int:
    web_failures = sum(len(result.failures) for result in state.web_results.values())
    vision_failures = sum(
        len(result.failures) for result in state.vision_results.values()
    )
    return web_failures + vision_failures


def _artifact_payload(state: ResearchState) -> dict[str, object]:
    return {
        "query": state.query,
        "plan": state.plan,
        "route_decisions": state.route_decisions,
        "local_results": state.local_results,
        "web_results": state.web_results,
        "vision_results": state.vision_results,
        "evidence_pool": state.evidence_pool,
        "report": state.report,
        "final_output": state.final_output,
    }
