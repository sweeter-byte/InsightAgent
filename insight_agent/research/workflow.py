"""LangGraph workflow for sequential per-task source routing."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from insight_agent.evidence import Evidence, EvidenceCollector
from insight_agent.planning.models import ResearchPlan, ResearchState, ResearchTask
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing.models import (
    RetrievalSource,
    RouteDecision,
    RoutingError,
)
from insight_agent.web_search.errors import WebSearchConfigurationError
from insight_agent.web_search.models import WebRetrievalResult
from insight_agent.vision_retrieval.errors import VisionRetrievalError
from insight_agent.vision_retrieval.models import VisionRetrievalResult


class TaskRouter(Protocol):
    def route(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        available_sources: set[RetrievalSource],
    ) -> RouteDecision:
        """Choose one available source for ``task`` without retrieving it."""
        ...


class LocalTaskRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return raw Local retrieval results through the shared retriever."""
        ...


class WebTaskRetriever(Protocol):
    def retrieve(self, query: str) -> WebRetrievalResult:
        """Fetch public Web material for one already-routed task."""
        ...


class VisionTaskRetriever(Protocol):
    def retrieve(
        self,
        *,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
    ) -> VisionRetrievalResult:
        """Analyze original images for one already-routed task."""
        ...


class _GraphState(TypedDict):
    """LangGraph-only schema adapted from the canonical dataclass at the boundary."""

    query: str
    plan: ResearchPlan | None
    available_sources: set[RetrievalSource]
    task_index: int
    current_task: ResearchTask | None
    current_route: RouteDecision | None
    route_decisions: list[RouteDecision]
    local_results: dict[str, list[RetrievalResult]]
    web_results: dict[str, WebRetrievalResult]
    vision_results: dict[str, VisionRetrievalResult]
    evidence_pool: dict[str, list[Evidence]]


class ResearchRoutingWorkflow:
    """Run every planned task through one source branch in plan order."""

    def __init__(
        self,
        router: TaskRouter,
        web_retriever: WebTaskRetriever | None = None,
        vision_retriever: VisionTaskRetriever | None = None,
        local_retriever: LocalTaskRetriever | None = None,
    ) -> None:
        self.router = router
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever
        self.vision_retriever = vision_retriever
        self.evidence_collector = EvidenceCollector()
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(_GraphState)
        builder.add_node("select_task", self._select_task_node)
        builder.add_node("route_task", self._route_task_node)
        builder.add_node("local_entry", self._local_entry_node)
        builder.add_node("web_entry", self._web_entry_node)
        builder.add_node("vision_entry", self._vision_entry_node)
        builder.add_node("collect_evidence", self._collect_evidence_node)
        builder.add_node("advance_task", self._advance_task_node)

        builder.add_edge(START, "select_task")
        builder.add_edge("select_task", "route_task")
        builder.add_conditional_edges(
            "route_task",
            self._choose_source_edge,
            {
                RetrievalSource.LOCAL.value: "local_entry",
                RetrievalSource.WEB.value: "web_entry",
                RetrievalSource.VISION.value: "vision_entry",
            },
        )
        for entry_node in ("local_entry", "web_entry", "vision_entry"):
            builder.add_edge(entry_node, "collect_evidence")
        builder.add_edge("collect_evidence", "advance_task")
        builder.add_conditional_edges(
            "advance_task",
            self._next_step_edge,
            {"select_task": "select_task", END: END},
        )
        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final canonical state."""
        if not isinstance(state, ResearchState):
            raise RoutingError("workflow state must be a ResearchState")
        if state.plan is None:
            raise RoutingError("research state must contain a plan")
        if not state.available_sources:
            raise RoutingError("research state available_sources must not be empty")
        if any(
            not isinstance(source, RetrievalSource)
            for source in state.available_sources
        ):
            raise RoutingError(
                "research state available_sources must contain RetrievalSource values"
            )

        result = self.graph.invoke(self._to_graph_state(state))
        final_state = self._from_graph_state(result)
        if len(final_state.route_decisions) != len(final_state.plan.tasks):
            raise RoutingError(
                "workflow must produce exactly one route decision per research task"
            )
        return final_state

    def select_task(self, state: ResearchState) -> dict[str, Any]:
        """Select the task at ``task_index`` and clear the previous route."""
        if state.plan is None:
            raise RoutingError("cannot select a task without a research plan")
        if not 0 <= state.task_index < len(state.plan.tasks):
            raise RoutingError(
                f"research task_index {state.task_index} is outside the plan"
            )
        return {
            "current_task": state.plan.tasks[state.task_index],
            "current_route": None,
        }

    def route_task(self, state: ResearchState) -> dict[str, RouteDecision]:
        """Ask the injected router to choose a source for the current task."""
        if state.plan is None:
            raise RoutingError("cannot route a task without a research plan")
        if state.current_task is None:
            raise RoutingError("cannot route without a current research task")
        decision = self.router.route(
            task=state.current_task,
            objective=state.plan.objective,
            constraints=state.plan.constraints,
            available_sources=state.available_sources,
        )
        if not isinstance(decision, RouteDecision):
            raise RoutingError("task router must return a RouteDecision")
        if decision.task_id != state.current_task.id:
            raise RoutingError("route decision task_id does not match current task")
        if not isinstance(decision.source, RetrievalSource):
            raise RoutingError("route decision source must be a RetrievalSource")
        if decision.source not in state.available_sources:
            raise RoutingError(
                f"retrieval source {decision.source.value!r} is not available "
                "for this run"
            )
        return {"current_route": decision}

    @staticmethod
    def choose_source(state: ResearchState) -> str:
        """Return the conditional-edge key for the current route."""
        if state.current_route is None:
            raise RoutingError("cannot choose a source without a route decision")
        return state.current_route.source.value

    def local_entry(self, state: ResearchState) -> dict[str, Any]:
        """Retrieve Local chunks for the current task and retain raw results."""
        route_update = self._record_current_route(state, RetrievalSource.LOCAL)
        if state.current_task is None:
            raise RoutingError("cannot retrieve local results without a current task")
        if (
            state.current_route is None
            or state.current_route.task_id != state.current_task.id
        ):
            raise RoutingError("local route does not correspond to current task")
        if self.local_retriever is None:
            raise RoutingError("Local Retrieval is not configured for this workflow")
        results = self.local_retriever.retrieve(state.current_task.question)
        if not isinstance(results, list) or any(
            not isinstance(result, RetrievalResult) for result in results
        ):
            raise RoutingError(
                "local retriever must return a list of RetrievalResult objects"
            )
        return {
            **route_update,
            "local_results": {
                **state.local_results,
                state.current_task.id: results,
            },
        }

    def web_entry(self, state: ResearchState) -> dict[str, Any]:
        """Retrieve public pages for the current Web-routed task."""
        route_update = self._record_current_route(state, RetrievalSource.WEB)
        if state.current_task is None:
            raise RoutingError("cannot retrieve web results without a current task")
        if (
            state.current_route is None
            or state.current_route.task_id != state.current_task.id
        ):
            raise RoutingError("web route does not correspond to current task")
        if self.web_retriever is None:
            raise WebSearchConfigurationError(
                "Web Search is not configured for this workflow"
            )
        result = self.web_retriever.retrieve(state.current_task.question)
        if not isinstance(result, WebRetrievalResult):
            raise RoutingError("web retriever must return a WebRetrievalResult")
        return {
            **route_update,
            "web_results": {
                **state.web_results,
                state.current_task.id: result,
            },
        }

    def vision_entry(self, state: ResearchState) -> dict[str, Any]:
        """Analyze original images for the current Vision-routed task."""
        route_update = self._record_current_route(state, RetrievalSource.VISION)
        if state.plan is None:
            raise RoutingError("cannot retrieve vision results without a plan")
        if state.current_task is None:
            raise RoutingError("cannot retrieve vision results without a current task")
        if (
            state.current_route is None
            or state.current_route.task_id != state.current_task.id
        ):
            raise RoutingError("vision route does not correspond to current task")
        if self.vision_retriever is None:
            raise VisionRetrievalError(
                "Vision Retrieval is not configured for this workflow"
            )
        result = self.vision_retriever.retrieve(
            task=state.current_task,
            objective=state.plan.objective,
            constraints=state.plan.constraints,
        )
        if not isinstance(result, VisionRetrievalResult):
            raise RoutingError(
                "vision retriever must return a VisionRetrievalResult"
            )
        if result.task_id != state.current_task.id:
            raise RoutingError(
                "vision retrieval result task_id does not match current task"
            )
        if result.query != state.current_task.question:
            raise RoutingError(
                "vision retrieval result query does not match current task"
            )
        return {
            **route_update,
            "vision_results": {
                **state.vision_results,
                state.current_task.id: result,
            },
        }

    def collect_evidence(self, state: ResearchState) -> dict[str, Any]:
        """Normalize the current task's routed raw results into Evidence."""
        task = state.current_task
        route = state.current_route
        if task is None:
            raise RoutingError("cannot collect evidence without a current task")
        if route is None or route.task_id != task.id:
            raise RoutingError("cannot collect evidence for a mismatched route")

        if route.source is RetrievalSource.LOCAL:
            if task.id not in state.local_results:
                raise RoutingError("current task has no Local retrieval result")
            incoming = self.evidence_collector.collect_local(
                task.id,
                state.local_results[task.id],
            )
        elif route.source is RetrievalSource.WEB:
            if task.id not in state.web_results:
                raise RoutingError("current task has no Web retrieval result")
            incoming = self.evidence_collector.collect_web(
                task.id,
                state.web_results[task.id],
            )
        elif route.source is RetrievalSource.VISION:
            if task.id not in state.vision_results:
                raise RoutingError("current task has no Vision retrieval result")
            incoming = self.evidence_collector.collect_vision(
                task.id,
                state.vision_results[task.id],
            )
        else:
            raise RoutingError("cannot collect evidence for an unknown source")

        return {
            "evidence_pool": {
                **state.evidence_pool,
                task.id: self.evidence_collector.merge(
                    state.evidence_pool.get(task.id, []),
                    incoming,
                ),
            }
        }

    @staticmethod
    def _record_current_route(
        state: ResearchState,
        expected_source: RetrievalSource,
    ) -> dict[str, list[RouteDecision]]:
        decision = state.current_route
        if decision is None:
            raise RoutingError("cannot record an absent route decision")
        if decision.source is not expected_source:
            raise RoutingError("route decision entered the wrong source branch")
        if state.plan is None or not 0 <= state.task_index < len(state.plan.tasks):
            raise RoutingError("cannot record a route for an invalid task_index")
        expected_task = state.plan.tasks[state.task_index]
        if decision.task_id != expected_task.id:
            raise RoutingError("route decision does not correspond to current task")
        if len(state.route_decisions) != state.task_index:
            raise RoutingError("route decision sequence is out of sync with task_index")
        return {"route_decisions": [*state.route_decisions, decision]}

    @staticmethod
    def advance_task(state: ResearchState) -> dict[str, Any]:
        """Move to the next index and clear transient per-task state."""
        return {
            "task_index": state.task_index + 1,
            "current_task": None,
            "current_route": None,
        }

    @staticmethod
    def next_step(state: ResearchState) -> str:
        """Loop when tasks remain; otherwise terminate the graph."""
        if state.plan is None:
            raise RoutingError("cannot advance without a research plan")
        return "select_task" if state.task_index < len(state.plan.tasks) else END

    @staticmethod
    def _to_graph_state(state: ResearchState) -> _GraphState:
        return {
            "query": state.query,
            "plan": state.plan,
            "available_sources": set(state.available_sources),
            "task_index": state.task_index,
            "current_task": state.current_task,
            "current_route": state.current_route,
            "route_decisions": list(state.route_decisions),
            "local_results": {
                task_id: list(results)
                for task_id, results in state.local_results.items()
            },
            "web_results": dict(state.web_results),
            "vision_results": dict(state.vision_results),
            "evidence_pool": {
                task_id: list(evidence)
                for task_id, evidence in state.evidence_pool.items()
            },
        }

    @staticmethod
    def _from_graph_state(state: _GraphState) -> ResearchState:
        return ResearchState(
            query=state["query"],
            plan=state["plan"],
            available_sources=set(state["available_sources"]),
            task_index=state["task_index"],
            current_task=state["current_task"],
            current_route=state["current_route"],
            route_decisions=list(state["route_decisions"]),
            local_results={
                task_id: list(results)
                for task_id, results in state["local_results"].items()
            },
            web_results=dict(state["web_results"]),
            vision_results=dict(state["vision_results"]),
            evidence_pool={
                task_id: list(evidence)
                for task_id, evidence in state["evidence_pool"].items()
            },
        )

    def _select_task_node(self, state: _GraphState) -> dict[str, Any]:
        return self.select_task(self._from_graph_state(state))

    def _route_task_node(self, state: _GraphState) -> dict[str, RouteDecision]:
        return self.route_task(self._from_graph_state(state))

    def _local_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, Any]:
        return self.local_entry(self._from_graph_state(state))

    def _web_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, Any]:
        return self.web_entry(self._from_graph_state(state))

    def _vision_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, Any]:
        return self.vision_entry(self._from_graph_state(state))

    def _collect_evidence_node(self, state: _GraphState) -> dict[str, Any]:
        return self.collect_evidence(self._from_graph_state(state))

    def _advance_task_node(self, state: _GraphState) -> dict[str, Any]:
        return self.advance_task(self._from_graph_state(state))

    def _choose_source_edge(self, state: _GraphState) -> str:
        return self.choose_source(self._from_graph_state(state))

    def _next_step_edge(self, state: _GraphState) -> str:
        return self.next_step(self._from_graph_state(state))
