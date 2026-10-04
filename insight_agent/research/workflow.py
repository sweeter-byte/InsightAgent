"""LangGraph workflow for sequential per-task source routing."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from insight_agent.planning.models import ResearchPlan, ResearchState, ResearchTask
from insight_agent.routing.models import (
    RetrievalSource,
    RouteDecision,
    RoutingError,
)


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


class _GraphState(TypedDict):
    """LangGraph-only schema adapted from the canonical dataclass at the boundary."""

    query: str
    plan: ResearchPlan | None
    available_sources: set[RetrievalSource]
    task_index: int
    current_task: ResearchTask | None
    current_route: RouteDecision | None
    route_decisions: list[RouteDecision]


class ResearchRoutingWorkflow:
    """Run every planned task through one source branch in plan order."""

    def __init__(self, router: TaskRouter) -> None:
        self.router = router
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(_GraphState)
        builder.add_node("select_task", self._select_task_node)
        builder.add_node("route_task", self._route_task_node)
        builder.add_node("local_entry", self._local_entry_node)
        builder.add_node("web_entry", self._web_entry_node)
        builder.add_node("vision_entry", self._vision_entry_node)
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
            builder.add_edge(entry_node, "advance_task")
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

    def local_entry(self, state: ResearchState) -> dict[str, list[RouteDecision]]:
        """Record a local route without executing local retrieval."""
        return self._record_current_route(state, RetrievalSource.LOCAL)

    def web_entry(self, state: ResearchState) -> dict[str, list[RouteDecision]]:
        """Record a web route without executing web search."""
        return self._record_current_route(state, RetrievalSource.WEB)

    def vision_entry(self, state: ResearchState) -> dict[str, list[RouteDecision]]:
        """Record a vision route without executing vision retrieval."""
        return self._record_current_route(state, RetrievalSource.VISION)

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
        )

    def _select_task_node(self, state: _GraphState) -> dict[str, Any]:
        return self.select_task(self._from_graph_state(state))

    def _route_task_node(self, state: _GraphState) -> dict[str, RouteDecision]:
        return self.route_task(self._from_graph_state(state))

    def _local_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, list[RouteDecision]]:
        return self.local_entry(self._from_graph_state(state))

    def _web_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, list[RouteDecision]]:
        return self.web_entry(self._from_graph_state(state))

    def _vision_entry_node(
        self,
        state: _GraphState,
    ) -> dict[str, list[RouteDecision]]:
        return self.vision_entry(self._from_graph_state(state))

    def _advance_task_node(self, state: _GraphState) -> dict[str, Any]:
        return self.advance_task(self._from_graph_state(state))

    def _choose_source_edge(self, state: _GraphState) -> str:
        return self.choose_source(self._from_graph_state(state))

    def _next_step_edge(self, state: _GraphState) -> str:
        return self.next_step(self._from_graph_state(state))
