# Retrieval Routing Workflow Design

## Scope

Chapter 8 adds source selection and sequential workflow orchestration between the
existing `ResearchPlanner` and `ResearchAgent`. It does not execute web or vision
retrieval and does not change the existing local hybrid retrieval implementation.

## Architecture

- `insight_agent.routing.models` defines `RetrievalSource`, `RouteDecision`, and
  `RoutingError`.
- `insight_agent.routing.router` owns the source-selection prompt, calls the
  existing `LLMClient` with `tools=None`, and strictly validates its JSON result.
- The canonical `ResearchState` remains in `insight_agent.planning.models` and is
  extended with routing workflow fields. Runtime imports are arranged so
  planning and routing do not form a cycle.
- `insight_agent.research.workflow` compiles one LangGraph `StateGraph` during
  construction. Its nodes are `select_task`, `route_task`, `local_entry`,
  `web_entry`, `vision_entry`, and `advance_task`. Conditional edges select a
  source branch and either continue with the next task or reach `END`.
- `ResearchCoordinator` remains the research-path façade and runs planner,
  workflow, context formatting, and the existing agent in that order.

## State and data flow

`ResearchState` retains `query` and `plan` and adds `available_sources`,
`task_index`, `current_task`, `current_route`, and `route_decisions`. Entry nodes
append exactly one current decision. `advance_task` increments the index and
clears transient task and route fields. Decisions therefore retain plan order.

The CLI explicitly enables all three routable sources. Web and vision are valid
workflow destinations but their entry nodes only record control information.
The execution context tells the downstream agent that these routes have not
performed retrieval and cannot be substituted with model memory or local RAG.

## Validation and errors

The router rejects invalid task/objective/constraints/source-set inputs, empty
responses, invalid or non-object JSON, missing fields, invalid reasons, unknown
sources, and sources outside the allowed set. It never repairs or guesses model
output. `RoutingError` is handled by the CLI alongside the existing planning and
agent errors.

The workflow rejects missing plans, empty allowed-source sets, invalid task
indices, absent current tasks/routes at the relevant nodes, and incomplete final
decision sequences with clear errors rather than incidental index/attribute
failures.

## Testing

Network-free fakes cover all three route values, strict failures, prompt/tool
constraints, runtime-owned task IDs, every workflow branch, ordered multi-task
iteration and termination, coordinator state/context handoff, application path
isolation, CLI wiring, and CLI error reporting. Existing retrieval and agent
tests remain unchanged except where Chapter 8 composition requires new
assertions.

