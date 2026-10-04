# Retrieval Routing Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every planned research task to one allowed source through a sequential LangGraph workflow before handing the enriched context to the existing research agent.

**Architecture:** Keep planning models and the existing agent/retrieval boundaries intact. Add a strict LLM-backed routing package, extend the canonical dataclass state, and compile a fixed LangGraph workflow once per injected workflow object. Integrate only through `ResearchCoordinator` and CLI composition.

**Tech Stack:** Python 3.11+, dataclasses, OpenAI-compatible `LLMClient`, LangGraph `StateGraph`, pytest.

---

### Task 1: Routing domain model and strict router

**Files:**
- Create: `insight_agent/routing/__init__.py`
- Create: `insight_agent/routing/models.py`
- Create: `insight_agent/routing/router.py`
- Test: `tests/test_retrieval_router.py`

- [x] Write tests for all valid source values, `tools=None`, prompt constraints,
  runtime-owned task IDs, input validation, malformed output, and unavailable
  sources.
- [x] Run `conda run --no-capture-output -n insight-agent pytest -q tests/test_retrieval_router.py`
  and confirm collection fails because the routing package is absent.
- [x] Implement the enum, decision/error types, dedicated prompt, one LLM call,
  exact JSON parsing, and deterministic validation.
- [x] Re-run the router test file and confirm it passes.

The public call contract is:

```python
decision = RetrievalRouter(llm).route(
    task=task,
    objective=plan.objective,
    constraints=plan.constraints,
    available_sources={RetrievalSource.LOCAL, RetrievalSource.WEB},
)
```

### Task 2: Canonical state and LangGraph workflow

**Files:**
- Modify: `insight_agent/planning/models.py`
- Modify: `insight_agent/planning/__init__.py`
- Create: `insight_agent/research/__init__.py`
- Create: `insight_agent/research/workflow.py`
- Modify: `pyproject.toml`
- Test: `tests/test_research_workflow.py`

- [x] Write tests for state defaults, task selection, every conditional branch,
  one decision per task, stable multi-task order, final cleanup, and graph
  termination.
- [x] Run the workflow test file and confirm it fails because workflow/state
  support and LangGraph are absent.
- [x] Add `langgraph` using the repository's lower-bound dependency style and
  install the project into the confirmed `insight-agent` Conda environment.
- [x] Extend `ResearchState` without importing routing at runtime from the
  planning model module.
- [x] Implement and compile this graph:

```text
START -> select_task -> route_task
route_task -(source)-> local_entry | web_entry | vision_entry
each entry -> advance_task
advance_task -(remaining tasks)-> select_task | END
```

- [x] Re-run workflow and router tests and confirm they pass.

### Task 3: Coordinator and execution context

**Files:**
- Modify: `insight_agent/planning/coordinator.py`
- Modify: `insight_agent/planning/__init__.py`
- Modify: `tests/test_coordinator.py`

- [x] Update coordinator tests first so a fake workflow must run between planner
  and agent, `last_state` must be the returned final state, and the formatted
  context must include task-aligned route decisions plus non-evidence/source
  capability warnings.
- [x] Run `conda run --no-capture-output -n insight-agent pytest -q tests/test_coordinator.py`
  and confirm failures describe the missing workflow integration.
- [x] Add a workflow protocol and source-set constructor dependency, then render
  the complete plan and decisions without treating them as evidence.
- [x] Re-run coordinator tests and confirm they pass.

### Task 4: Composition and CLI error handling

**Files:**
- Modify: `insight_agent/__main__.py`
- Modify: `tests/test_main.py`
- Verify: `tests/test_app.py`

- [x] Add tests requiring one shared LLM across intent router, planner, retrieval
  router, and agent; explicit all-source CLI state configuration; workflow
  injection; and friendly `RoutingError` handling.
- [x] Run the main/app tests and confirm the new assertions fail before wiring.
- [x] Construct `RetrievalRouter` and `ResearchRoutingWorkflow` once in
  `_build_app()`, pass all three routable sources to the coordinator, and catch
  `RoutingError` in `_run_once()`.
- [x] Re-run the main/app tests and confirm direct/analyze behavior is unchanged.

### Task 5: Regression verification

**Files:**
- Review all changed files; no commit or push.

- [x] Run:

```bash
conda run --no-capture-output -n insight-agent pytest -q \
  tests/test_retrieval_router.py \
  tests/test_research_workflow.py \
  tests/test_coordinator.py \
  tests/test_app.py \
  tests/test_main.py
```

- [x] Run `conda run --no-capture-output -n insight-agent pytest -q`.
- [x] Run `git diff --check` and inspect `git diff --stat` plus the final diff.
- [x] Compare the result line-by-line with the approved Chapter 8 requirements
  and report any deliberate deviation.
