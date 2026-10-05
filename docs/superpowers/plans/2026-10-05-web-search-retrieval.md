# Web Search Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Web-routed research tasks through Tavily candidate discovery and the existing URL Loader, then retain task-scoped results in Research State.

**Architecture:** Add a provider-neutral `web_search` package with environment-backed configuration, a thin Tavily REST adapter, and a retriever that composes search with the public URL Loader. Inject the retriever into the existing LangGraph workflow without changing graph topology, and expose WEB to the router only when configuration succeeds.

**Tech Stack:** Python 3.11+, dataclasses, Protocol, httpx, LangGraph, pytest.

**Repository constraint:** Do not commit or push any changes.

---

### Task 1: Web Search configuration and domain models

**Files:**
- Create: `insight_agent/web_search/__init__.py`
- Create: `insight_agent/web_search/models.py`
- Create: `insight_agent/web_search/errors.py`
- Create: `insight_agent/web_search/config.py`
- Test: `tests/test_web_search_config.py`
- Test: `tests/test_web_search_provider.py`

- [x] Write failing tests for the three dataclasses and `WebSearchConfig.from_env()` defaults, explicit values, missing optional key, required-key error, and invalid numeric values.
- [x] Run the two new test files and confirm import/behavior failures are caused by the absent package.
- [x] Implement frozen/slotted data models, explicit error classes, and validated configuration with these public contracts:

```python
@dataclass(frozen=True, slots=True)
class WebSearchHit:
    rank: int
    title: str
    url: str
    snippet: str
    score: float | None = None

@dataclass(frozen=True, slots=True)
class WebFetchFailure:
    url: str
    reason: str

@dataclass(frozen=True, slots=True)
class WebRetrievalResult:
    query: str
    hits: list[WebSearchHit]
    documents: list[Document]
    failures: list[WebFetchFailure]

WebSearchConfig.from_env(environ, required=False) -> WebSearchConfig | None
```

- [x] Re-run configuration/model tests and confirm they pass.

### Task 2: Tavily Search Provider

**Files:**
- Create: `insight_agent/web_search/provider.py`
- Modify: `insight_agent/web_search/__init__.py`
- Test: `tests/test_web_search_provider.py`

- [x] Write failing tests that patch `httpx.post`, assert the REST request contains only candidate-search options, verify rank order and optional score mapping, and require request/status/JSON/schema errors to become `WebSearchError`.
- [x] Run provider tests and confirm the provider is missing.
- [x] Implement `SearchProvider` and `TavilySearchProvider`:

```python
class SearchProvider(Protocol):
    def search(self, query: str, *, limit: int) -> list[WebSearchHit]: ...

class TavilySearchProvider:
    def __init__(self, *, api_key: str, timeout: float = 30.0) -> None: ...
    def search(self, query: str, *, limit: int) -> list[WebSearchHit]: ...
```

  POST to `https://api.tavily.com/search`, request `query`, `max_results`,
  `include_answer=False`, `include_raw_content=False`, and map only `results`.
- [x] Re-run provider tests and confirm they pass.

### Task 3: Web Retriever

**Files:**
- Create: `insight_agent/web_search/retriever.py`
- Modify: `insight_agent/web_search/__init__.py`
- Test: `tests/test_web_retriever.py`

- [x] Write failing fake-driven tests for query/search-limit propagation, exact URL deduplication, first-occurrence order, fetch limit, existing metadata preservation, search metadata, optional score omission, URL Loader failures, empty documents/content, continuation after failure, and provider error propagation.
- [x] Run retriever tests and confirm the retriever is missing.
- [x] Implement an injected URL loader callable whose default delegates to public `insight_agent.ingestion.load_url`; pass configured timeout, copy Documents before metadata enrichment, and return `WebRetrievalResult`.
- [x] Re-run retriever tests and confirm they pass.

### Task 4: Research State and Web workflow node

**Files:**
- Modify: `insight_agent/planning/models.py`
- Modify: `insight_agent/research/workflow.py`
- Modify: `tests/test_research_workflow.py`

- [x] Add failing tests requiring `web_results` to default empty, Web tasks to call the injected retriever with `task.question`, local/vision branches not to call it, multiple Web task results to accumulate, unavailable retrievers to fail clearly, and all tasks still to reach END.
- [x] Run workflow tests and confirm the new assertions fail for missing state/retrieval behavior.
- [x] Extend canonical and graph state, inject a `WebTaskRetriever | None`, and change only `web_entry` so it records the route and returns `{**old_results, task.id: result}`.
- [x] Re-run workflow tests and confirm they pass.

### Task 5: Runtime composition and documentation

**Files:**
- Modify: `insight_agent/__main__.py`
- Modify: `insight_agent/planning/coordinator.py`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `tests/test_main.py`
- Modify: `tests/test_coordinator.py`

- [x] Add failing tests proving WEB is omitted without `TAVILY_API_KEY`, included with valid Web configuration, the workflow receives a retriever, and formatted research context no longer claims Web Search is unimplemented.
- [x] Run composition/coordinator tests and confirm expected failures.
- [x] Construct config/provider/retriever in `_build_app()`, preserve local and vision sources, update context wording to treat Web documents as retrieval material rather than Evidence, and document the four environment variables.
- [x] Re-run composition/coordinator tests and confirm they pass.

### Task 6: Verification

**Files:**
- Review every changed file; do not commit or push.

- [x] Run all new Web Search tests.
- [x] Run workflow, coordinator, main, routing, ingestion, and app regression tests.
- [x] Run the complete suite with the confirmed `insight-agent` Conda environment.
- [x] Run `git diff --check`, inspect the full diff, and run `git status --short`.
- [x] Report exact commands/counts, real-network status, configuration, exclusions, and only observed problems.
