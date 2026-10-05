# Web Search Retrieval Design

## Scope

Chapter 9 turns the existing `web_entry` routing destination into a real,
network-backed retrieval branch. It discovers public candidate pages with
Tavily Search, fetches a small subset through the existing URL Loader, and
stores task-scoped retrieval results. Search hits and fetched documents remain
raw retrieval material, not Evidence, Citation, confidence, or a report.

## Architecture

- `insight_agent.web_search.models` defines provider-neutral `WebSearchHit`,
  `WebFetchFailure`, and `WebRetrievalResult` dataclasses.
- `insight_agent.web_search.errors` defines explicit configuration and provider
  errors without exposing Tavily or `httpx` exception types to callers.
- `insight_agent.web_search.config` owns `TAVILY_API_KEY`, timeout, search-limit,
  and fetch-limit parsing. An absent key means Web Search is unavailable rather
  than preventing the rest of the application from starting.
- `insight_agent.web_search.provider` defines the small `SearchProvider`
  protocol and implements Tavily's REST Search endpoint with the repository's
  existing `httpx` dependency. The provider requests candidate results only and
  maps `title`, `url`, `content`, and optional `score` into `WebSearchHit`.
- `insight_agent.web_search.retriever` injects a provider and URL-loading
  callable. It preserves provider hit order, removes only exact duplicate URLs,
  fetches at most `fetch_limit` unique candidates, and collects per-page
  failures while continuing with later candidates.
- `insight_agent.research.workflow` receives an optional `WebRetriever`. Its
  existing graph topology is unchanged; only `web_entry` gains retrieval and
  task-keyed state updates.

## Data flow and state

For a Web-routed task, `web_entry` uses `current_task.question` unchanged as the
query. `WebRetriever` passes the configured limit to the provider, then fetches
the first unique candidate URLs with the existing public `load_url` API. Each
non-empty returned `Document` is copied with its original metadata plus
`search_query`, `search_rank`, `search_title`, `search_snippet`, and optional
`search_score`. Web content is stored as inert data and never controls routing,
configuration, graph edges, prompts, tools, or permissions.

`ResearchState` and the LangGraph boundary schema add:

```python
web_results: dict[str, WebRetrievalResult]
```

The workflow returns a new mapping containing all earlier entries plus the
current task ID. Because this graph has no mapping reducer, explicit copying is
required to prevent one task from replacing another task's results.

## Configuration and composition

The environment-backed configuration uses:

- `TAVILY_API_KEY`
- `WEB_SEARCH_TIMEOUT` (default `30.0` seconds)
- `WEB_SEARCH_LIMIT` (default `5`)
- `WEB_FETCH_LIMIT` (default `3`)

All numeric values are validated as positive, and fetch limit may not exceed
search limit. `_build_app()` asks the configuration layer for an optional Web
Search configuration. If present, it constructs the Tavily provider and Web
Retriever and includes `RetrievalSource.WEB`; if absent, it omits WEB before
the router runs. Existing local and vision availability behavior remains
unchanged.

## Errors and partial failure

Missing configuration is reported clearly when code explicitly requests a
configured provider. Tavily request failures, non-success status codes, invalid
JSON, and malformed result objects become `WebSearchError`; provider failure is
not converted into an empty successful result.

Once hits exist, each URL Loader exception, empty loader result, or empty page
body becomes one `WebFetchFailure(url, reason)`. Retrieval then proceeds to the
next unique candidate. Failure reasons contain concise exception messages, not
tracebacks. A successful retrieval may therefore contain both documents and
failures.

## Testing

All normal tests are network-free. Provider tests patch the HTTP boundary and
cover ordered mapping, optional scores, request parameters, and error
normalization. Retriever tests use a fake provider and fake URL Loader to cover
query/limit propagation, exact deduplication, fetch limits, metadata merging,
empty content, partial failures, and provider-wide failure propagation.

Workflow tests inject a fake retriever and cover the Web branch, local/vision
isolation, multi-Web-task accumulation, and normal termination. Composition and
configuration tests verify WEB availability is conditional on a valid API key.
The existing complete suite remains the regression gate. No default test calls
Tavily or arbitrary public pages.

## Explicit exclusions

This chapter does not implement Evidence, Citation, reports, query rewriting,
multi-query or iterative search, Tavily answers/raw content/extract/crawl/
research, browser automation, retries, authority scoring, parallel retrieval,
or Vision Retrieval.
