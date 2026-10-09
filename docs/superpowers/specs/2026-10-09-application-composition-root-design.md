# Application Configuration and Composition Root Design

## Scope

This change introduces the first integration layer for InsightAgent. It centralizes
the backend application's startup configuration and dependency assembly without
changing the research, retrieval, ingestion, indexing, runtime, or evaluation
algorithms.

The supported application entry points remain:

- `python -m insight_agent` for the CLI and REPL.
- `uvicorn insight_agent.runtime.app:app` for the HTTP research runtime.

Both entry points will use the same public composition function. The CLI module
will no longer contain private dependency builders, and the Runtime module will
no longer import construction logic from the CLI.

Frontend work, new Agent behavior, Evaluation behavior changes, and a general
dependency-injection framework are outside this phase.

## Current State

The complete research object graph is currently assembled in
`insight_agent.__main__`. The Runtime imports the CLI's private `_build_app`
function, then separately reads Redis and execution-policy configuration in
`insight_agent.runtime.app`.

Configuration is split across constructors and `from_env()` methods:

- `LLMClient` reads required `LLM_*` values in its constructor.
- `VisionModelConfig`, `HybridRetrievalConfig`, and `WebSearchConfig` provide
  reusable validation.
- `SentenceTransformerEmbedder` and `QdrantVectorStore` read their own
  environment variables.
- `SQLiteCheckpointStore` uses a fixed default path.
- `RuntimePolicy` reads runtime policy variables, while the Redis URL is read in
  `runtime.app`.
- `TextChunker` reads ingestion/indexing defaults at module import time.

The local knowledge tool deliberately constructs Qdrant, the embedding model,
BM25, and the reranker only on first use. Vision capability detection is
different: when Vision is configured, startup opens Qdrant briefly to confirm
that accessible indexed images exist and keeps shared retrieval resources only
when Vision can be enabled.

## Architecture

Add an `insight_agent.application` package with two focused modules:

- `config.py` owns application-startup configuration values and environment
  parsing.
- `bootstrap.py` owns the object graph and the resources acquired while building
  it.

`insight_agent.app.InsightAgent` remains the user-facing dispatch facade. Existing
routers, agents, planners, retrievers, workflows, checkpoint implementations,
and runtime services are reused rather than wrapped or reimplemented.

There will be one canonical core builder:

```python
config = AppConfig.from_env()
application = build_application(config)
```

The CLI calls this builder directly. The HTTP lifespan calls the same builder
once at startup and shares the resulting coordinator across all requests through
one `ResearchRuntimeService`.

The old `_build_app`, `_compose_app`, and `_build_vision_runtime` functions in
`insight_agent.__main__` are removed rather than retained as compatibility
aliases. They are private implementation details, and keeping them would leave
two apparent composition entry points.

## Configuration Model

`AppConfig` is an immutable dataclass composed from small immutable value
objects. It covers:

- required text LLM endpoint, model, and credential;
- optional Vision configuration through the existing `VisionModelConfig`;
- existing validated `HybridRetrievalConfig` plus the embedding model name;
- Qdrant path and collection name;
- optional Web Search configuration through the existing `WebSearchConfig`;
- SQLite checkpoint path;
- Runtime Redis URL and the existing validated `RuntimePolicy`.

`AppConfig.from_env(environ=None)` accepts an explicit mapping for deterministic
tests and otherwise reads the process environment. Existing environment names
and defaults remain valid. The checkpoint path receives a narrowly scoped
`CHECKPOINT_PATH` override while preserving `.insight_agent/checkpoints.sqlite`
as its default.

Existing validation is reused by passing the same mapping to
`VisionModelConfig.from_env`, `HybridRetrievalConfig.from_env`, and
`WebSearchConfig.from_env`. Runtime numeric parsing continues through
`RuntimePolicy`, extended only as needed to accept an explicit mapping. No new
configuration dependency is introduced.

Credential fields are excluded from dataclass representations. Configuration
errors identify missing or invalid variable names but never include API key
values. Bootstrap code does not log configuration objects.

Centralization applies to the official CLI and HTTP construction paths. Legacy
standalone APIs retain their current no-argument environment behavior so the
examples and earlier chapter interfaces keep working. In particular:

- `LLMClient()` may still resolve `LLM_*` directly when no explicit values are
  supplied.
- `QdrantVectorStore()` and `SentenceTransformerEmbedder()` retain their
  environment-backed defaults.
- `VisionModelConfig.from_env()`, retrieval builders, and
  `RuntimePolicy.from_env()` remain supported.
- `TextChunker`'s `CHUNK_*` import-time defaults remain a documented legacy
  configuration point because the application composition path does not build
  an indexing write pipeline in this phase.

This preserves the first seventeen chapters without maintaining a second
application assembly path.

## Composition and Lazy Loading

`build_application` constructs one shared `LLMClient` and injects it into the
intent router, research agent, planner, retrieval router, evidence grader,
report generator, self-checker, and repairer.

The local knowledge tool remains lazy. The builder captures the resolved
retrieval, embedding, and Qdrant settings in its retriever factory, but does not
open Qdrant or load Sentence Transformer/Cross Encoder models during ordinary
application startup.

When Vision is not configured, bootstrap does not touch Qdrant for Vision. When
Vision is configured, bootstrap uses the configured Qdrant path and collection
to perform the existing image-availability probe. If no usable indexed image is
present, it closes the probe store and omits the Vision source. If Vision is
available, the local knowledge tool and `VisionRetriever` share the same hybrid
retriever and Qdrant store, matching existing behavior.

Web Search remains optional. Absence of `TAVILY_API_KEY` removes WEB from the
coordinator's available sources; invalid values fail configuration loading
rather than silently disabling a configured feature.

The checkpoint store is created once per composed application and its saver is
shared by the research workflow for the application's lifetime.

## Resource Ownership and Failure Handling

The composition root owns every closeable resource created by its default
factories. An `ExitStack`-style resource registry records cleanup immediately
after each acquisition. If a later construction step fails, already-acquired
resources are closed in reverse order before the exception is propagated.

On successful construction, the cleanup registry is transferred to the
application facade. `InsightAgent.close()` remains idempotent and invokes the
single transferred cleanup operation. This avoids duplicated ownership between
the facade, knowledge tool, Vision runtime, and checkpoint store.

The LLM client gains an idempotent close operation for its underlying SDK
client. Lazy retrieval closes nothing if it was never initialized. A shared
Vision/local hybrid closes its Qdrant store exactly once. Cleanup attempts
continue after an individual close failure so one faulty resource does not leak
the remaining resources; the first close failure may be propagated after all
callbacks have run.

The HTTP lifespan owns Runtime-specific resources in this order:

1. Resolve `AppConfig` and build the shared research application.
2. Create and ping the Redis client.
3. Create registry, event store, runner, and runtime service.
4. Reconcile interrupted runs and serve requests.
5. Stop the runtime service, close Redis, then close the research application.

Every failure path releases only resources that were successfully acquired.
Repeated application or service shutdown remains harmless.

## Test Injection

Bootstrap exposes a small immutable factory bundle for construction boundaries
that acquire resources or are expensive to instantiate. Tests can replace the
LLM, checkpoint, Qdrant, Vision-client, and retrieval factories with fakes while
exercising the real wiring logic. The production defaults remain ordinary
functions and classes; no service locator or global container is introduced.

Runtime construction keeps explicit injectable factories for the composed
application and Redis client. It additionally accepts an `AppConfig` or config
factory so tests require neither environment variables nor external services.

Injected factories follow the same ownership contract as production factories:
objects returned to the composition root are application-owned and are closed
when construction fails or the application shuts down. Tests that need
externally owned collaborators can return non-closeable fakes.

## Testing Strategy

Tests move composition assertions from `tests/test_main.py` into focused
application tests. They cover:

- complete and partial environment parsing using an explicit mapping;
- existing variable names, defaults, and validation behavior;
- secret-safe representations and errors;
- one LLM instance shared across all LLM-backed collaborators;
- optional Web and Vision source selection;
- no Qdrant/model initialization for the ordinary lazy local path;
- explicit Qdrant, embedding, and checkpoint settings reaching their factories;
- fake injection without network, Redis, model downloads, or persistent project
  data;
- reverse-order cleanup, partial-build cleanup, idempotent close, and shared
  resource close-once behavior;
- CLI behavior through the public application builder;
- Runtime startup sharing one application/service across requests and closing
  all resources on normal and failed startup.

Existing research, retrieval, persistence, Runtime service, API, and Evaluation
tests remain regression coverage. Evaluation modules and their business logic
are not modified.

## Documentation

`.env.example` documents the checkpoint override and keeps all existing names.
README startup/configuration text is updated only where it currently describes
the CLI as the composition owner. The documentation explicitly distinguishes
the centralized application path from legacy environment-aware standalone
constructors.
