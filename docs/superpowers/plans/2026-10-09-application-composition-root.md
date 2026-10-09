# Application Configuration and Composition Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize InsightAgent backend configuration and dependency assembly so the CLI and HTTP Runtime share one testable, resource-safe composition root.

**Architecture:** Add an immutable `AppConfig` aggregator and one public `build_application` function under `insight_agent.application`. Keep component-level environment constructors for earlier chapters, but make official entry points pass resolved values explicitly. Use `ExitStack` for reverse-order, idempotent cleanup while preserving lazy local retrieval and the conditional Vision probe.

**Tech Stack:** Python 3.11 dataclasses, `contextlib.ExitStack`, existing OpenAI/Qdrant/LangGraph/FastAPI/Redis adapters, pytest and pytest-asyncio.

---

## File map

- Create `insight_agent/application/{__init__,config,bootstrap}.py` for the public integration layer.
- Create `tests/application/{__init__,test_config,test_bootstrap}.py` for mapping-based configuration, wiring, laziness, fake injection, and cleanup tests.
- Modify `insight_agent/llm.py`, `indexing/vector_store.py`, `retrieval/tool.py`, and `runtime/policies.py` to accept explicit resolved configuration while retaining legacy defaults.
- Modify `insight_agent/ingestion/vision.py` and `web_search/config.py` so credentials never appear in dataclass representations.
- Modify `insight_agent/__main__.py` and `runtime/app.py` to consume the public integration layer.
- Modify existing component, CLI, and Runtime tests plus `.env.example`, `README.md`, and `README.zh.md`.

### Task 1: Reusable leaf configuration values

**Files:**
- Modify: `insight_agent/llm.py`
- Modify: `insight_agent/ingestion/vision.py`
- Modify: `insight_agent/web_search/config.py`
- Modify: `insight_agent/indexing/vector_store.py`
- Modify: `insight_agent/indexing/__init__.py`
- Modify: `insight_agent/runtime/policies.py`
- Test: `tests/test_llm.py`
- Test: `tests/test_vector_store.py`
- Test: `tests/test_vision.py`
- Test: `tests/test_web_search_config.py`
- Test: `tests/runtime/test_app.py`

- [ ] **Step 1: Write failing explicit LLM configuration tests**

Add to `tests/test_llm.py`:

```python
from insight_agent.llm import LLMClient, LLMConfig


def test_llm_config_reads_mapping_without_exposing_secret() -> None:
    config = LLMConfig.from_env({
        "LLM_API_KEY": "super-secret",
        "LLM_BASE_URL": "https://example.invalid/v1",
        "LLM_MODEL": "model-a",
    })
    assert config.model == "model-a"
    assert "super-secret" not in repr(config)


def test_llm_client_accepts_config_and_closes_once(monkeypatch) -> None:
    class FakeSDK:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr("insight_agent.llm.OpenAI", FakeSDK)
    client = LLMClient(LLMConfig("secret", "https://example.invalid/v1", "model"))
    client.close()
    client.close()
    assert client.client.kwargs["api_key"] == "secret"
    assert client.client.close_calls == 1
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/test_llm.py -q`.

Expected: import failure for `LLMConfig`.

- [ ] **Step 3: Implement `LLMConfig` and explicit client ownership**

Add to `insight_agent/llm.py`:

```python
@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str
    model: str

    def __post_init__(self) -> None:
        for name in ("api_key", "base_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLM {name} must not be empty")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LLMConfig":
        values = os.environ if environ is None else environ
        resolved: dict[str, str] = {}
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            value = values.get(name, "").strip()
            if not value:
                raise RuntimeError(
                    f"Missing required environment variable {name!r}. "
                    f"{_ENV_VAR_HINTS[name]} See `.env.example` for reference."
                )
            resolved[name] = value
        return cls(resolved["LLM_API_KEY"], resolved["LLM_BASE_URL"], resolved["LLM_MODEL"])
```

Change `LLMClient.__init__` to accept `config: LLMConfig | None = None`, use
`config or LLMConfig.from_env()`, store `_closed = False`, and add an idempotent
`close()` that closes the SDK client once. Keep `_require_env` for the legacy
Vision helper.

- [ ] **Step 4: Write failing Qdrant and Runtime configuration tests**

Add to `tests/test_vector_store.py`:

```python
def test_qdrant_config_reads_explicit_mapping() -> None:
    config = QdrantConfig.from_env({
        "QDRANT_PATH": "/tmp/configured-qdrant",
        "QDRANT_COLLECTION": "configured_collection",
    })
    assert config.path == Path("/tmp/configured-qdrant")
    assert config.collection_name == "configured_collection"
```

Add to `tests/runtime/test_app.py`:

```python
def test_runtime_config_reads_explicit_mapping() -> None:
    config = RuntimeConfig.from_env({
        "RUNTIME_REDIS_URL": "redis://runtime.example/8",
        "RUNTIME_MAX_CONCURRENCY": "4",
        "RUNTIME_RUN_TIMEOUT_SECONDS": "12.5",
    })
    assert config.redis_url == "redis://runtime.example/8"
    assert config.policy.max_concurrency == 4
    assert config.policy.run_timeout_seconds == 12.5
```

- [ ] **Step 5: Verify RED**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/test_vector_store.py tests/runtime/test_app.py::test_runtime_config_reads_explicit_mapping -q
```

Expected: import failures for `QdrantConfig` and `RuntimeConfig`.

- [ ] **Step 6: Implement Qdrant and Runtime value types**

In `indexing/vector_store.py`, add immutable `QdrantConfig(path,
collection_name)` with a mapping-aware `from_env`. Update `QdrantVectorStore`
to accept keyword-only `config: QdrantConfig | None`; reject combining it with
the legacy `path` or `collection_name` arguments, and otherwise preserve current
defaults. Export `QdrantConfig` from `indexing/__init__.py`.

```python
@dataclass(frozen=True, slots=True)
class QdrantConfig:
    path: Path = Path(DEFAULT_QDRANT_PATH)
    collection_name: str = DEFAULT_QDRANT_COLLECTION

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("QDRANT_PATH must not be empty")
        if not isinstance(self.collection_name, str) or not self.collection_name.strip():
            raise ValueError("QDRANT_COLLECTION must not be empty")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "QdrantConfig":
        values = os.environ if environ is None else environ
        path = values.get("QDRANT_PATH", DEFAULT_QDRANT_PATH).strip()
        collection = values.get("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
        return cls(
            Path(path or DEFAULT_QDRANT_PATH),
            collection or DEFAULT_QDRANT_COLLECTION,
        )


def __init__(
    self,
    path: str | Path | None = None,
    collection_name: str | None = None,
    *,
    config: QdrantConfig | None = None,
) -> None:
    if config is not None and (path is not None or collection_name is not None):
        raise ValueError("config cannot be combined with path or collection_name")
    resolved = config or QdrantConfig.from_env()
    self.path = Path(path) if path is not None else resolved.path
    self.collection_name = collection_name or resolved.collection_name
    self.client = QdrantClient(path=str(self.path))
```

In `runtime/policies.py`, change `RuntimePolicy.from_env` and its numeric helpers
to accept an explicit mapping, then add:

```python
DEFAULT_REDIS_URL = "redis://localhost:6379/0"

@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    redis_url: str = DEFAULT_REDIS_URL
    policy: RuntimePolicy = RuntimePolicy()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        values = os.environ if environ is None else environ
        url = values.get("RUNTIME_REDIS_URL", DEFAULT_REDIS_URL).strip()
        return cls(url or DEFAULT_REDIS_URL, RuntimePolicy.from_env(values))
```

Validate non-empty direct `redis_url` values in `__post_init__`.

- [ ] **Step 7: Make existing credential reprs secret-safe**

Use `field(repr=False)` for `VisionModelConfig.api_key` and
`WebSearchConfig.api_key`. Extend existing tests to assert their configured
secret values are absent from `repr(config)`.

- [ ] **Step 8: Verify GREEN and commit**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/test_llm.py tests/test_vector_store.py tests/test_vision.py \
  tests/test_web_search_config.py \
  tests/runtime/test_app.py::test_runtime_policy_reads_environment \
  tests/runtime/test_app.py::test_runtime_config_reads_explicit_mapping -q
```

Expected: PASS.

Commit:

```bash
git add insight_agent tests/test_llm.py tests/test_vector_store.py \
  tests/test_vision.py tests/test_web_search_config.py tests/runtime/test_app.py
git commit -m "refactor: expose explicit component configuration"
```

### Task 2: Aggregate `AppConfig`

**Files:**
- Create: `insight_agent/application/__init__.py`
- Create: `insight_agent/application/config.py`
- Create: `tests/application/__init__.py`
- Create: `tests/application/test_config.py`
- Modify: `insight_agent/indexing/__init__.py`

- [ ] **Step 1: Write failing aggregate tests**

Create `tests/application/test_config.py`:

```python
from pathlib import Path
import pytest
from insight_agent.application.config import AppConfig, CheckpointConfig


def required() -> dict[str, str]:
    return {
        "LLM_API_KEY": "llm-secret",
        "LLM_BASE_URL": "https://llm.invalid/v1",
        "LLM_MODEL": "llm-model",
        "RERANKER_MODEL": "reranker-model",
    }


def test_app_config_aggregates_all_startup_settings() -> None:
    config = AppConfig.from_env({
        **required(),
        "EMBEDDING_MODEL": "embedding-model",
        "QDRANT_PATH": "/tmp/qdrant-app",
        "QDRANT_COLLECTION": "documents",
        "CHECKPOINT_PATH": "/tmp/checkpoints.sqlite",
        "TAVILY_API_KEY": "tavily-secret",
        "VISION_API_KEY": "vision-secret",
        "VISION_BASE_URL": "https://vision.invalid/v1",
        "VISION_MODEL": "vision-model",
        "RUNTIME_REDIS_URL": "redis://runtime.example/2",
        "RUNTIME_MAX_CONCURRENCY": "5",
    })
    assert config.llm.model == "llm-model"
    assert config.vision is not None
    assert config.embedding_model == "embedding-model"
    assert config.qdrant.path == Path("/tmp/qdrant-app")
    assert config.web_search is not None
    assert config.checkpoint.path == Path("/tmp/checkpoints.sqlite")
    assert config.runtime.policy.max_concurrency == 5
    assert all(secret not in repr(config) for secret in (
        "llm-secret", "vision-secret", "tavily-secret"
    ))


def test_app_config_defaults_optional_sources_and_paths() -> None:
    config = AppConfig.from_env(required())
    assert config.vision is None
    assert config.web_search is None
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.qdrant.path == Path(".data/qdrant")
    assert config.checkpoint == CheckpointConfig(Path(".insight_agent/checkpoints.sqlite"))


def test_app_config_reuses_hybrid_validation() -> None:
    with pytest.raises(RuntimeError, match="HYBRID_FINAL_TOP_K"):
        AppConfig.from_env({**required(), "HYBRID_FINAL_TOP_K": "9"})
```

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/application/test_config.py -q`.

Expected: `insight_agent.application` is missing.

- [ ] **Step 3: Implement the aggregate**

Create immutable `CheckpointConfig(path=DEFAULT_CHECKPOINT_PATH)` and
`AppConfig` in `application/config.py` with these fields:

```python
llm: LLMConfig
vision: VisionModelConfig | None
retrieval: HybridRetrievalConfig
embedding_model: str
qdrant: QdrantConfig
web_search: WebSearchConfig | None
checkpoint: CheckpointConfig
runtime: RuntimeConfig
```

`AppConfig.from_env(environ=None)` must call `load_dotenv()` only when no mapping
is supplied, then pass the same mapping into every existing `from_env` validator.
It reads `EMBEDDING_MODEL` with `DEFAULT_EMBEDDING_MODEL` and `CHECKPOINT_PATH`
with `DEFAULT_CHECKPOINT_PATH`. It must not log or interpolate any credential.
Export `AppConfig` and `CheckpointConfig` from `application/__init__.py` and
`DEFAULT_EMBEDDING_MODEL` from `indexing/__init__.py`.

```python
@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    path: Path = DEFAULT_CHECKPOINT_PATH

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("CHECKPOINT_PATH must not be empty")


@dataclass(frozen=True, slots=True)
class AppConfig:
    llm: LLMConfig
    vision: VisionModelConfig | None
    retrieval: HybridRetrievalConfig
    embedding_model: str
    qdrant: QdrantConfig
    web_search: WebSearchConfig | None
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_model, str) or not self.embedding_model.strip():
            raise ValueError("EMBEDDING_MODEL must not be empty")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AppConfig":
        if environ is None:
            load_dotenv()
            values: Mapping[str, str] = os.environ
        else:
            values = environ
        embedding = values.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        checkpoint = values.get("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)).strip()
        return cls(
            llm=LLMConfig.from_env(values),
            vision=VisionModelConfig.from_env(values, required=False),
            retrieval=HybridRetrievalConfig.from_env(values),
            embedding_model=embedding or DEFAULT_EMBEDDING_MODEL,
            qdrant=QdrantConfig.from_env(values),
            web_search=WebSearchConfig.from_env(values, required=False),
            checkpoint=CheckpointConfig(Path(checkpoint or DEFAULT_CHECKPOINT_PATH)),
            runtime=RuntimeConfig.from_env(values),
        )
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/application/test_config.py tests/test_hybrid_config.py \
  tests/test_web_search_config.py tests/test_llm.py -q
```

Expected: PASS.

Commit:

```bash
git add insight_agent/application insight_agent/indexing/__init__.py tests/application
git commit -m "feat: add centralized application configuration"
```

### Task 3: Canonical composition root and cleanup

**Files:**
- Create: `insight_agent/application/bootstrap.py`
- Modify: `insight_agent/application/__init__.py`
- Modify: `insight_agent/retrieval/tool.py`
- Modify: `insight_agent/app.py`
- Create: `tests/application/test_bootstrap.py`

- [ ] **Step 1: Write failing wiring, laziness, and cleanup tests**

Create fakes for LLM and checkpoint resources and assert:

```python
def test_build_application_shares_llm_and_keeps_qdrant_lazy() -> None:
    llm = FakeLLM()
    checkpoint = FakeCheckpointStore()
    factories = ApplicationFactories(
        llm=lambda _config: llm,
        checkpoint=lambda _config: checkpoint,
        qdrant=lambda _config: (_ for _ in ()).throw(
            AssertionError("Qdrant must stay lazy")
        ),
    )
    app = build_application(base_config(), factories=factories)
    assert app.llm is app.router.llm is app.research_agent.llm
    assert app.llm is app.research_coordinator.planner.llm
    assert app.research_coordinator.available_sources == {RetrievalSource.LOCAL}
    app.close()
    app.close()
    assert llm.close_calls == 1
    assert checkpoint.close_calls == 1


def test_build_application_closes_llm_when_checkpoint_creation_fails() -> None:
    llm = FakeLLM()
    factories = ApplicationFactories(
        llm=lambda _config: llm,
        checkpoint=lambda _config: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError, match="failed"):
        build_application(base_config(), factories=factories)
    assert llm.close_calls == 1
```

`base_config()` constructs `AppConfig` directly with no external endpoints.

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/application/test_bootstrap.py -q`.

Expected: missing `application.bootstrap`.

- [ ] **Step 3: Pass explicit retrieval settings without defeating laziness**

Extend `build_default_hybrid_retriever` with keyword-only `qdrant_config` and
`embedding_model`. When it owns the store, construct
`QdrantVectorStore(config=qdrant_config)`; always construct
`SentenceTransformerEmbedder(embedding_model)`. Preserve legacy defaults when
both arguments are omitted and preserve close-on-construction-failure behavior.

- [ ] **Step 4: Implement `ApplicationFactories` and `build_application`**

In `application/bootstrap.py`, define default wrapper functions and:

```python
@dataclass(frozen=True, slots=True)
class ApplicationFactories:
    llm: Callable[[LLMConfig], Any] = _create_llm
    checkpoint: Callable[[CheckpointConfig], Any] = _create_checkpoint
    qdrant: Callable[[QdrantConfig], Any] = _create_qdrant
    vision_client: Callable[[VisionModelConfig], Any] = _create_vision_client
    hybrid: Callable[[HybridRetrievalConfig, QdrantConfig, str, Any | None], Any] = _create_hybrid
```

Implement `build_application(config, *, factories=None)` with `ExitStack`:

1. Create/register LLM.
2. Build/register optional Vision resources.
3. Create/register checkpoint store.
4. Create/register `KnowledgeSearchTool`; without Vision, its retriever factory
   captures explicit retrieval/Qdrant/embedding settings and remains lazy.
5. Move the unchanged Router/Agent/Planner/Workflow wiring from `__main__`.
6. Transfer `resources.pop_all().close` into the returned `InsightAgent`.
7. On any exception, close all already-acquired resources and re-raise.

Add `InsightAgent.add_close_callback(callback)` so bootstrap does not mutate a
private list directly. Export `ApplicationFactories` and `build_application`.

The implementation skeleton is:

```python
def _own(stack: ExitStack, resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        stack.callback(close)


def build_application(
    config: AppConfig,
    *,
    factories: ApplicationFactories | None = None,
) -> InsightAgent:
    resolved = factories or ApplicationFactories()
    resources = ExitStack()
    try:
        llm = resolved.llm(config.llm)
        _own(resources, llm)
        vision = _build_vision_runtime(config, resolved)
        if vision is not None:
            resources.callback(vision.close)
        checkpoint = resolved.checkpoint(config.checkpoint)
        _own(resources, checkpoint)
        knowledge = _build_knowledge_tool(config, resolved, vision)
        resources.callback(knowledge.close)
        app = _compose_application(
            config=config,
            llm=llm,
            knowledge=knowledge,
            vision=vision,
            checkpoint=checkpoint,
        )
        owned = resources.pop_all()
        app.add_close_callback(owned.close)
        return app
    except BaseException:
        resources.close()
        raise


def _build_knowledge_tool(
    config: AppConfig,
    factories: ApplicationFactories,
    vision: _VisionRuntime | None,
) -> KnowledgeSearchTool:
    if vision is not None:
        return KnowledgeSearchTool(
            vision.hybrid,
            default_top_k=config.retrieval.final_top_k,
        )
    return KnowledgeSearchTool(
        retriever_factory=lambda: factories.hybrid(
            config.retrieval,
            config.qdrant,
            config.embedding_model,
            None,
        ),
        default_top_k=config.retrieval.final_top_k,
    )
```

`_compose_application` contains the existing constructors verbatim, substitutes
`config.web_search` for the old `from_env()` call, uses
`checkpoint.checkpointer`, and returns `InsightAgent` without its own resource
callbacks. The transferred `ExitStack` is the sole owner.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/application/test_bootstrap.py tests/test_retrieval.py \
  tests/test_hybrid_retrieval.py tests/test_app.py -q
```

Expected: PASS without opening Qdrant or loading model weights.

Commit:

```bash
git add insight_agent/application insight_agent/retrieval/tool.py \
  insight_agent/app.py tests/application/test_bootstrap.py
git commit -m "feat: add canonical application composition root"
```

### Task 4: Vision sharing and exact resource ownership

**Files:**
- Modify: `insight_agent/application/bootstrap.py`
- Modify: `tests/application/test_bootstrap.py`

- [ ] **Step 1: Write failing Vision tests**

Using fake Qdrant, hybrid, and Vision clients plus a temporary accessible PNG,
assert that configured Vision:

- receives the exact `QdrantConfig`, retrieval config, and embedding model;
- shares one hybrid retriever with `KnowledgeSearchTool`;
- adds `RetrievalSource.VISION` only with an accessible indexed image;
- closes Vision client and Qdrant exactly once after repeated app close;
- closes the probe store immediately and never creates a Vision client when the
  collection/image is unavailable;
- closes the probe store if later hybrid or Vision-client construction fails.

The fake store returns a real `Chunk` with `SourceType.IMAGE`, ensuring the test
exercises existing source validation rather than a mock-only branch.

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/application/test_bootstrap.py -q`.

Expected: new Vision tests fail before the probe holder is implemented.

- [ ] **Step 3: Implement the Vision holder and probe**

Define private `_VisionRuntime(retriever, hybrid, _resources)` whose `close()`
closes a nested `ExitStack`. Move `_is_accessible_image` from `__main__.py` into
bootstrap. Preserve current warning-and-disable behavior for Qdrant open and
inspection errors. Create the hybrid and Vision client only after availability
is proven, transfer the nested stack on success, and close it on every early
return or exception. A shared hybrid does not own the injected Qdrant store; the
Vision resource holder owns it.

```python
@dataclass(slots=True)
class _VisionRuntime:
    retriever: VisionRetriever
    hybrid: HybridRetriever
    resources: ExitStack

    def close(self) -> None:
        self.resources.close()


def _build_vision_runtime(
    config: AppConfig,
    factories: ApplicationFactories,
) -> _VisionRuntime | None:
    if config.vision is None:
        return None
    resources = ExitStack()
    try:
        try:
            store = factories.qdrant(config.qdrant)
        except Exception as exc:
            logger.warning("Vision Retrieval disabled: cannot open Qdrant: %s", exc)
            return None
        _own(resources, store)
        try:
            available = store.collection_exists() and any(
                _is_accessible_image(chunk.source)
                for chunk in store.iter_chunks(source_types={"image"})
            )
        except Exception as exc:
            logger.warning("Vision Retrieval disabled: cannot inspect indexed images: %s", exc)
            return None
        if not available:
            return None
        hybrid = factories.hybrid(
            config.retrieval,
            config.qdrant,
            config.embedding_model,
            store,
        )
        client = factories.vision_client(config.vision)
        _own(resources, client)
        owned = resources.pop_all()
        return _VisionRuntime(
            retriever=VisionRetriever(
                hybrid_retriever=hybrid,
                analyzer=VisionAnalyzer(image_request=client.analyze_image),
            ),
            hybrid=hybrid,
            resources=owned,
        )
    finally:
        resources.close()
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/application/test_bootstrap.py tests/test_vision_retriever.py \
  tests/test_hybrid_retrieval.py tests/test_retrieval.py -q
```

Expected: PASS.

Commit:

```bash
git add insight_agent/application/bootstrap.py tests/application/test_bootstrap.py
git commit -m "refactor: centralize vision resource ownership"
```

### Task 5: Thin CLI adapter

**Files:**
- Modify: `insight_agent/__main__.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Replace private-builder tests with a failing public injection test**

Delete tests coupled to `_build_app`, `_compose_app`, and
`_build_vision_runtime`. Retain CLI output/error/REPL tests, injecting a fake via:

```python
rc = cli.main(
    ["question"],
    config_factory=lambda: config,
    application_factory=lambda value: (built.append(value), fake)[1],
)
assert rc == 0
assert built == [config]
assert fake.calls == ["question"]
assert fake.close_calls == 1
```

Add a config-failure case where `config_factory` raises and assert exit code 2.

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/test_main.py -q`.

Expected: `main` rejects the factory keyword arguments.

- [ ] **Step 3: Remove all CLI construction logic**

Import `AppConfig` and `build_application`. Change `main` to:

```python
def main(
    argv: Optional[list[str]] = None,
    *,
    config_factory: Callable[[], AppConfig] = AppConfig.from_env,
    application_factory: Callable[[AppConfig], InsightAgent] = build_application,
) -> int:
```

Resolve config and build once inside the existing config-error `try`, preserve
the REPL/error behavior and `finally: app.close()`, and delete all old private
builders and their imports. There must be no compatibility aliases.

- [ ] **Step 4: Verify GREEN and commit**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/test_main.py tests/application tests/test_app.py -q`.

Expected: PASS.

Commit:

```bash
git add insight_agent/__main__.py tests/test_main.py
git commit -m "refactor: make cli consume application composition root"
```

### Task 6: HTTP Runtime on the shared root

**Files:**
- Modify: `insight_agent/runtime/app.py`
- Modify: `tests/runtime/test_app.py`

- [ ] **Step 1: Rewrite lifespan tests without the broken TestClient boundary**

Replace the two `TestClient` lifecycle tests with `pytest.mark.asyncio` tests
using `async with app.router.lifespan_context(app)`. Inject an object config,
fake application, and fake Redis. Assert build/ping/reconcile happen once and
normal close order releases service, Redis, and application. Add separate Redis
factory and Redis ping failure tests proving the already-built application is
closed exactly once.

- [ ] **Step 2: Verify RED**

Run `conda run --no-capture-output -n insight-agent python -m pytest tests/runtime/test_app.py -q`.

Expected: `create_runtime_app` lacks canonical config/application factory
parameters.

- [ ] **Step 3: Implement lazy Runtime composition**

Use this boundary:

```python
def create_runtime_app(
    *,
    config: AppConfig | None = None,
    config_factory: Callable[[], AppConfig] = AppConfig.from_env,
    application_factory: Callable[[AppConfig], ComposedResearchApp] = build_application,
    redis_client_factory: Callable[[str], Any] = _redis_from_url,
    runtime_config: RuntimeConfig | None = None,
) -> FastAPI:
```

Inside lifespan, resolve `config or config_factory()`, select
`runtime_config or resolved_config.runtime`, build the application once, then
create/ping Redis and construct the existing registry/events/runner/service.
Keep nested cleanup order: service, Redis, application. Leave module-level
`app = create_runtime_app()` free of configuration parsing and resource
acquisition; all work starts in lifespan.

```python
@asynccontextmanager
async def lifespan(application: FastAPI):
    resolved_config = config or config_factory()
    resolved_runtime = runtime_config or resolved_config.runtime
    research_app = application_factory(resolved_config)
    try:
        redis_client = redis_client_factory(resolved_runtime.redis_url)
    except BaseException:
        research_app.close()
        raise
    service: ResearchRuntimeService | None = None
    try:
        await redis_client.ping()
        registry = RedisRunRegistry(redis_client)
        events = RedisRuntimeEventStore(
            redis_client,
            event_ttl_seconds=resolved_runtime.policy.event_ttl_seconds,
        )
        service = ResearchRuntimeService(
            registry=registry,
            events=events,
            runner=LangGraphResearchRunner(research_app.research_coordinator),
            policy=resolved_runtime.policy,
        )
        application.state.redis = redis_client
        application.state.research_app = research_app
        application.state.runtime_service = service
        await service.reconcile_interrupted_runs()
        yield
    finally:
        try:
            if service is not None:
                await service.close()
        finally:
            try:
                await redis_client.aclose()
            finally:
                research_app.close()
```

- [ ] **Step 4: Verify GREEN and commit**

Run:

```bash
conda run --no-capture-output -n insight-agent python -m pytest \
  tests/runtime/test_app.py tests/runtime/test_runner.py \
  tests/runtime/test_service.py tests/runtime/test_stores.py \
  tests/runtime/test_redis_stores.py tests/runtime/test_models.py \
  tests/runtime/test_checkpoint_integration.py -q
```

Expected: PASS. `tests/runtime/test_api.py` remains excluded because its
pre-existing `TestClient.__enter__()` hang was reproduced on unmodified `main`.

Commit:

```bash
git add insight_agent/runtime/app.py tests/runtime/test_app.py
git commit -m "refactor: share composition root with http runtime"
```

### Task 7: Documentation and final verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `README.zh.md`

- [ ] **Step 1: Document configuration and compatibility boundaries**

Add:

```dotenv
# SQLite path for LangGraph research checkpoints.
CHECKPOINT_PATH=.insight_agent/checkpoints.sqlite
```

Document `AppConfig` and `build_application` as the canonical CLI/HTTP path,
the lazy local retriever and conditional Vision probe, and the retained
environment defaults of standalone earlier-chapter APIs.

- [ ] **Step 2: Run static checks**

```bash
git diff --check
conda run --no-capture-output -n insight-agent python -m compileall -q insight_agent tests
```

Expected: exit 0 with no output.

- [ ] **Step 3: Run the complete non-TestClient suite**

```bash
conda run --no-capture-output -n insight-agent \
  python -m pytest --ignore=tests/runtime/test_api.py -q
```

Expected: PASS, including every Evaluation test.

- [ ] **Step 4: Reconfirm the approved baseline limitation**

```bash
timeout 10s conda run --no-capture-output -n insight-agent python -m pytest \
  tests/runtime/test_api.py::test_create_returns_202_without_exposing_research_state -q
```

Expected: exit 124, matching the pre-existing Starlette
`TestClient.__enter__()` hang on unmodified `main`.

- [ ] **Step 5: Audit scope and obsolete entry points**

```bash
rg -n "_build_app|_compose_app|_build_vision_runtime" insight_agent tests
rg -n "from_env\(" insight_agent/application insight_agent/__main__.py insight_agent/runtime/app.py
git diff --name-only d01b46f
```

Expected: no old CLI builder names; official entry points route through
`AppConfig`; no `evals/`, `tests/evals/`, frontend, or dependency file changes.

- [ ] **Step 6: Commit docs and review the branch**

```bash
git add .env.example README.md README.zh.md
git commit -m "docs: explain centralized application startup"
git status --short
git log --oneline d01b46f..HEAD
git diff --stat d01b46f..HEAD
```

Expected: clean worktree with focused configuration/composition commits.
