"""Composition contracts exercised without models, network, or databases."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import insight_agent.application as application
from insight_agent.application import AppConfig, CheckpointConfig
from insight_agent.indexing import Chunk, QdrantConfig, make_chunk_id, make_document_id
from insight_agent.ingestion import Document, IngestionError, SourceType, VisionModelConfig
from insight_agent.llm import LLMConfig
from insight_agent.retrieval import HybridRetrievalConfig, KnowledgeSearchTool
from insight_agent.routing import RetrievalSource
from insight_agent.runtime.policies import RuntimeConfig
from insight_agent.web_search import TavilySearchProvider, WebRetriever, WebSearchConfig


class Resource:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.events.append(self.name)


class Checkpoint(Resource):
    checkpointer = None


class Hybrid(Resource):
    def __init__(self, events: list[str]) -> None:
        super().__init__("hybrid", events)
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list:
        self.queries.append((query, top_k))
        return []


def config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(api_key="secret", base_url="https://llm.example/v1", model="llm"),
        vision=None,
        retrieval=HybridRetrievalConfig(
            dense_k=20,
            sparse_k=20,
            rerank_k=20,
            rrf_k=60,
            reranker_model="custom/reranker",
            final_top_k=3,
        ),
        embedding_model="custom/embedder",
        qdrant=QdrantConfig(path=Path("custom/qdrant"), collection_name="custom"),
        web_search=None,
        checkpoint=CheckpointConfig(path=Path("custom/checkpoints.sqlite")),
        runtime=RuntimeConfig(),
    )


def resources():
    events: list[str] = []
    calls: list[tuple] = []
    llm = Resource("llm", events)
    checkpoint = Checkpoint("checkpoint", events)
    hybrid = Hybrid(events)

    def create_llm(value):
        calls.append(("llm", value))
        return llm

    def create_checkpoint(value):
        calls.append(("checkpoint", value))
        return checkpoint

    def create_hybrid(*values):
        calls.append(("hybrid", *values))
        return hybrid

    factories = application.ApplicationFactories(
        llm=create_llm,
        checkpoint=create_checkpoint,
        qdrant=lambda value: pytest.fail("Qdrant must stay lazy without Vision"),
        vision_client=lambda value: pytest.fail("Vision must be absent"),
        hybrid=create_hybrid,
    )
    return factories, llm, checkpoint, hybrid, calls, events


def image_chunk(source: Path) -> Chunk:
    content = f"Indexed image description for {source.name}"
    document = Document(
        content=content,
        source=str(source),
        source_type=SourceType.IMAGE,
        metadata={"file_name": source.name},
    )
    document_id = make_document_id(document)
    return Chunk(
        id=make_chunk_id(document_id, 0, content),
        document_id=document_id,
        content=content,
        source=document.source,
        source_type=document.source_type,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
        metadata=document.metadata,
    )


class QdrantStore(Resource):
    def __init__(self, events: list[str], image: Path) -> None:
        super().__init__("qdrant", events)
        self.collection_present = True
        self.chunks = [image_chunk(image)]
        self.inspections: list[tuple] = []

    def collection_exists(self):
        self.inspections.append(("collection_exists",))
        return self.collection_present

    def iter_chunks(self, *, source_types):
        assert source_types == {"image"}
        self.inspections.append(("iter_chunks", source_types))
        return iter(self.chunks)


class VisionClient(Resource):
    def analyze_image(self, *args, **kwargs):
        pytest.fail("Vision model must not run during composition")


@pytest.fixture
def vision_resources(tmp_path):
    factories, llm, checkpoint, hybrid, calls, events = resources()
    image = tmp_path / "indexed.png"
    image.write_bytes(b"image")
    store = QdrantStore(events, image)
    client = VisionClient("vision", events)
    settings = replace(
        config(),
        vision=VisionModelConfig(
            api_key="vision-secret", base_url="https://vision.example/v1", model="vision"
        ),
    )

    def create_qdrant(value):
        calls.append(("qdrant", value))
        return store

    def create_vision(value):
        calls.append(("vision", value))
        return client

    return SimpleNamespace(
        settings=settings,
        factories=replace(factories, qdrant=create_qdrant, vision_client=create_vision),
        llm=llm,
        checkpoint=checkpoint,
        hybrid=hybrid,
        store=store,
        client=client,
        calls=calls,
        events=events,
        image=image,
    )


def test_vision_client_failure_closes_pending_hybrid_and_store(
    vision_resources, caplog
) -> None:
    setup = vision_resources

    def fail_client(value):
        assert value is setup.settings.vision
        setup.calls.append(("vision", value))
        raise RuntimeError("Vision client failed")

    with pytest.raises(RuntimeError, match="Vision client failed"):
        application.build_application(
            setup.settings,
            factories=replace(setup.factories, vision_client=fail_client),
        )

    assert setup.events == ["hybrid", "qdrant", "llm"]
    assert [call[0] for call in setup.calls] == ["llm", "qdrant", "hybrid", "vision"]
    assert [setup.hybrid.close_calls, setup.store.close_calls, setup.llm.close_calls] == [
        1, 1, 1
    ]
    assert setup.checkpoint.close_calls == setup.client.close_calls == 0
    assert not caplog.records


def test_composition_shares_one_llm_and_keeps_local_retrieval_lazy() -> None:
    factories, llm, checkpoint, hybrid, calls, events = resources()
    settings = config()

    app = application.build_application(settings, factories=factories)
    coordinator = app.research_coordinator
    workflow = coordinator.workflow
    collaborators = [
        app,
        app.router,
        app.research_agent,
        coordinator.planner,
        workflow.router,
        workflow.grader,
        workflow.report_generator,
        workflow.self_checker,
        workflow.report_repairer,
    ]
    assert all(collaborator.llm is llm for collaborator in collaborators)
    assert coordinator.available_sources == {RetrievalSource.LOCAL}
    assert workflow.graph.checkpointer is checkpoint.checkpointer
    assert workflow.local_retriever is app.research_agent.registry.get(
        "search_knowledge_base"
    )
    assert workflow.web_retriever is None
    assert workflow.vision_retriever is None
    assert calls == [("llm", settings.llm), ("checkpoint", settings.checkpoint)]
    app.close()
    app.close()
    assert events == ["checkpoint", "llm"]
    assert hybrid.close_calls == 0


def test_eager_backend_composition_shares_knowledge_resources() -> None:
    factories, llm, checkpoint, hybrid, calls, events = resources()
    store = QdrantStore(events, Path("missing.png"))
    store.collection_present = False
    embedder = object()
    chunker = object()

    def create_qdrant(value):
        calls.append(("qdrant", value))
        return store

    eager_factories = replace(
        factories,
        qdrant=create_qdrant,
        embedder=lambda model: calls.append(("embedder", model)) or embedder,
        chunker=lambda: calls.append(("chunker",)) or chunker,
    )

    app = application.build_application(
        config(), factories=eager_factories, eager_knowledge=True
    )
    tool = app.research_coordinator.workflow.local_retriever

    assert tool._get_retriever() is hybrid
    assert app.knowledge_service.embedder is embedder
    assert app.knowledge_service.chunker is chunker
    assert app.knowledge_service.vector_store is store
    assert app.vector_store is store
    assert app.checkpoint_store is checkpoint
    assert [call[0] for call in calls] == [
        "llm", "qdrant", "embedder", "hybrid", "chunker", "checkpoint"
    ]
    hybrid_args = next(call[1:] for call in calls if call[0] == "hybrid")
    assert hybrid_args[-2:] == (store, embedder)

    app.close()
    assert events == ["checkpoint", "hybrid", "qdrant", "llm"]


def test_eager_hybrid_failure_closes_qdrant_then_llm() -> None:
    factories, llm, checkpoint, hybrid, calls, events = resources()
    store = QdrantStore(events, Path("missing.png"))
    store.collection_present = False

    def fail_hybrid(*values):
        calls.append(("hybrid", *values))
        raise RuntimeError("shared hybrid failed")

    with pytest.raises(RuntimeError, match="shared hybrid failed"):
        application.build_application(
            config(),
            factories=replace(
                factories,
                qdrant=lambda value: store,
                embedder=lambda model: object(),
                hybrid=fail_hybrid,
            ),
            eager_knowledge=True,
        )

    assert events == ["qdrant", "llm"]
    assert store.close_calls == llm.close_calls == 1
    assert checkpoint.close_calls == hybrid.close_calls == 0


def test_lazy_local_search_uses_explicit_settings_and_closes_once() -> None:
    factories, llm, checkpoint, hybrid, calls, events = resources()
    settings = config()
    app = application.build_application(settings, factories=factories)
    tool = app.research_coordinator.workflow.local_retriever

    tool("first query")
    tool.retrieve("second query", top_k=2)

    assert calls[-1] == (
        "hybrid", settings.retrieval, settings.qdrant, settings.embedding_model, None
    )
    assert len([call for call in calls if call[0] == "hybrid"]) == 1
    assert hybrid.queries == [("first query", 3), ("second query", 2)]
    app.close()
    app.close()
    assert events == ["hybrid", "checkpoint", "llm"]
    assert [hybrid.close_calls, checkpoint.close_calls, llm.close_calls] == [1, 1, 1]


def test_optional_web_uses_explicit_provider_and_retriever_settings() -> None:
    factories, *unused = resources()
    web = WebSearchConfig(api_key="web-secret", timeout=12.5, search_limit=8, fetch_limit=4)
    app = application.build_application(
        replace(config(), web_search=web), factories=factories
    )
    coordinator = app.research_coordinator
    retriever = coordinator.workflow.web_retriever

    assert coordinator.available_sources == {RetrievalSource.LOCAL, RetrievalSource.WEB}
    assert isinstance(retriever, WebRetriever)
    assert isinstance(retriever.provider, TavilySearchProvider)
    assert retriever.provider.api_key == web.api_key
    assert retriever.provider.timeout == web.timeout
    assert (retriever.timeout, retriever.search_limit, retriever.fetch_limit) == (12.5, 8, 4)
    app.close()


def test_failure_after_llm_acquisition_closes_llm() -> None:
    factories, llm, *unused = resources()

    def fail_checkpoint(value):
        raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        application.build_application(
            config(), factories=replace(factories, checkpoint=fail_checkpoint)
        )

    assert llm.close_calls == 1


def test_failure_after_knowledge_creation_closes_in_reverse_order(monkeypatch) -> None:
    from insight_agent.application import bootstrap

    factories, llm, checkpoint, hybrid, calls, events = resources()

    class TrackedKnowledgeTool(KnowledgeSearchTool):
        def close(self) -> None:
            events.append("knowledge")
            super().close()

    def fail_planner(*, llm):
        raise RuntimeError("planner failed")

    monkeypatch.setattr(bootstrap, "KnowledgeSearchTool", TrackedKnowledgeTool)
    monkeypatch.setattr(bootstrap, "ResearchPlanner", fail_planner)

    with pytest.raises(RuntimeError, match="planner failed"):
        application.build_application(config(), factories=factories)

    assert events == ["knowledge", "checkpoint", "llm"]
    assert hybrid.close_calls == 0


def test_factories_are_frozen_and_slotted() -> None:
    factories, *unused = resources()
    with pytest.raises(FrozenInstanceError):
        factories.llm = lambda value: None
    assert not hasattr(factories, "__dict__")


def test_vision_shares_hybrid_and_closes_each_resource_once(vision_resources) -> None:
    setup = vision_resources
    app = application.build_application(setup.settings, factories=setup.factories)
    workflow = app.research_coordinator.workflow

    assert workflow.local_retriever is app.research_agent.registry.get(
        "search_knowledge_base"
    )
    assert workflow.local_retriever._get_retriever() is setup.hybrid
    assert workflow.vision_retriever.hybrid_retriever is setup.hybrid
    assert workflow.vision_retriever.analyzer.image_request == setup.client.analyze_image
    assert app.research_coordinator.available_sources == {
        RetrievalSource.LOCAL, RetrievalSource.VISION
    }
    assert [call[0] for call in setup.calls] == [
        "llm", "qdrant", "hybrid", "vision", "checkpoint"
    ]
    assert setup.calls[1][1] is setup.settings.qdrant
    hybrid_arguments = setup.calls[2][1:]
    assert hybrid_arguments[0] is setup.settings.retrieval
    assert hybrid_arguments[1] is setup.settings.qdrant
    assert hybrid_arguments[2] is setup.settings.embedding_model
    assert hybrid_arguments[3] is setup.store
    assert setup.calls[3][1] is setup.settings.vision
    assert setup.store.inspections == [
        ("collection_exists",), ("iter_chunks", {"image"})
    ]
    app.close()
    app.close()
    assert setup.events == ["checkpoint", "hybrid", "vision", "qdrant", "llm"]
    assert [
        setup.hybrid.close_calls, setup.client.close_calls, setup.store.close_calls,
        setup.checkpoint.close_calls, setup.llm.close_calls,
    ] == [1, 1, 1, 1, 1]


def assert_probe_disabled_vision(setup, app) -> None:
    workflow = app.research_coordinator.workflow
    assert workflow.vision_retriever is None
    assert workflow.local_retriever is not None
    expected_sources = {RetrievalSource.LOCAL}
    if setup.settings.web_search is not None:
        expected_sources.add(RetrievalSource.WEB)
    assert app.research_coordinator.available_sources == expected_sources
    assert setup.store.close_calls == 1
    assert setup.events == ["qdrant"]
    assert [call[0] for call in setup.calls] == ["llm", "qdrant", "checkpoint"]
    assert setup.hybrid.close_calls == setup.client.close_calls == 0
    app.close()
    app.close()
    assert setup.events == ["qdrant", "checkpoint", "llm"]
    assert [
        setup.store.close_calls, setup.checkpoint.close_calls, setup.llm.close_calls
    ] == [1, 1, 1]


def test_disabled_vision_never_constructs_vision_resources(vision_resources) -> None:
    setup = vision_resources
    app = application.build_application(
        replace(setup.settings, vision=None), factories=setup.factories
    )

    assert [call[0] for call in setup.calls] == ["llm", "checkpoint"]
    assert app.research_coordinator.available_sources == {RetrievalSource.LOCAL}
    assert app.research_coordinator.workflow.vision_retriever is None
    app.close()
    assert [setup.store.close_calls, setup.hybrid.close_calls, setup.client.close_calls] == [
        0, 0, 0
    ]


def test_qdrant_open_failure_warns_and_keeps_application_available(
    vision_resources, caplog
) -> None:
    setup = vision_resources

    def fail_qdrant(value):
        assert value is setup.settings.qdrant
        setup.calls.append(("qdrant", value))
        raise OSError("store cannot open")

    app = application.build_application(
        setup.settings, factories=replace(setup.factories, qdrant=fail_qdrant)
    )

    assert app.research_coordinator.available_sources == {RetrievalSource.LOCAL}
    assert app.research_coordinator.workflow.vision_retriever is None
    assert [call[0] for call in setup.calls] == ["llm", "qdrant", "checkpoint"]
    assert "cannot open Qdrant: store cannot open" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)
    app.close()
    assert setup.events == ["checkpoint", "llm"]
    assert [setup.store.close_calls, setup.hybrid.close_calls, setup.client.close_calls] == [
        0, 0, 0
    ]


@pytest.mark.parametrize("web_enabled", [False, True])
def test_missing_collection_closes_probe_without_inspecting_chunks(
    vision_resources, web_enabled
) -> None:
    setup = vision_resources
    setup.store.collection_present = False
    if web_enabled:
        setup.settings = replace(
            setup.settings, web_search=WebSearchConfig(api_key="web-secret")
        )

    app = application.build_application(setup.settings, factories=setup.factories)

    assert setup.store.inspections == [("collection_exists",)]
    assert_probe_disabled_vision(setup, app)


@pytest.mark.parametrize("boundary", ["collection_exists", "iter_chunks", "iteration"])
def test_probe_inspection_failure_warns_and_closes_store(
    vision_resources, monkeypatch, caplog, boundary
) -> None:
    setup = vision_resources

    def fail_inspection(*args, **kwargs):
        if boundary == "iter_chunks":
            assert kwargs["source_types"] == {"image"}
        raise OSError("inspection failed")

    def fail_iteration(*, source_types):
        assert source_types == {"image"}

        def chunks():
            raise OSError("inspection failed")
            yield

        return chunks()

    if boundary == "iteration":
        monkeypatch.setattr(setup.store, "iter_chunks", fail_iteration)
    else:
        monkeypatch.setattr(setup.store, boundary, fail_inspection)

    app = application.build_application(setup.settings, factories=setup.factories)

    assert "cannot inspect indexed images: inspection failed" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)
    assert_probe_disabled_vision(setup, app)


@pytest.mark.parametrize(
    "source_kind",
    ["no_chunks", "missing", "directory", "empty", "unsupported", "unreadable", "mixed"],
)
def test_no_accessible_images_closes_probe_and_omits_vision(
    vision_resources, tmp_path, monkeypatch, source_kind
) -> None:
    setup = vision_resources
    missing = tmp_path / "missing.png"
    directory = tmp_path / "directory.png"
    directory.mkdir()
    empty = tmp_path / "empty.jpg"
    empty.touch()
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_bytes(b"data")
    unreadable = tmp_path / "unreadable.webp"
    unreadable.write_bytes(b"image")
    original_open = Path.open

    def open_image(path, *args, **kwargs):
        if path == unreadable:
            raise PermissionError("image cannot be read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_image)
    sources = {
        "no_chunks": [],
        "missing": [missing],
        "directory": [directory],
        "empty": [empty],
        "unsupported": [unsupported],
        "unreadable": [unreadable],
        "mixed": [missing, directory, empty, unsupported, unreadable],
    }[source_kind]
    setup.store.chunks = [image_chunk(source) for source in sources]

    app = application.build_application(setup.settings, factories=setup.factories)

    assert_probe_disabled_vision(setup, app)


def test_probe_skips_unusable_chunks_before_enabling_vision(
    vision_resources, tmp_path
) -> None:
    setup = vision_resources
    empty = tmp_path / "empty.jpg"
    empty.touch()
    sources = [tmp_path / "missing.png", empty, setup.image]
    setup.store.chunks = [image_chunk(source) for source in sources]

    app = application.build_application(setup.settings, factories=setup.factories)
    workflow = app.research_coordinator.workflow

    assert app.research_coordinator.available_sources == {
        RetrievalSource.LOCAL, RetrievalSource.VISION
    }
    assert workflow.vision_retriever.hybrid_retriever is setup.hybrid
    assert workflow.local_retriever._get_retriever() is setup.hybrid
    assert setup.store.inspections == [
        ("collection_exists",), ("iter_chunks", {"image"})
    ]
    assert setup.store.close_calls == 0
    app.close()
    assert setup.events == ["checkpoint", "hybrid", "vision", "qdrant", "llm"]
    assert [setup.hybrid.close_calls, setup.client.close_calls, setup.store.close_calls] == [
        1, 1, 1
    ]


def test_hybrid_construction_failure_closes_probed_store(vision_resources, caplog) -> None:
    setup = vision_resources

    def fail_hybrid(*values):
        setup.calls.append(("hybrid", *values))
        raise ValueError("hybrid configuration failed")

    with pytest.raises(ValueError, match="hybrid configuration failed"):
        application.build_application(
            setup.settings, factories=replace(setup.factories, hybrid=fail_hybrid)
        )

    assert [call[0] for call in setup.calls] == ["llm", "qdrant", "hybrid"]
    assert setup.events == ["qdrant", "llm"]
    assert setup.store.close_calls == setup.llm.close_calls == 1
    assert [
        setup.hybrid.close_calls, setup.client.close_calls, setup.checkpoint.close_calls
    ] == [0, 0, 0]
    assert not caplog.records


@pytest.mark.parametrize(
    "boundary", ["KnowledgeSearchTool", "VisionAnalyzer", "VisionRetriever"]
)
def test_vision_consumer_construction_failure_closes_all_acquired_resources(
    vision_resources, monkeypatch, caplog, boundary
) -> None:
    from insight_agent.application import bootstrap

    setup = vision_resources

    def fail_consumer(*args, **kwargs):
        raise RuntimeError(f"{boundary} failed")

    monkeypatch.setattr(bootstrap, boundary, fail_consumer)

    with pytest.raises(RuntimeError, match=f"{boundary} failed"):
        application.build_application(setup.settings, factories=setup.factories)

    assert setup.events == ["hybrid", "vision", "qdrant", "llm"]
    assert [
        setup.hybrid.close_calls, setup.client.close_calls, setup.store.close_calls,
        setup.llm.close_calls,
    ] == [1, 1, 1, 1]
    assert setup.checkpoint.close_calls == 0
    assert not caplog.records


def test_checkpoint_failure_closes_transferred_vision_runtime(vision_resources) -> None:
    setup = vision_resources

    def fail_checkpoint(value):
        assert value is setup.settings.checkpoint
        raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        application.build_application(
            setup.settings, factories=replace(setup.factories, checkpoint=fail_checkpoint)
        )

    assert setup.events == ["hybrid", "vision", "qdrant", "llm"]
    assert [
        setup.hybrid.close_calls, setup.client.close_calls, setup.store.close_calls,
        setup.llm.close_calls,
    ] == [1, 1, 1, 1]
    assert setup.checkpoint.close_calls == 0


@pytest.mark.parametrize(
    "boundary", ["ResearchRoutingWorkflow", "ResearchCoordinator", "InsightAgent"]
)
def test_later_composition_failure_closes_checkpoint_before_vision_runtime(
    vision_resources, monkeypatch, boundary
) -> None:
    from insight_agent.application import bootstrap

    setup = vision_resources

    class TrackedKnowledgeTool(KnowledgeSearchTool):
        def close(self):
            setup.events.append("knowledge")
            super().close()

    def fail_composition(*args, **kwargs):
        raise RuntimeError(f"{boundary} failed")

    monkeypatch.setattr(bootstrap, "KnowledgeSearchTool", TrackedKnowledgeTool)
    monkeypatch.setattr(bootstrap, boundary, fail_composition)

    with pytest.raises(RuntimeError, match=f"{boundary} failed"):
        application.build_application(setup.settings, factories=setup.factories)

    assert setup.events == [
        "checkpoint", "knowledge", "hybrid", "vision", "qdrant", "llm"
    ]
    assert [
        setup.checkpoint.close_calls, setup.hybrid.close_calls, setup.client.close_calls,
        setup.store.close_calls, setup.llm.close_calls,
    ] == [1, 1, 1, 1, 1]


def test_hybrid_cleanup_error_still_closes_every_resource_once(
    vision_resources, monkeypatch
) -> None:
    setup = vision_resources
    app = application.build_application(setup.settings, factories=setup.factories)
    close_hybrid = setup.hybrid.close

    def fail_hybrid_close():
        close_hybrid()
        raise RuntimeError("hybrid cleanup failed")

    monkeypatch.setattr(setup.hybrid, "close", fail_hybrid_close)

    with pytest.raises(RuntimeError, match="hybrid cleanup failed"):
        app.close()

    assert setup.events == ["checkpoint", "hybrid", "vision", "qdrant", "llm"]
    app.close()
    assert setup.events == ["checkpoint", "hybrid", "vision", "qdrant", "llm"]
    assert [
        setup.checkpoint.close_calls, setup.hybrid.close_calls, setup.client.close_calls,
        setup.store.close_calls, setup.llm.close_calls,
    ] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("extension", ["png", "jpg", "jpeg", "webp"])
def test_accessible_image_accepts_nonempty_supported_files(tmp_path, extension) -> None:
    from insight_agent.application.bootstrap import _is_accessible_image

    image = tmp_path / f"image.{extension}"
    image.write_bytes(b"image")

    assert _is_accessible_image(str(image))


@pytest.mark.parametrize("source_kind", ["missing", "directory", "empty", "unsupported"])
def test_accessible_image_rejects_unusable_sources(tmp_path, source_kind) -> None:
    from insight_agent.application.bootstrap import _is_accessible_image

    image = tmp_path / ("image.txt" if source_kind == "unsupported" else "image.png")
    if source_kind == "directory":
        image.mkdir()
    elif source_kind == "empty":
        image.touch()
    elif source_kind == "unsupported":
        image.write_bytes(b"data")

    assert not _is_accessible_image(str(image))


@pytest.mark.parametrize(
    "error", [PermissionError("unreadable"), IngestionError("invalid image")]
)
def test_accessible_image_handles_os_and_ingestion_errors(
    tmp_path, monkeypatch, error
) -> None:
    from insight_agent.application import bootstrap

    image = tmp_path / "image.png"
    image.write_bytes(b"image")

    def fail_open(*args, **kwargs):
        raise error

    if isinstance(error, IngestionError):
        monkeypatch.setattr(bootstrap, "guess_mime_type", fail_open)
    else:
        monkeypatch.setattr(Path, "open", fail_open)

    assert not bootstrap._is_accessible_image(str(image))


def test_app_close_hook_runs_once() -> None:
    from insight_agent.app import InsightAgent

    events: list[str] = []
    app = InsightAgent(router=None, llm=None, research_agent=None, research_coordinator=None)
    app.add_close_callback(lambda: events.append("owned"))

    app.close()
    app.close()

    assert events == ["owned"]
