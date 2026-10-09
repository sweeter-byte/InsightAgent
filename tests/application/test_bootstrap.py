"""Composition contracts exercised without models, network, or databases."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import insight_agent.application as application
from insight_agent.application import AppConfig, CheckpointConfig
from insight_agent.indexing import QdrantConfig
from insight_agent.ingestion import VisionModelConfig
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


def test_vision_shares_hybrid_and_closes_each_resource_once(tmp_path) -> None:
    factories, llm, checkpoint, hybrid, calls, events = resources()
    image = tmp_path / "indexed.png"
    image.write_bytes(b"image")

    class Store(Resource):
        def collection_exists(self):
            return True

        def iter_chunks(self, *, source_types):
            assert source_types == {"image"}
            return [type("Chunk", (), {"source": str(image)})()]

    class VisionClient(Resource):
        def analyze_image(self, *args, **kwargs):
            pytest.fail("Vision model must not run during composition")

    store = Store("qdrant", events)
    client = VisionClient("vision", events)
    vision = VisionModelConfig(
        api_key="vision-secret", base_url="https://vision.example/v1", model="vision"
    )
    settings = replace(config(), vision=vision)
    factories = replace(
        factories, qdrant=lambda value: store, vision_client=lambda value: client
    )

    app = application.build_application(settings, factories=factories)
    workflow = app.research_coordinator.workflow

    assert workflow.local_retriever._get_retriever() is hybrid
    assert workflow.vision_retriever.hybrid_retriever is hybrid
    assert app.research_coordinator.available_sources == {
        RetrievalSource.LOCAL, RetrievalSource.VISION
    }
    assert (
        "hybrid", settings.retrieval, settings.qdrant, settings.embedding_model, store
    ) in calls
    app.close()
    app.close()
    assert events == ["checkpoint", "hybrid", "vision", "qdrant", "llm"]
    assert [hybrid.close_calls, client.close_calls, store.close_calls] == [1, 1, 1]


def test_app_close_hook_runs_once() -> None:
    from insight_agent.app import InsightAgent

    events: list[str] = []
    app = InsightAgent(router=None, llm=None, research_agent=None, research_coordinator=None)
    app.add_close_callback(lambda: events.append("owned"))

    app.close()
    app.close()

    assert events == ["owned"]
