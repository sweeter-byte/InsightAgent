"""Compose one application from explicit configuration and owned resources."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeVar

from insight_agent.agent import ResearchAgent
from insight_agent.app import InsightAgent
from insight_agent.application.config import AppConfig, CheckpointConfig
from insight_agent.application.knowledge import KnowledgeService
from insight_agent.evidence import EvidenceGrader
from insight_agent.indexing import (
    Embedder,
    QdrantConfig,
    QdrantVectorStore,
    SentenceTransformerEmbedder,
    TextChunker,
)
from insight_agent.ingestion import (
    IngestionError,
    OpenAICompatibleVisionClient,
    VisionModelConfig,
    guess_mime_type,
    ingest_file,
)
from insight_agent.llm import LLMClient, LLMConfig
from insight_agent.planning import ResearchCoordinator, ResearchPlanner
from insight_agent.research import ResearchRoutingWorkflow, SQLiteCheckpointStore
from insight_agent.reporting import ReportGenerator
from insight_agent.retrieval import (
    HybridRetrievalConfig,
    HybridRetriever,
    KnowledgeSearchTool,
    build_default_hybrid_retriever,
)
from insight_agent.router import IntentRouter
from insight_agent.routing import RetrievalSource
from insight_agent.routing.router import RetrievalRouter
from insight_agent.self_check import ReportRepairer, ReportSelfChecker
from insight_agent.tools.registry import build_default_registry, default_tool_schemas
from insight_agent.vision_retrieval import VisionAnalyzer, VisionRetriever
from insight_agent.web_search import TavilySearchProvider, WebRetriever


logger = logging.getLogger(__name__)
Resource = TypeVar("Resource")


def _default_llm(config: LLMConfig) -> LLMClient:
    return LLMClient(config)


def _default_checkpoint(config: CheckpointConfig) -> SQLiteCheckpointStore:
    return SQLiteCheckpointStore(config.path)


def _default_qdrant(config: QdrantConfig) -> QdrantVectorStore:
    return QdrantVectorStore(config=config)


def _default_vision_client(config: VisionModelConfig) -> OpenAICompatibleVisionClient:
    return OpenAICompatibleVisionClient(config)


def _default_embedder(model_name: str) -> Embedder:
    return SentenceTransformerEmbedder(model_name)


def _default_chunker() -> TextChunker:
    return TextChunker()


def _default_hybrid(
    config: HybridRetrievalConfig,
    qdrant_config: QdrantConfig,
    embedding_model: str,
    vector_store: QdrantVectorStore | None,
    embedder: Embedder | None = None,
) -> HybridRetriever:
    return build_default_hybrid_retriever(
        config,
        vector_store,
        qdrant_config=qdrant_config,
        embedding_model=embedding_model,
        embedder=embedder,
    )


@dataclass(frozen=True, slots=True)
class ApplicationFactories:
    """Resource constructors; every returned resource is application-owned."""

    llm: Callable[[LLMConfig], LLMClient] = _default_llm
    checkpoint: Callable[[CheckpointConfig], SQLiteCheckpointStore] = _default_checkpoint
    qdrant: Callable[[QdrantConfig], QdrantVectorStore] = _default_qdrant
    vision_client: Callable[
        [VisionModelConfig], OpenAICompatibleVisionClient
    ] = _default_vision_client
    embedder: Callable[[str], Embedder] = _default_embedder
    chunker: Callable[[], TextChunker] = _default_chunker
    hybrid: Callable[..., HybridRetriever] = _default_hybrid


def _own(stack: ExitStack, resource: Resource) -> Resource:
    """Register acquired closeables before constructing their consumers."""
    close = getattr(resource, "close", None)
    if callable(close):
        stack.callback(close)
    return resource


@dataclass(frozen=True, slots=True)
class _VisionRuntime:
    retriever: VisionRetriever
    knowledge_search_tool: KnowledgeSearchTool
    _close_callback: Callable[[], None]

    def close(self) -> None:
        self._close_callback()


def _build_vision_runtime(
    config: AppConfig,
    factories: ApplicationFactories,
) -> _VisionRuntime | None:
    if config.vision is None:
        return None

    with ExitStack() as stack:
        try:
            vector_store = _own(stack, factories.qdrant(config.qdrant))
        except Exception as exc:
            logger.warning("Vision Retrieval disabled: cannot open Qdrant: %s", exc)
            return None
        try:
            if not vector_store.collection_exists():
                return None
            image_chunks = vector_store.iter_chunks(source_types={"image"})
            if not any(_is_accessible_image(chunk.source) for chunk in image_chunks):
                return None
        except Exception as exc:
            logger.warning(
                "Vision Retrieval disabled: cannot inspect indexed images: %s", exc
            )
            return None

        with ExitStack() as pending_hybrid:
            shared_hybrid = _own(
                pending_hybrid,
                factories.hybrid(
                    config.retrieval,
                    config.qdrant,
                    config.embedding_model,
                    vector_store,
                ),
            )
            vision_client = _own(stack, factories.vision_client(config.vision))
            knowledge = _own(
                stack,
                KnowledgeSearchTool(
                    shared_hybrid, default_top_k=config.retrieval.final_top_k
                ),
            )
            # The tool now owns the shared hybrid. Drop the temporary guard so
            # the hybrid closes once, before the Vision client and shared store.
            pending_hybrid.pop_all()

        retriever = VisionRetriever(
            hybrid_retriever=shared_hybrid,
            analyzer=VisionAnalyzer(image_request=vision_client.analyze_image),
        )
        return _VisionRuntime(retriever, knowledge, stack.pop_all().close)


def _is_accessible_image(source: str) -> bool:
    path = Path(source)
    if not path.is_file():
        return False
    try:
        guess_mime_type(source)
        with path.open("rb") as image_file:
            return bool(image_file.read(1))
    except (IngestionError, OSError):
        return False


def build_application(
    config: AppConfig,
    *,
    factories: ApplicationFactories | None = None,
    eager_knowledge: bool = False,
) -> InsightAgent:
    """Build the existing business graph and transfer its cleanup to the app."""
    factories = factories if factories is not None else ApplicationFactories()
    with ExitStack() as stack:
        llm = _own(stack, factories.llm(config.llm))
        knowledge_service: KnowledgeService | None = None
        vector_store: QdrantVectorStore | None = None
        vision_runtime: _VisionRuntime | None = None
        vision_retriever: VisionRetriever | None = None

        if eager_knowledge:
            vector_store = _own(stack, factories.qdrant(config.qdrant))
            embedder = factories.embedder(config.embedding_model)
            shared_hybrid = _own(
                stack,
                factories.hybrid(
                    config.retrieval,
                    config.qdrant,
                    config.embedding_model,
                    vector_store,
                    embedder,
                ),
            )
            knowledge = KnowledgeSearchTool(
                shared_hybrid,
                default_top_k=config.retrieval.final_top_k,
            )
            vision_client = (
                _own(stack, factories.vision_client(config.vision))
                if config.vision is not None
                else None
            )
            if vision_client is not None and _has_accessible_indexed_image(vector_store):
                vision_retriever = VisionRetriever(
                    hybrid_retriever=shared_hybrid,
                    analyzer=VisionAnalyzer(image_request=vision_client.analyze_image),
                )
            chunker = factories.chunker()
            knowledge_service = KnowledgeService(
                ingestor=partial(ingest_file, vision_client=vision_client),
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                refresher=shared_hybrid,
            )
            checkpoint = _own(stack, factories.checkpoint(config.checkpoint))
        else:
            vision_runtime = _own(stack, _build_vision_runtime(config, factories))
            checkpoint = _own(stack, factories.checkpoint(config.checkpoint))
            if vision_runtime is not None:
                knowledge = vision_runtime.knowledge_search_tool
            else:

                def create_retriever() -> HybridRetriever:
                    return factories.hybrid(
                        config.retrieval, config.qdrant, config.embedding_model, None
                    )

                knowledge = _own(
                    stack,
                    KnowledgeSearchTool(
                        retriever_factory=create_retriever,
                        default_top_k=config.retrieval.final_top_k,
                    ),
                )

        research_agent = ResearchAgent(
            llm=llm,
            registry=build_default_registry(knowledge_search_tool=knowledge),
            tool_schemas=default_tool_schemas(),
        )
        planner = ResearchPlanner(llm=llm)
        retrieval_router = RetrievalRouter(llm=llm)
        evidence_grader = EvidenceGrader(llm=llm)
        report_generator = ReportGenerator(llm=llm)
        self_checker = ReportSelfChecker(llm=llm)
        report_repairer = ReportRepairer(llm=llm)
        web_retriever = None
        available_sources = {RetrievalSource.LOCAL}
        if vision_runtime is not None:
            vision_retriever = vision_runtime.retriever
        if vision_retriever is not None:
            available_sources.add(RetrievalSource.VISION)
        if config.web_search is not None:
            web = config.web_search
            web_retriever = WebRetriever(
                TavilySearchProvider(api_key=web.api_key, timeout=web.timeout),
                timeout=web.timeout,
                search_limit=web.search_limit,
                fetch_limit=web.fetch_limit,
            )
            available_sources.add(RetrievalSource.WEB)

        workflow = ResearchRoutingWorkflow(
            router=retrieval_router,
            local_retriever=knowledge,
            web_retriever=web_retriever,
            vision_retriever=vision_retriever,
            grader=evidence_grader,
            report_generator=report_generator,
            self_checker=self_checker,
            report_repairer=report_repairer,
            checkpointer=checkpoint.checkpointer,
        )
        coordinator = ResearchCoordinator(
            planner=planner, workflow=workflow, available_sources=available_sources
        )
        app = InsightAgent(
            router=IntentRouter(llm=llm),
            llm=llm,
            research_agent=research_agent,
            research_coordinator=coordinator,
            knowledge_service=knowledge_service,
            vector_store=vector_store,
            checkpoint_store=checkpoint,
        )
        app.add_close_callback(stack.pop_all().close)
        return app


def _has_accessible_indexed_image(vector_store: QdrantVectorStore) -> bool:
    """Probe existing image payloads without making Qdrant optional."""
    try:
        if not vector_store.collection_exists():
            return False
        return any(
            _is_accessible_image(chunk.source)
            for chunk in vector_store.iter_chunks(source_types={"image"})
        )
    except Exception as exc:
        logger.warning("Vision Retrieval disabled: cannot inspect indexed images: %s", exc)
        return False
