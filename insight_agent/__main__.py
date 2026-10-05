"""Command-line entry: ``python -m insight_agent``.

Behavior:
  * With no arguments — start an interactive REPL, one query per line.
  * With arguments — run a single query built from ``sys.argv[1:]`` and exit.

Nothing here is required by the agent itself; the CLI is just a thin wrapper
that assembles the app — a shared ``LLMClient``, a default registry,
``ResearchAgent``, both routers, the research workflow, and the ``InsightAgent``
façade — and reads a line of user input.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from insight_agent.agent import AgentStepsExceeded, ResearchAgent
from insight_agent.app import InsightAgent
from insight_agent.indexing import QdrantVectorStore
from insight_agent.ingestion import (
    IngestionError,
    OpenAICompatibleVisionClient,
    VisionModelConfig,
    guess_mime_type,
)
from insight_agent.llm import LLMClient
from insight_agent.planning import PlanningError, ResearchCoordinator, ResearchPlanner
from insight_agent.research import ResearchRoutingWorkflow
from insight_agent.retrieval import (
    HybridRetrievalConfig,
    HybridRetriever,
    KnowledgeSearchTool,
    build_default_hybrid_retriever,
    build_default_knowledge_search_tool,
)
from insight_agent.router import IntentRouter
from insight_agent.routing import RetrievalSource, RoutingError
from insight_agent.routing.router import RetrievalRouter
from insight_agent.tools.registry import build_default_registry, default_tool_schemas
from insight_agent.web_search import (
    TavilySearchProvider,
    WebRetriever,
    WebSearchConfig,
    WebSearchError,
)
from insight_agent.vision_retrieval import (
    VisionAnalyzer,
    VisionRetrievalError,
    VisionRetriever,
)


_BANNER = (
    "InsightAgent — minimal Research Agent. Type a request, or `exit`/Ctrl-D to quit."
)
logger = logging.getLogger(__name__)


def _build_app() -> InsightAgent:
    """Compose the full application object.

    A single ``LLMClient`` instance is shared by both routers, the planner, the
    direct-answer path, and the ResearchAgent. We deliberately avoid a factory
    or DI framework for this small object graph.
    """
    llm = LLMClient()
    retrieval_config = HybridRetrievalConfig.from_env()
    vision_retriever, shared_hybrid, vision_cleanup = _build_vision_runtime(
        retrieval_config
    )
    try:
        return _compose_app(
            llm=llm,
            retrieval_config=retrieval_config,
            vision_retriever=vision_retriever,
            shared_hybrid=shared_hybrid,
            vision_cleanup=vision_cleanup,
        )
    except BaseException:
        if vision_cleanup is not None:
            try:
                vision_cleanup()
            except Exception:
                logger.exception(
                    "Failed to release Vision Retrieval resources after "
                    "application composition failed."
                )
        raise


def _compose_app(
    *,
    llm: LLMClient,
    retrieval_config: HybridRetrievalConfig,
    vision_retriever: VisionRetriever | None,
    shared_hybrid: HybridRetriever | None,
    vision_cleanup: Callable[[], None] | None,
) -> InsightAgent:
    """Wire collaborators after optional Vision resources have been acquired."""
    knowledge_search_tool = (
        KnowledgeSearchTool(
            shared_hybrid,
            default_top_k=retrieval_config.final_top_k,
        )
        if shared_hybrid is not None
        else build_default_knowledge_search_tool(retrieval_config)
    )
    research_agent = ResearchAgent(
        llm=llm,
        registry=build_default_registry(
            knowledge_search_tool=knowledge_search_tool,
        ),
        tool_schemas=default_tool_schemas(),
    )
    planner = ResearchPlanner(llm=llm)
    retrieval_router = RetrievalRouter(llm=llm)
    web_config = WebSearchConfig.from_env()
    web_retriever: WebRetriever | None = None
    available_sources = {
        RetrievalSource.LOCAL,
    }
    if vision_retriever is not None:
        available_sources.add(RetrievalSource.VISION)
    if web_config is not None:
        web_retriever = WebRetriever(
            TavilySearchProvider(
                api_key=web_config.api_key,
                timeout=web_config.timeout,
            ),
            timeout=web_config.timeout,
            search_limit=web_config.search_limit,
            fetch_limit=web_config.fetch_limit,
        )
        available_sources.add(RetrievalSource.WEB)
    research_workflow = ResearchRoutingWorkflow(
        router=retrieval_router,
        local_retriever=knowledge_search_tool,
        web_retriever=web_retriever,
        vision_retriever=vision_retriever,
    )
    research_coordinator = ResearchCoordinator(
        planner=planner,
        workflow=research_workflow,
        research_agent=research_agent,
        available_sources=available_sources,
    )
    router = IntentRouter(llm=llm)
    return InsightAgent(
        router=router,
        llm=llm,
        research_agent=research_agent,
        research_coordinator=research_coordinator,
        close_callbacks=[
            knowledge_search_tool.close,
            *([vision_cleanup] if vision_cleanup is not None else []),
        ],
    )


def _build_vision_runtime(
    retrieval_config: HybridRetrievalConfig,
) -> tuple[
    VisionRetriever | None,
    HybridRetriever | None,
    Callable[[], None] | None,
]:
    """Probe Vision capability once and build its shared retrieval resources."""
    vision_config = VisionModelConfig.from_env(required=False)
    if vision_config is None:
        return None, None, None

    try:
        vector_store = QdrantVectorStore()
    except Exception as exc:
        logger.warning("Vision Retrieval disabled: cannot open Qdrant: %s", exc)
        return None, None, None
    try:
        if not vector_store.collection_exists():
            vector_store.close()
            return None, None, None
        image_chunks = vector_store.iter_chunks(source_types={"image"})
        if not any(_is_accessible_image(chunk.source) for chunk in image_chunks):
            vector_store.close()
            return None, None, None
    except Exception as exc:
        vector_store.close()
        logger.warning(
            "Vision Retrieval disabled: cannot inspect indexed images: %s",
            exc,
        )
        return None, None, None

    try:
        hybrid_retriever = build_default_hybrid_retriever(
            retrieval_config,
            vector_store,
        )
    except Exception:
        vector_store.close()
        raise

    try:
        vision_client = OpenAICompatibleVisionClient(vision_config)
    except Exception:
        vector_store.close()
        raise

    def cleanup() -> None:
        try:
            vision_client.close()
        finally:
            vector_store.close()

    return (
        VisionRetriever(
            hybrid_retriever=hybrid_retriever,
            analyzer=VisionAnalyzer(image_request=vision_client.analyze_image),
        ),
        hybrid_retriever,
        cleanup,
    )


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


def _print_answer(answer: str) -> None:
    print(f"\n{answer.strip()}\n" if answer and answer.strip() else "\n[empty response]\n")


def _run_once(app: InsightAgent, query: str) -> Optional[int]:
    """Execute a single query. Returns an exit code, or ``None`` to keep going."""
    try:
        answer = app.run(query)
    except AgentStepsExceeded as exc:
        print(f"[agent stopped] {exc}", file=sys.stderr)
        return 1
    except PlanningError as exc:
        print(f"[planning error] {exc}", file=sys.stderr)
        return 1
    except RoutingError as exc:
        print(f"[routing error] {exc}", file=sys.stderr)
        return 1
    except WebSearchError as exc:
        print(f"[web search error] {exc}", file=sys.stderr)
        return 1
    except VisionRetrievalError as exc:
        print(f"[vision retrieval error] {exc}", file=sys.stderr)
        return 1
    _print_answer(answer)
    return None


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    try:
        app = _build_app()
    except RuntimeError as exc:  # missing env vars
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    try:
        if args:
            rc = _run_once(app, " ".join(args))
            return rc if rc is not None else 0

        print(_BANNER)
        while True:
            try:
                line = input(">> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\n[interrupted]")
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit", ":q"}:
                break
            rc = _run_once(app, line)
            if rc is not None:
                return rc
        return 0
    finally:
        close = getattr(app, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
