"""Production composition for benchmark generation without HTTP or Redis."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit

from evals.deepresearch_bench.artifacts import benchmark_checkpoint_path
from insight_agent.application import AppConfig, CheckpointConfig, build_application
from insight_agent.planning import ResearchCoordinator
from insight_agent.planning.models import MAX_RESEARCH_TASKS
from insight_agent.research.workflow import MAX_REPAIR_ROUNDS, MAX_RETRIEVAL_ROUNDS
from insight_agent.routing import RetrievalSource


@dataclass(frozen=True, slots=True)
class ProductionSession:
    coordinator: ResearchCoordinator
    usage_meter: Any
    metadata: dict[str, Any]


@contextmanager
def build_production_session(
    run_directory: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    config: AppConfig | None = None,
) -> Iterator[ProductionSession]:
    """Build the real LangGraph stack with a run-local checkpoint database."""
    config = config if config is not None else resolve_generation_config(environ)
    run_path = Path(run_directory)
    isolated = replace(
        config,
        vision=None,
        checkpoint=CheckpointConfig(benchmark_checkpoint_path(run_path)),
    )
    app = build_application(isolated, eager_knowledge=False)
    production = app.research_coordinator
    coordinator = ResearchCoordinator(
        planner=production.planner,
        workflow=production.workflow,
        available_sources={RetrievalSource.WEB},
    )
    metadata = generation_metadata(isolated, run_directory=run_path)
    try:
        yield ProductionSession(coordinator, app.llm, metadata)
    finally:
        app.close()


def resolve_generation_config(
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    config = AppConfig.from_env(environ)
    if config.web_search is None:
        raise RuntimeError(
            "Missing required environment variable 'TAVILY_API_KEY'. "
            "DeepResearch Bench generation does not fall back to mock or offline search."
        )
    return config


def generation_metadata(
    config: AppConfig,
    *,
    run_directory: str | Path,
) -> dict[str, Any]:
    assert config.web_search is not None
    return {
        "research_model": config.llm.model,
        "api_backend": _public_endpoint(config.llm.base_url),
        "api_version": None,
        "actual_model_identifier": None,
        "embedding_model": config.embedding_model,
        "reranker_model": config.retrieval.reranker_model,
        "web_search": {
            "provider": "tavily",
            "search_limit": config.web_search.search_limit,
            "fetch_limit": config.web_search.fetch_limit,
            "timeout": config.web_search.timeout,
        },
        "coverage": {
            "production_composition_root": True,
            "langgraph_research_workflow": True,
            "planner": True,
            "retrieval_router": True,
            "web_retrieval": True,
            "evidence_grading": True,
            "report_generation": True,
            "self_check": True,
            "intent_router": False,
            "http_runtime": False,
            "redis_runtime": False,
            "local_knowledge_base": False,
            "vision_retrieval": False,
        },
        "available_sources": [RetrievalSource.WEB.value],
        "research_parameters": {
            "max_plan_tasks": MAX_RESEARCH_TASKS,
            "max_retrieval_rounds": MAX_RETRIEVAL_ROUNDS,
            "max_repair_rounds": MAX_REPAIR_ROUNDS,
        },
        "qdrant": {"used": False},
        "checkpoint_path": str(benchmark_checkpoint_path(run_directory)),
        "checkpoint_scope": "per-run",
    }


def _public_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
