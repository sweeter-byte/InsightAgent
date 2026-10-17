from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.deepresearch_bench.runtime import (
    build_production_session,
    generation_metadata,
    resolve_generation_config,
)
from insight_agent.routing import RetrievalSource


def _environment() -> dict[str, str]:
    return {
        "LLM_API_KEY": "research-secret",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_MODEL": "research-model",
        "TAVILY_API_KEY": "tavily-secret",
        "RERANKER_MODEL": "reranker",
    }


def test_production_session_uses_isolated_checkpoint_and_web_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    production = SimpleNamespace(
        planner=object(), workflow=object(), available_sources={RetrievalSource.LOCAL}
    )
    app = SimpleNamespace(
        research_coordinator=production,
        llm=object(),
        close=lambda: closed.append(True),
    )
    captured = {}

    def build(config, **kwargs):
        captured["config"] = config
        return app

    monkeypatch.setattr("evals.deepresearch_bench.runtime.build_application", build)

    with build_production_session(tmp_path / "run", environ=_environment()) as session:
        assert session.coordinator.planner is production.planner
        assert session.coordinator.workflow is production.workflow
        assert session.coordinator.available_sources == {RetrievalSource.WEB}
        assert session.usage_meter is app.llm
        assert session.metadata["checkpoint_path"] == str(
            (tmp_path / "run" / "checkpoints" / "checkpoints.sqlite").resolve()
        )
        assert session.metadata["checkpoint_scope"] == "per-run"

    assert captured["config"].checkpoint.path == (
        tmp_path / "run" / "checkpoints" / "checkpoints.sqlite"
    )
    assert closed == [True]


def test_production_session_requires_tavily_without_mock_fallback(tmp_path: Path) -> None:
    environment = _environment()
    environment.pop("TAVILY_API_KEY")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        with build_production_session(tmp_path / "run", environ=environment):
            pass


def test_generation_metadata_records_actual_run_checkpoint_path(tmp_path: Path) -> None:
    run_directory = tmp_path / "results" / "smoke-123"

    metadata = generation_metadata(
        resolve_generation_config(_environment()),
        run_directory=run_directory,
    )

    assert metadata["checkpoint_path"] == str(
        (run_directory / "checkpoints" / "checkpoints.sqlite").resolve()
    )
    assert metadata["checkpoint_scope"] == "per-run"
