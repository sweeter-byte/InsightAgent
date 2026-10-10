from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from insight_agent.application import KnowledgeService, MaterialStorage, QueryService
from insight_agent.indexing import TextChunker
from insight_agent.ingestion import Document, SourceType
from insight_agent.router import Intent
from insight_agent.runtime.api import create_api
from insight_agent.runtime.events import InMemoryRuntimeEventStore
from insight_agent.runtime.policies import RuntimePolicy
from insight_agent.runtime.registry import InMemoryRunRegistry
from insight_agent.runtime.service import ProgressCallback, ResearchRuntimeService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OPERATIONS = {
    ("get", "/health"),
    ("get", "/ready"),
    ("post", "/v1/materials"),
    ("post", "/v1/query"),
    ("post", "/v1/research/runs"),
    ("get", "/v1/research/runs/{run_id}"),
    ("get", "/v1/research/runs/{run_id}/events"),
    ("post", "/v1/research/runs/{run_id}/resume"),
}


class MemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Any] = []

    def ensure_collection(self, vector_size: int) -> None:
        assert vector_size == 2

    def upsert(self, chunks: list[Any], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.chunks = list(chunks)


class StableEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class Refresher:
    def __init__(self) -> None:
        self.calls = 0

    def refresh(self) -> None:
        self.calls += 1


class DemoApplication:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.direct_calls: list[str] = []
        self.analyze_calls: list[str] = []

    def route(self, query: str) -> Intent:
        if query.startswith("direct:"):
            return Intent.DIRECT
        if query.startswith("analyze:"):
            return Intent.ANALYZE
        return Intent.RESEARCH

    def answer_direct(self, query: str) -> str:
        self.direct_calls.append(query)
        return "direct answer"

    def analyze(self, query: str) -> str:
        self.analyze_calls.append(query)
        return "\n".join(chunk.content for chunk in self.store.chunks)


class InstantResearchRunner:
    async def run(
        self,
        *,
        thread_id: str,
        query: str | None,
        resume: bool,
        on_progress: ProgressCallback,
    ) -> str:
        assert query == "research: compare approaches"
        assert resume is False
        await on_progress("reporting", {"thread_id": thread_id})
        return "# Research Report\n\nOffline acceptance report."


def _demo_api(tmp_path: Path) -> tuple[Any, DemoApplication, Refresher]:
    store = MemoryStore()
    refresher = Refresher()
    knowledge = KnowledgeService(
        ingestor=lambda path: [
            Document(
                Path(path).read_text(encoding="utf-8"),
                path,
                SourceType.MARKDOWN,
                {},
            )
        ],
        chunker=TextChunker(chunk_size=500, chunk_overlap=0),
        embedder=StableEmbedder(),
        vector_store=store,
        refresher=refresher,
    )
    runtime = ResearchRuntimeService(
        registry=InMemoryRunRegistry(),
        events=InMemoryRuntimeEventStore(),
        runner=InstantResearchRunner(),
        policy=RuntimePolicy(infra_retry_backoff_seconds=0),
        run_id_factory=lambda: "demo-run",
        thread_id_factory=lambda: "demo-thread",
    )
    application = DemoApplication(store)
    return (
        create_api(
            runtime,
            query_service=QueryService(application, runtime),
            material_storage=MaterialStorage(tmp_path / "uploads", max_bytes=4096),
            knowledge_service=knowledge,
        ),
        application,
        refresher,
    )


def test_final_http_contract_uses_compatible_research_prefix() -> None:
    schema = create_api().openapi()
    actual = {
        (method, path)
        for path, methods in schema["paths"].items()
        for method in methods
        if method in {"get", "post"}
    }

    assert actual == EXPECTED_OPERATIONS
    assert not any(path.startswith("/research/") for _, path in actual)


def test_offline_http_flow_covers_import_intents_runtime_and_sse(
    tmp_path: Path,
) -> None:
    app, application, refresher = _demo_api(tmp_path)
    material = b"The Chapter 18 acceptance marker is IA-CH18-READY."

    with TestClient(app) as client:
        imported = client.post(
            "/v1/materials",
            files={"file": ("demo.md", material, "text/markdown")},
        )
        direct = client.post("/v1/query", json={"query": "direct: hello"})
        analyze = client.post(
            "/v1/query", json={"query": "analyze: what is the marker?"}
        )
        research = client.post(
            "/v1/query", json={"query": "research: compare approaches"}
        )
        run_id = research.json()["run_id"]
        events = client.get(f"/v1/research/runs/{run_id}/events")
        status = client.get(f"/v1/research/runs/{run_id}")

    assert imported.status_code == 201
    assert imported.json()["status"] == "indexed"
    assert imported.json()["chunk_count"] == 1
    assert refresher.calls == 1
    assert direct.status_code == 200
    assert direct.json() == {
        "request_id": direct.headers["x-request-id"],
        "intent": "direct",
        "status": "completed",
        "answer": "direct answer",
        "run_id": None,
    }
    assert analyze.status_code == 200
    assert analyze.json()["intent"] == "analyze"
    assert "IA-CH18-READY" in analyze.json()["answer"]
    assert research.status_code == 202
    assert research.json()["intent"] == "research"
    assert application.direct_calls == ["direct: hello"]
    assert application.analyze_calls == ["analyze: what is the marker?"]
    assert status.json()["status"] == "completed"
    assert status.json()["final_output"].startswith("# Research Report")
    assert "event: run.queued" in events.text
    assert "event: run.started" in events.text
    assert "event: workflow.progress" in events.text
    assert "event: run.completed" in events.text


def test_evaluation_cli_is_independent_and_exposes_real_arguments() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "evals", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for option in (
        "--dataset",
        "--fixture",
        "--register-baseline",
        "--baseline",
        "--candidate",
    ):
        assert option in help_result.stdout

    isolation = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import evals.__main__; "
            "assert 'insight_agent.runtime.app' not in sys.modules",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert isolation.returncode == 0, isolation.stderr


def test_demo_assets_cover_all_five_scenarios_without_secrets() -> None:
    material = (PROJECT_ROOT / "examples" / "chapter18_demo_material.md").read_text(
        encoding="utf-8"
    )
    script = (PROJECT_ROOT / "examples" / "chapter18_demo.sh").read_text(
        encoding="utf-8"
    )

    assert "IA-CH18-READY" in material
    for value in (
        "/health",
        "/ready",
        "/v1/materials",
        "/v1/query",
        "/v1/research/runs",
        "/events",
        "/resume",
        "python -m evals",
        "--dataset",
        "--fixture",
        "--register-baseline",
        "--candidate",
    ):
        assert value in script
    assert "curl -N" in script
    assert "INTERRUPTED_RUN_ID" in script
    assert "API_KEY=" not in script


def test_delivery_docs_describe_the_current_runtime_contract() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (PROJECT_ROOT / "README.zh.md").read_text(encoding="utf-8")
    environment = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    combined = english + chinese

    for value in (
        "LangGraph",
        "conda run --no-capture-output -n insight-agent",
        "redis-server",
        "uvicorn insight_agent.runtime.app:app",
        "/v1/materials",
        "/v1/query",
        "/v1/research/runs",
        "python -m evals",
        "--dataset",
        "--fixture",
    ):
        assert value in english
        assert value in chinese
    assert "RUNTIME_REDIS_URL=" in environment
    assert "MATERIAL_MAX_BYTES=" in environment
    assert "LLM_API_KEY=\n" in environment
    assert "TAVILY_API_KEY=\n" in environment
