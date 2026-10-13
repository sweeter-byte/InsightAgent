"""Real component composition for SciFact prepare, index, and benchmark runs."""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
from importlib import metadata
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.beir_scifact.artifacts import (
    append_prediction,
    build_prediction_record,
    finalize_run,
    initialize_run,
    mark_run_failed,
)
from evals.beir_scifact.config import BenchmarkConfig, BenchmarkPaths
from evals.beir_scifact.dataset import (
    SCIFACT_ARCHIVE_MD5,
    SCIFACT_ARCHIVE_URL,
    SCIFACT_DATASET_VERSION,
    dataset_to_documents,
    download_archive,
    load_scifact,
    safe_extract_zip,
    verify_archive,
    verify_extracted_files,
)
from evals.beir_scifact.index import (
    build_or_reuse_index,
    expected_chunk_id_digest,
    index_fingerprint,
    validate_completed_index,
)
from evals.beir_scifact.metrics import official_evaluator_factory
from evals.beir_scifact.retrieval import SciFactExperimentRunner
from insight_agent.indexing import (
    QdrantConfig,
    QdrantVectorStore,
    SentenceTransformerEmbedder,
    TextChunker,
)
from insight_agent.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    VectorRetriever,
)


class BatchedEmbedder:
    """Limit document embedding calls without changing the production adapter."""

    def __init__(self, embedder: Any, batch_size: int) -> None:
        self.embedder = embedder
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(
                self.embedder.embed_documents(texts[offset : offset + self.batch_size])
            )
        return vectors


def prepare_dataset(paths: BenchmarkPaths) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    if not paths.archive_path.exists():
        download_archive(paths.archive_path)
    checksum = verify_archive(paths.archive_path)
    if not paths.dataset_path.is_dir():
        safe_extract_zip(paths.archive_path, paths.dataset_parent)
    file_hashes = verify_extracted_files(paths.archive_path, paths.dataset_path)
    dataset = load_scifact(paths.dataset_path)
    manifest = {
        "schema_version": 1,
        "dataset": "BEIR SciFact",
        "dataset_version": SCIFACT_DATASET_VERSION,
        "split": "test",
        "source_url": SCIFACT_ARCHIVE_URL,
        "archive_md5": checksum,
        "corpus_count": len(dataset.corpus),
        "query_count": len(dataset.queries),
        "qrel_query_count": len(dataset.qrels),
        "files_sha256": file_hashes,
    }
    _write_json(paths.root / "dataset-manifest.json", manifest)
    return manifest


def build_index(paths: BenchmarkPaths, config: BenchmarkConfig) -> dict[str, Any]:
    _require_prepared(paths)
    dataset = load_scifact(paths.dataset_path)
    documents = dataset_to_documents(dataset)
    fingerprint = index_fingerprint(
        config,
        corpus_md5=SCIFACT_ARCHIVE_MD5,
        collection_name=paths.collection_name,
    )
    store = QdrantVectorStore(
        config=QdrantConfig(paths.qdrant_path, paths.collection_name)
    )
    try:
        base_embedder = SentenceTransformerEmbedder(config.embedding_model)
        result = build_or_reuse_index(
            documents,
            chunker=TextChunker(config.chunk_size, config.chunk_overlap),
            embedder=BatchedEmbedder(base_embedder, config.embedding_batch_size),
            vector_store=store,
            manifest_path=paths.index_manifest_path,
            fingerprint=fingerprint,
            document_batch_size=config.document_batch_size,
            progress=lambda completed, total: print(
                f"[scifact:index] documents {completed}/{total}", flush=True
            ),
        )
    finally:
        store.close()
    return asdict(result)


def run_benchmark(
    paths: BenchmarkPaths,
    config: BenchmarkConfig,
    *,
    smoke: bool,
    results_root: Path,
    run_id: str | None = None,
    query_limit: int = 5,
    query_ids: Iterable[str] | None = None,
) -> Path:
    _require_prepared(paths)
    dataset = load_scifact(paths.dataset_path)
    expected_fingerprint = index_fingerprint(
        config,
        corpus_md5=SCIFACT_ARCHIVE_MD5,
        collection_name=paths.collection_name,
    )
    index_manifest = _load_json(paths.index_manifest_path)
    if index_manifest.get("status") != "complete":
        raise ValueError("SciFact index is incomplete; run the index command first")
    if index_manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("SciFact index configuration does not match this benchmark run")
    current_chunk_digest = expected_chunk_id_digest(
        dataset_to_documents(dataset),
        TextChunker(config.chunk_size, config.chunk_overlap),
    )
    if index_manifest.get("chunk_id_digest") != current_chunk_digest:
        raise ValueError("SciFact extracted corpus does not match the completed index")

    selected = _select_queries(
        dataset.queries,
        smoke=smoke,
        query_limit=query_limit,
        query_ids=query_ids,
    )
    identifier = run_id or _new_run_id("smoke" if smoke else "full")
    run_directory = safe_run_directory(results_root, identifier)
    manifest = {
        "run_id": identifier,
        "evaluation_type": "real_beir_retrieval_benchmark",
        "smoke": smoke,
        "dataset": {
            "name": "BEIR SciFact",
            "version": SCIFACT_DATASET_VERSION,
            "archive_md5": SCIFACT_ARCHIVE_MD5,
            "split": "test",
            "corpus_count": len(dataset.corpus),
            "query_count": len(selected),
            "official_test_query_count": len(dataset.queries),
        },
        "config": asdict(config),
        "index_fingerprint": expected_fingerprint,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "environment": runtime_environment(),
        "reproduce": _reproduce_command(
            paths,
            config,
            smoke=smoke,
            results_root=results_root,
            query_limit=query_limit,
            query_ids=query_ids,
        ),
    }
    initialize_run(run_directory, manifest)
    log_path = run_directory / "benchmark.log"
    store: QdrantVectorStore | None = None
    try:
        store = QdrantVectorStore(
            config=QdrantConfig(paths.qdrant_path, paths.collection_name)
        )
        validate_completed_index(
            store,
            index_manifest,
            expected_fingerprint=expected_fingerprint,
        )
        chunks = store.load_chunks()
        if not chunks:
            raise ValueError("SciFact index contains no chunks")
        query_embedder = SentenceTransformerEmbedder(config.embedding_model)
        experiment_runner = SciFactExperimentRunner(
            VectorRetriever(query_embedder, store),
            BM25Retriever(lambda: chunks),
            CrossEncoderReranker(
                config.reranker_model,
                batch_size=config.reranker_batch_size,
            ),
            config=config,
        )
        records: list[dict[str, Any]] = []
        _log(log_path, f"run={identifier} queries={len(selected)} smoke={smoke}")
        for position, (query_id, query) in enumerate(selected.items(), start=1):
            _log(log_path, f"query {position}/{len(selected)} id={query_id}")
            results = experiment_runner.run_query(query_id, query)
            record = build_prediction_record(
                query_id,
                query,
                results,
                dataset.qrels[query_id],
            )
            append_prediction(run_directory, record)
            records.append(record)
        finalize_run(
            run_directory,
            records,
            evaluator_factory=official_evaluator_factory,
        )
        _log(log_path, "status=complete")
    except Exception as exc:
        mark_run_failed(run_directory, exc)
        _log(log_path, f"status=failed error={str(exc).strip() or type(exc).__name__}")
        raise
    finally:
        if store is not None:
            store.close()
    return run_directory


def _select_queries(
    queries: Mapping[str, str],
    *,
    smoke: bool,
    query_limit: int,
    query_ids: Iterable[str] | None,
) -> dict[str, str]:
    if not smoke:
        if len(queries) != 300:
            raise ValueError("full SciFact run requires all 300 official test queries")
        return dict(queries)
    requested = list(query_ids or [])
    if requested:
        unknown = [query_id for query_id in requested if query_id not in queries]
        if unknown:
            raise ValueError(f"unknown smoke query ID: {unknown[0]}")
        return {query_id: queries[query_id] for query_id in requested}
    if query_limit <= 0:
        raise ValueError("smoke query_limit must be positive")
    ordered_ids = sorted(queries, key=lambda value: (len(value), value))[:query_limit]
    return {query_id: queries[query_id] for query_id in ordered_ids}


def runtime_environment() -> dict[str, Any]:
    """Capture versions that materially affect local retrieval reproducibility."""
    package_names = (
        "beir",
        "torch",
        "sentence-transformers",
        "qdrant-client",
        "rank-bm25",
    )
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available: bool | None = None
    torch_cuda: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = torch.version.cuda
    except ImportError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "cuda_available": cuda_available,
        "torch_cuda": torch_cuda,
    }


def safe_run_directory(results_root: str | Path, run_id: str) -> Path:
    """Resolve one run below the configured root without traversal or reuse."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run ID must be a non-empty directory name")
    name = Path(run_id)
    if name.is_absolute() or len(name.parts) != 1 or name.name in {".", ".."}:
        raise ValueError("run ID must be a single safe directory name")
    root = Path(results_root)
    target = root / name
    if target.resolve().parent != root.resolve():
        raise ValueError("run ID escapes the configured results root")
    if target.exists():
        raise FileExistsError(f"run directory already exists: {target}")
    return target


def _require_prepared(paths: BenchmarkPaths) -> None:
    if not paths.dataset_path.is_dir() or not paths.archive_path.is_file():
        raise ValueError("SciFact data is not prepared; run the prepare command first")
    verify_archive(paths.archive_path)
    verify_extracted_files(paths.archive_path, paths.dataset_path)


def _new_run_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{kind}"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _reproduce_command(
    paths: BenchmarkPaths,
    config: BenchmarkConfig,
    *,
    smoke: bool,
    results_root: Path,
    query_limit: int,
    query_ids: Iterable[str] | None,
) -> str:
    arguments = [
        "scripts/benchmarks/scifact.sh",
        "smoke" if smoke else "full",
        "--root",
        str(paths.root),
        "--collection",
        paths.collection_name,
        "--results-root",
        str(results_root),
    ]
    for field_name in (
        "embedding_model",
        "reranker_model",
        "chunk_size",
        "chunk_overlap",
        "embedding_batch_size",
        "document_batch_size",
        "dense_k",
        "sparse_k",
        "rerank_k",
        "document_top_k",
        "rrf_k",
        "reranker_batch_size",
    ):
        arguments.extend(
            [f"--{field_name.replace('_', '-')}", str(getattr(config, field_name))]
        )
    if smoke:
        arguments.extend(["--query-limit", str(query_limit)])
        for query_id in query_ids or ():
            arguments.extend(["--query-id", query_id])
    return shlex.join(arguments)


def _log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(f"[scifact] {message}", flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
