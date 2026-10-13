"""Durable run artifacts for real SciFact benchmark executions."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.beir_scifact.metrics import compute_official_metrics
from evals.beir_scifact.ranking import rankings_to_beir_results
from evals.beir_scifact.retrieval import ExperimentQueryResult


EXPERIMENTS = ("A", "B", "C", "D")


def initialize_run(run_directory: str | Path, manifest: Mapping[str, Any]) -> Path:
    directory = Path(run_directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"run directory already exists and is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload.update(
        {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(directory / "manifest.json", payload)
    (directory / "predictions.jsonl").touch()
    return directory


def append_prediction(run_directory: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(run_directory) / "predictions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def build_prediction_record(
    query_id: str,
    query: str,
    results: Mapping[str, ExperimentQueryResult],
    qrels: Mapping[str, int],
) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    for name, result in results.items():
        rank_scores = rankings_to_beir_results({query_id: result.documents})[query_id]
        ranking = []
        for entry in result.documents:
            relevance = int(qrels.get(entry.document_id, 0))
            ranking.append(
                {
                    "document_id": entry.document_id,
                    "chunk_id": entry.chunk_id,
                    "rank": entry.rank,
                    "score": rank_scores[entry.document_id],
                    "relevance": relevance,
                    "hit": relevance > 0,
                }
            )
        experiments[name] = {
            "latency_ms": result.latency_ms,
            "ranking": ranking,
            "candidate_chunk_ids": list(result.candidate_chunk_ids),
            "candidate_pool_digest": result.candidate_pool_digest,
        }
    return {
        "query_id": query_id,
        "query": query,
        "qrels": {doc_id: int(value) for doc_id, value in qrels.items()},
        "experiments": experiments,
    }


def finalize_run(
    run_directory: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    evaluator_factory: Callable[..., Any],
) -> dict[str, Any]:
    directory = Path(run_directory)
    for record in records:
        names = set(record.get("experiments", {}))
        if names != set(EXPERIMENTS):
            raise ValueError("every prediction must contain all four experiments")

    metrics: dict[str, Any] = {}
    qrels = {
        str(record["query_id"]): dict(record["qrels"])
        for record in records
    }
    for name in EXPERIMENTS:
        results = {
            str(record["query_id"]): {
                str(item["document_id"]): float(item["score"])
                for item in record["experiments"][name]["ranking"]
            }
            for record in records
        }
        summary = compute_official_metrics(
            qrels,
            results,
            evaluator_factory=evaluator_factory,
        )
        latencies = [
            float(record["experiments"][name]["latency_ms"])
            for record in records
        ]
        summary["latency_ms"] = latency_summary(latencies)
        metrics[name] = summary

    bad_cases = extract_bad_cases(records)
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "bad_cases.json", bad_cases)
    from evals.beir_scifact.report import render_summary

    manifest = _read_json(directory / "manifest.json")
    (directory / "summary.md").write_text(
        render_summary(manifest, metrics, bad_cases),
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "query_count": len(records),
        }
    )
    _write_json(directory / "manifest.json", manifest)
    return metrics


def mark_run_failed(run_directory: str | Path, error: BaseException) -> None:
    path = Path(run_directory) / "manifest.json"
    manifest = _read_json(path)
    manifest.update(
        {
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(error).strip() or type(error).__name__,
        }
    )
    _write_json(path, manifest)


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies must be finite and nonnegative")
    ordered = sorted(float(value) for value in values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def extract_bad_cases(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    misses: list[dict[str, str]] = []
    low_rank: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for record in records:
        query_id = str(record["query_id"])
        experiments = record["experiments"]
        first_ranks: dict[str, int | None] = {}
        for name in EXPERIMENTS:
            rank = next(
                (
                    int(item["rank"])
                    for item in experiments[name]["ranking"]
                    if item["hit"]
                ),
                None,
            )
            first_ranks[name] = rank
            if rank is None:
                misses.append({"query_id": query_id, "experiment": name})
            elif rank > 5:
                low_rank.append(
                    {
                        "query_id": query_id,
                        "experiment": name,
                        "first_relevant_rank": rank,
                    }
                )
        c_rank = first_ranks["C"] or 11
        d_rank = first_ranks["D"] or 11
        comparison = {
            "query_id": query_id,
            "c_first_relevant_rank": first_ranks["C"],
            "d_first_relevant_rank": first_ranks["D"],
        }
        if d_rank < c_rank:
            improvements.append(comparison)
        elif d_rank > c_rank:
            regressions.append(comparison)
    return {
        "misses": misses,
        "low_rank": low_rank,
        "improvements": improvements,
        "regressions": regressions,
    }


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prediction line {line_number} is not an object")
            records.append(value)
    return records


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
