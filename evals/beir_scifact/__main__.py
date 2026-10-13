"""Command-line entry point for the real BEIR SciFact benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evals.beir_scifact.config import BenchmarkConfig, BenchmarkPaths
from evals.beir_scifact.report import regenerate_report
from evals.beir_scifact.runtime import build_index, prepare_dataset, run_benchmark


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("SCIFACT_ROOT", ".data/benchmarks/beir_scifact")),
    )


def _add_benchmark_config(parser: argparse.ArgumentParser) -> None:
    _add_root(parser)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("SCIFACT_EMBEDDING_MODEL", "BAAI/bge-m3"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("SCIFACT_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
    for option, env_name, default in (
        ("chunk-size", "SCIFACT_CHUNK_SIZE", 1200),
        ("chunk-overlap", "SCIFACT_CHUNK_OVERLAP", 200),
        ("embedding-batch-size", "SCIFACT_EMBEDDING_BATCH_SIZE", 32),
        ("document-batch-size", "SCIFACT_DOCUMENT_BATCH_SIZE", 64),
        ("dense-k", "SCIFACT_DENSE_K", 50),
        ("sparse-k", "SCIFACT_SPARSE_K", 50),
        ("rerank-k", "SCIFACT_RERANK_K", 100),
        ("document-top-k", "SCIFACT_DOCUMENT_TOP_K", 10),
        ("rrf-k", "SCIFACT_RRF_K", 60),
        ("reranker-batch-size", "SCIFACT_RERANKER_BATCH_SIZE", 16),
    ):
        parser.add_argument(
            f"--{option}",
            type=int,
            default=_env_int(env_name, default),
        )
    parser.add_argument(
        "--collection",
        default=os.getenv("SCIFACT_COLLECTION", "beir_scifact_chunks"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEIR SciFact retrieval benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download and verify SciFact")
    _add_root(prepare)
    index = commands.add_parser("index", help="build or reuse the isolated index")
    _add_benchmark_config(index)
    for name in ("smoke", "full"):
        command = commands.add_parser(name, help=f"run the {name} benchmark")
        _add_benchmark_config(command)
        command.add_argument(
            "--results-root",
            type=Path,
            default=Path(
                os.getenv(
                    "SCIFACT_RESULTS_ROOT",
                    "eval_results/benchmarks/beir_scifact",
                )
            ),
        )
        command.add_argument("--run-id", default=os.getenv("SCIFACT_RUN_ID") or None)
        if name == "smoke":
            command.add_argument(
                "--query-limit",
                type=int,
                default=_env_int("SCIFACT_SMOKE_QUERY_LIMIT", 5),
            )
            command.add_argument("--query-id", action="append", dest="query_ids")
    report = commands.add_parser("report", help="regenerate a completed report")
    report.add_argument("run_directory", type=Path)
    return parser


def _configuration(args: argparse.Namespace) -> tuple[BenchmarkConfig, BenchmarkPaths]:
    config = BenchmarkConfig(
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embedding_batch_size,
        document_batch_size=args.document_batch_size,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        rerank_k=args.rerank_k,
        document_top_k=args.document_top_k,
        rrf_k=args.rrf_k,
        reranker_batch_size=args.reranker_batch_size,
    )
    return config, BenchmarkPaths(args.root, args.collection)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "prepare":
            result = prepare_dataset(BenchmarkPaths(root=args.root))
        elif args.command == "report":
            result = {"summary": str(regenerate_report(args.run_directory))}
        else:
            config, paths = _configuration(args)
            if args.command == "index":
                result = build_index(paths, config)
            else:
                run_directory = run_benchmark(
                    paths,
                    config,
                    smoke=args.command == "smoke",
                    results_root=args.results_root,
                    run_id=args.run_id,
                    query_limit=getattr(args, "query_limit", 300),
                    query_ids=getattr(args, "query_ids", None),
                )
                result = {"run_directory": str(run_directory)}
    except Exception as exc:  # noqa: BLE001 - CLI boundary returns nonzero
        detail = str(exc).strip() or type(exc).__name__
        print(f"[scifact error] {detail}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
