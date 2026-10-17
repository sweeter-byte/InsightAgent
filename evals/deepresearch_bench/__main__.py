"""Command-line entry point for generation, export, scoring, and reporting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from evals.deepresearch_bench.artifacts import RunStore
from evals.deepresearch_bench.dataset import (
    load_tasks,
    select_tasks,
)
from evals.deepresearch_bench.official import OfficialEvaluator
from evals.deepresearch_bench.prepare import (
    prepare_official_checkout,
    validate_official_checkout,
)
from evals.deepresearch_bench.reporting import write_reports
from evals.deepresearch_bench.runner import BenchmarkRunner
from evals.deepresearch_bench.runtime import (
    build_production_session,
    generation_metadata,
    resolve_generation_config,
)


DEFAULT_DATA_ROOT = Path(".data/benchmarks/deepresearch_bench")
DEFAULT_RESULTS_ROOT = Path("eval_results/benchmarks/deepresearch_bench")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.deepresearch_bench",
        description="InsightAgent DeepResearch Bench adapter",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download and validate the pinned official benchmark")
    prepare.add_argument("--official-root", type=Path, default=DEFAULT_DATA_ROOT / "official")

    doctor = subparsers.add_parser("doctor", help="check local generation and scoring prerequisites without API calls")
    _add_paths(doctor)

    for mode in ("smoke", "pilot", "full"):
        command = subparsers.add_parser(mode, help=f"generate reports for the fixed {mode} task set")
        _add_paths(command)
        command.add_argument("--run-id")
        command.add_argument("--task-ids", help="comma-separated official task IDs")
        command.add_argument("--language", choices=("zh", "en"))
        command.add_argument("--max-tasks", type=int)
        command.add_argument("--timeout-seconds", type=float, default=1800.0)
        command.add_argument("--max-retries", type=int, default=2)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--retry-failed", action="store_true")

    export = subparsers.add_parser("export", help="export saved successful reports to official JSONL")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path)

    score = subparsers.add_parser("score", help="run the pinned official RACE and/or FACT evaluator")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--official-root", type=Path, default=DEFAULT_DATA_ROOT / "official")
    score.add_argument("--race", action="store_true")
    score.add_argument("--fact", action="store_true")
    score.add_argument("--workers", type=int, default=1)
    score.add_argument("--force", action="store_true", help="rerun requested scoring stages from saved reports")
    score.add_argument("--confirm-paid", action="store_true")

    report = subparsers.add_parser("report", help="regenerate metrics and summary without API calls")
    report.add_argument("--run-dir", type=Path, required=True)
    return parser


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--official-root", type=Path, default=DEFAULT_DATA_ROOT / "official")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        metadata = prepare_official_checkout(args.official_root)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    if args.command == "doctor":
        return _doctor(args)
    if args.command in {"smoke", "pilot", "full"}:
        return _generate(args)
    if args.command == "export":
        store = RunStore(args.run_dir.resolve())
        output = store.export_jsonl(args.output)
        print(output)
        return 0
    if args.command == "score":
        return _score(args)
    if args.command == "report":
        metrics = write_reports(args.run_dir)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _doctor(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {}
    try:
        benchmark = validate_official_checkout(args.official_root)
        checks["official_benchmark"] = {"status": "ready", **benchmark}
    except Exception as exc:
        checks["official_benchmark"] = {"status": "error", "detail": str(exc)}
    try:
        config = resolve_generation_config()
        checks["generation"] = {
            "status": "ready",
            **generation_metadata(
                config,
                run_directory=args.results_root.resolve() / "<run-id>",
            ),
        }
    except Exception as exc:
        checks["generation"] = {"status": "error", "detail": str(exc)}
    backend = os.environ.get("LLM_BACKEND", "openrouter").lower()
    judge_key = "OPENAI_API_KEY" if backend == "openai" else "OPENROUTER_API_KEY"
    checks["official_judge"] = {
        "status": "ready" if os.environ.get(judge_key) else "unconfigured",
        "backend": backend,
        "required_key": judge_key,
        "jina": "configured" if os.environ.get("JINA_API_KEY") else "unconfigured",
    }
    missing_modules = [
        name
        for name in ("numpy", "pandas", "requests", "tqdm")
        if importlib.util.find_spec(name) is None
    ]
    checks["official_evaluation_dependencies"] = {
        "status": "ready" if not missing_modules else "unavailable",
        "missing": missing_modules,
    }
    try:
        args.results_root.mkdir(parents=True, exist_ok=True)
        checks["results_path"] = {"status": "ready", "path": str(args.results_root.resolve())}
    except OSError as exc:
        checks["results_path"] = {"status": "error", "detail": str(exc)}
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    required = ("official_benchmark", "generation", "results_path")
    return 0 if all(checks[name]["status"] == "ready" for name in required) else 1


def _generate(args: argparse.Namespace) -> int:
    if args.concurrency != 1:
        raise ValueError("this adapter currently supports --concurrency 1 only")
    if args.max_retries < 0:
        raise ValueError("--max-retries must not be negative")
    if args.max_retries > 5:
        raise ValueError("--max-retries must not exceed 5")
    benchmark = validate_official_checkout(
        args.official_root, include_evaluation_files=False
    )
    query_path = args.official_root / "data" / "prompt_data" / "query.jsonl"
    tasks = load_tasks(query_path)
    ids = _parse_task_ids(args.task_ids)
    selected = select_tasks(
        tasks,
        mode=args.command,
        task_ids=ids,
        language=args.language,
        max_tasks=args.max_tasks,
    )
    run_id = args.run_id or _new_run_id(args.command)
    config = resolve_generation_config()
    run_directory = (args.results_root / run_id).resolve()
    research = generation_metadata(config, run_directory=run_directory)
    git_commit = _git_value(["git", "rev-parse", "HEAD"])
    git_dirty = bool(_git_value(["git", "status", "--porcelain"]))
    source_sha256 = _source_digest()
    reproducibility = {
        "mode": args.command,
        "task_ids": [task.id for task in selected],
        "benchmark": benchmark,
        "research": research,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "concurrency": args.concurrency,
        "code": {"git_commit": git_commit, "source_sha256": source_sha256},
    }
    fingerprint = hashlib.sha256(
        json.dumps(reproducibility, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    manifest = {
        "run_id": run_id,
        "execution_mode": args.command,
        "task_count": len(selected),
        "task_ids": [task.id for task in selected],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_sha256": source_sha256,
        "benchmark": benchmark,
        "research": research,
        "limits": {
            "task_timeout_seconds": args.timeout_seconds,
            "max_tasks": args.max_tasks or len(selected),
            "api_retries": args.max_retries,
            "max_concurrency": 1,
        },
        "generation_input_boundary": "official data/prompt_data/query.jsonl only",
        "evaluation_status": "not_evaluated",
    }
    store = RunStore.create_or_resume(
        args.results_root,
        run_id,
        manifest=manifest,
        config_fingerprint=fingerprint,
    )
    _configure_file_logging(store.directory / "benchmark.log")
    with build_production_session(store.directory, config=config) as session:
        BenchmarkRunner(
            session.coordinator,
            store,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            usage_meter=session.usage_meter,
        ).run(selected, retry_failed=args.retry_failed)
    export = store.export_jsonl()
    write_reports(store.directory)
    print(f"run: {store.directory}")
    print(f"export: {export}")
    return 0


def _score(args: argparse.Namespace) -> int:
    validate_official_checkout(args.official_root)
    export = args.run_dir / "exports" / "insightagent.jsonl"
    evaluator = OfficialEvaluator(
        args.official_root,
        args.run_dir / "scores",
        export,
    )
    try:
        evaluator.score(
            race=args.race,
            fact=args.fact,
            confirm_paid=args.confirm_paid,
            environ=os.environ,
            workers=args.workers,
            force=args.force,
        )
    finally:
        _update_manifest_evaluation(args.run_dir)
        write_reports(args.run_dir)
    return 0


def _parse_task_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--task-ids must be comma-separated integers") from exc
    if not values:
        raise ValueError("--task-ids must not be empty")
    return values


def _new_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{mode}-{stamp}"


def _git_value(command: list[str]) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = [
        *root.joinpath("insight_agent").rglob("*.py"),
        *root.joinpath("evals").rglob("*.py"),
        root / "scripts" / "benchmarks" / "deepresearch.sh",
        root / "pyproject.toml",
    ]
    digest = hashlib.sha256()
    for path in sorted((path for path in paths if path.is_file()), key=lambda item: str(item)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _configure_file_logging(path: Path) -> None:
    resolved = path.resolve()
    root = logging.getLogger()
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == resolved
        for handler in root.handlers
    ):
        return
    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)


def _update_manifest_evaluation(run_directory: Path) -> None:
    manifest_path = run_directory / "manifest.json"
    evaluation_path = run_directory / "scores" / "evaluation.json"
    if not manifest_path.is_file() or not evaluation_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    requested = [
        evaluation.get(f"{name}_status")
        for name in ("race", "fact")
        if evaluation.get(f"{name}_status") != "not_requested"
    ]
    if requested and all(status == "succeeded" for status in requested):
        status = "complete"
    elif any(status == "failed" for status in requested):
        status = "failed"
    else:
        status = "partial"
    manifest["evaluation_status"] = status
    manifest["evaluation"] = {
        key: evaluation.get(key)
        for key in (
            "protocol",
            "official_commit",
            "backend",
            "api_base_url",
            "race_model",
            "clean_model",
            "fact_model",
            "race_status",
            "fact_status",
            "export_sha256",
            "race_provenance",
            "fact_provenance",
            "started_at",
            "ended_at",
        )
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("DeepResearch Bench interrupted; saved task state is resumable.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"DeepResearch Bench error: {exc}", file=sys.stderr)
        raise SystemExit(1)
