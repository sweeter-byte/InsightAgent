"""Offline aggregation and Chinese summary generation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from evals.deepresearch_bench.dataset import sha256_file


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def aggregate_metrics(run_directory: str | Path, *, total_tasks: int | None = None) -> dict[str, Any]:
    directory = Path(run_directory)
    records = []
    cases = directory / "cases"
    if cases.is_dir():
        for case in cases.iterdir():
            path = case / "execution.json"
            if path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
    total = total_tasks if total_tasks is not None else len(records)
    statuses = {name: sum(row.get("status") == name for row in records) for name in ("succeeded", "failed", "timeout", "interrupted")}
    durations = [float(row["elapsed_seconds"]) for row in records if isinstance(row.get("elapsed_seconds"), (int, float))]
    generated = sum(bool(row.get("report_generated")) for row in records)
    known_input = [row["input_tokens"] for row in records if isinstance(row.get("input_tokens"), int)]
    known_output = [row["output_tokens"] for row in records if isinstance(row.get("output_tokens"), int)]
    evaluation = _load_optional_json(directory / "scores" / "evaluation.json")
    export = directory / "exports" / "insightagent.jsonl"
    scoring_stale = bool(
        evaluation
        and (
            not export.is_file()
            or evaluation.get("export_sha256") != sha256_file(export)
        )
    )
    race = None
    fact = None
    if evaluation and not scoring_stale and evaluation.get("race_status") == "succeeded":
        race = _race_metrics(directory / "scores" / "race")
    if evaluation and not scoring_stale and evaluation.get("fact_status") == "succeeded":
        fact = _fact_metrics(
            directory / "scores" / "fact" / "fact_result.txt",
            directory / "scores" / "fact" / "validated.jsonl",
        )
    protocol = evaluation.get("protocol") if evaluation else None
    race_provenance = evaluation.get("race_provenance") if evaluation else None
    fact_provenance = evaluation.get("fact_provenance") if evaluation else None
    race_protocol = (
        race_provenance.get("protocol", protocol)
        if isinstance(race_provenance, dict)
        else protocol
    )
    fact_protocol = (
        fact_provenance.get("protocol", protocol)
        if isinstance(fact_provenance, dict)
        else protocol
    )
    custom_parts: dict[str, Any] = {}
    if race_protocol == "custom_judge" and race is not None:
        custom_parts["race"] = race
        race = None
    if fact_protocol == "custom_judge" and fact is not None:
        custom_parts["fact"] = fact
        fact = None
    custom = (
        {**custom_parts, "metadata": evaluation} if custom_parts else None
    )
    race_evaluated = 0
    if race is not None:
        race_evaluated = int(race.get("effective_samples", 0)) + int(
            race.get("failed_samples", 0)
        )
    fact_evaluated = _validated_fact_tasks(directory / "scores" / "fact" / "validated.jsonl")
    return {
        "total_tasks": total,
        "terminal_tasks": len(records),
        "successful_tasks": statuses["succeeded"],
        "failed_tasks": statuses["failed"],
        "timeout_tasks": statuses["timeout"],
        "interrupted_tasks": statuses["interrupted"],
        "report_generated_tasks": generated,
        "completion_rate": statuses["succeeded"] / total if total else None,
        "duration_seconds": {
            "mean": statistics.fmean(durations) if durations else None,
            "p50": statistics.median(durations) if durations else None,
            "p95": _percentile(durations, 0.95),
            "sample_count": len(durations),
        },
        "tokens": {
            "input_total": sum(known_input) if known_input else None,
            "output_total": sum(known_output) if known_output else None,
            "measured_input_tasks": len(known_input),
            "measured_output_tasks": len(known_output),
            "unavailable_tasks": len(records) - min(len(known_input), len(known_output)),
        },
        "race": race,
        "fact": fact,
        "custom_judge": custom,
        "evaluation_protocol": protocol,
        "scoring_stale": scoring_stale,
        "race_unscored_tasks": max(0, generated - race_evaluated),
        "fact_unscored_tasks": max(0, generated - fact_evaluated),
    }


def _race_metrics(directory: Path) -> dict[str, Any] | None:
    result = directory / "race_result.txt"
    raw = directory / "raw_results.jsonl"
    if not result.is_file():
        return None
    values: dict[str, float] = {}
    for line in result.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            values[name.strip().lower().replace(" ", "_")] = float(value.strip())
    rows = []
    if raw.is_file():
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    values["effective_samples"] = sum("error" not in row for row in rows)
    values["failed_samples"] = sum("error" in row for row in rows)
    return values


def _fact_metrics(path: Path, validated_path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            values[name.strip()] = float(value.strip())
    sample_counts = _fact_sample_counts(validated_path)
    return {
        "effective_citations": values.get("total_valid_citations"),
        "citation_accuracy": values.get("valid_rate"),
        "raw": values,
        **sample_counts,
    }


def _validated_fact_tasks(path: Path) -> int:
    if not path.is_file():
        return 0
    ids = {
        json.loads(line).get("id")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return len(ids - {None})


def _fact_sample_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {
            "processed_samples": 0,
            "effective_samples": 0,
            "no_citation_samples": 0,
            "validation_failed_samples": 0,
        }
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    no_citation = sum(not row.get("citations") for row in rows)
    failed = 0
    for row in rows:
        citations = row.get("citations_deduped", {})
        if isinstance(citations, dict) and any(
            isinstance(value, dict) and value.get("validate_error") is not None
            for value in citations.values()
        ):
            failed += 1
    return {
        "processed_samples": len(rows),
        "effective_samples": len(rows) - no_citation,
        "no_citation_samples": no_citation,
        "validation_failed_samples": failed,
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def write_reports(run_directory: str | Path) -> dict[str, Any]:
    directory = Path(run_directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metrics = aggregate_metrics(directory, total_tasks=int(manifest["task_count"]))
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = []
    executions = []
    for case in (directory / "cases").iterdir():
        execution = json.loads((case / "execution.json").read_text(encoding="utf-8"))
        executions.append(execution)
        if execution.get("status") != "succeeded":
            failures.append(execution)
    scoring = _load_optional_json(directory / "scores" / "evaluation.json")
    scoring_failures = {}
    if scoring:
        for name in ("race", "fact"):
            if scoring.get(f"{name}_status") == "failed":
                scoring_failures[name] = scoring.get(f"{name}_error")
    slowest = sorted(
        executions,
        key=lambda row: float(row.get("elapsed_seconds", 0)),
        reverse=True,
    )[:10]
    (directory / "bad_cases.json").write_text(
        json.dumps(
            {
                "execution_failures": failures,
                "scoring_failures": scoring_failures,
                "slowest_tasks": [
                    {
                        "task_id": row.get("task_id"),
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        "status": row.get("status"),
                    }
                    for row in slowest
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    race = "未评分" if metrics["race"] is None else str(metrics["race"])
    fact = "未评分" if metrics["fact"] is None else str(metrics["fact"])
    custom_line = (
        f"- Custom Judge：{metrics['custom_judge']}\n"
        if metrics.get("custom_judge") is not None
        else ""
    )
    summary = (
        "# DeepResearch Bench 实验汇总\n\n"
        f"- Run ID：`{manifest['run_id']}`\n"
        f"- 计划任务：{metrics['total_tasks']}\n"
        f"- 报告生成成功：{metrics['report_generated_tasks']}\n"
        f"- 失败：{metrics['failed_tasks']}，超时：{metrics['timeout_tasks']}，中断：{metrics['interrupted_tasks']}\n"
        f"- RACE：{race}\n"
        f"- FACT：{fact}\n"
        f"{custom_line}\n"
        "生成成功仅表示 InsightAgent 产出了非空报告，不表示 RACE 或 FACT 评分通过。"
        "未评分任务不会被当作 0 分计入官方均值。\n"
    )
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    return metrics
