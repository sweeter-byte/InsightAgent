"""Executable, failure-isolated orchestration for layered Evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager, Iterator, Protocol
from unittest.mock import patch

from evals.bad_cases import extract_bad_cases, write_bad_cases
from evals.models import (
    AnswerEvalResult,
    CaseExecutionStatus,
    EvalCase,
    EvalMode,
    EvalRunMetadata,
    EvaluationResult,
    EvaluationType,
    EvidenceEvalResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    MetricStatus,
    RetrievalEvalResult,
    RuntimeMetrics,
)
from evals.serialization import write_evaluation_result, write_json_artifact
from evals.workflow import EvaluationWorkflowRunner
from insight_agent.evidence import Evidence
from insight_agent.reporting import StructuredReport


class RetrievalEvaluation(Protocol):
    def evaluate(self, case: EvalCase, *, top_k: int = 5) -> RetrievalEvalResult: ...


class EvidenceEvaluation(Protocol):
    def evaluate(self, case: EvalCase, evidence: list[Evidence]) -> EvidenceEvalResult: ...


class AnswerEvaluation(Protocol):
    def evaluate(
        self,
        case: EvalCase,
        report: StructuredReport,
        evidence_by_id: Mapping[str, Evidence],
    ) -> AnswerEvalResult: ...


class OfflineDependencyError(RuntimeError):
    """Offline Evaluation attempted to open a network connection."""


@dataclass(frozen=True, slots=True)
class EvaluationRunSummary:
    metadata: EvalRunMetadata
    results: tuple[EvaluationResult, ...]
    aggregates: dict[str, object]
    run_directory: Path


@contextmanager
def offline_network_guard() -> Iterator[None]:
    """Block socket connection and DNS entry points during Offline cases."""

    def denied(*args: object, **kwargs: object) -> None:
        raise OfflineDependencyError(
            "Offline Evaluation blocked a network connection; use a recorded "
            "fixture/local component or run an explicitly labelled Live Test"
        )

    with (
        patch.object(socket.socket, "connect", denied),
        patch.object(socket.socket, "connect_ex", denied),
        patch.object(socket, "create_connection", denied),
        patch.object(socket, "getaddrinfo", denied),
    ):
        yield


class EvaluationRunner:
    """Run applicable layers per Case and persist every outcome independently."""

    def __init__(
        self,
        *,
        metadata: EvalRunMetadata,
        output_root: str | Path,
        retrieval_evaluator: RetrievalEvaluation | None = None,
        workflow_runner: EvaluationWorkflowRunner | None = None,
        evidence_evaluator: EvidenceEvaluation | None = None,
        answer_evaluator: AnswerEvaluation | None = None,
        top_k: int = 5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(metadata, EvalRunMetadata):
            raise TypeError("metadata must be EvalRunMetadata")
        if metadata.mode is EvalMode.OFFLINE and (
            metadata.evaluation_type is not EvaluationType.MOCK_TEST
        ):
            raise ValueError("offline runs must be labelled mock_test")
        if metadata.mode is EvalMode.LIVE and (
            metadata.evaluation_type is not EvaluationType.LIVE_TEST
        ):
            raise ValueError("live runs must be labelled live_test")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.metadata = metadata
        self.output_root = Path(output_root)
        self.retrieval_evaluator = retrieval_evaluator
        self.workflow_runner = workflow_runner
        self.evidence_evaluator = evidence_evaluator
        self.answer_evaluator = answer_evaluator
        self.top_k = top_k
        self.clock = clock

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        case_ids: set[str] | None = None,
    ) -> EvaluationRunSummary:
        case_list = list(cases)
        if any(not isinstance(case, EvalCase) for case in case_list):
            raise TypeError("cases must contain EvalCase values")
        seen_case_ids: set[str] = set()
        duplicate_case_ids: set[str] = set()
        for case in case_list:
            if case.case_id in seen_case_ids:
                duplicate_case_ids.add(case.case_id)
            seen_case_ids.add(case.case_id)
        if duplicate_case_ids:
            raise ValueError(
                "duplicate case_id(s): " + ", ".join(sorted(duplicate_case_ids))
            )
        selected = _select_cases(case_list, case_ids)
        run_directory = self.output_root / _safe_component(self.metadata.eval_run_id)
        run_directory.mkdir(parents=True, exist_ok=False)

        results = tuple(
            self._run_and_write_case(case, run_directory=run_directory)
            for case in selected
        )
        aggregates = _aggregate_results(results)
        bad_cases = extract_bad_cases(results)
        write_bad_cases(
            run_directory / "bad_cases.json",
            self.metadata.eval_run_id,
            bad_cases,
        )
        manifest = {
            "format_version": "evaluation_run.v1",
            "metadata": self.metadata,
            "aggregates": aggregates,
            "bad_cases": "bad_cases.json",
            "cases": [
                {
                    "case_id": result.case.case_id,
                    "execution_status": result.execution_status,
                    "error": result.error,
                    "result": result.artifact_paths["result"],
                    "intermediates": result.artifact_paths["intermediates"],
                }
                for result in results
            ],
        }
        write_json_artifact(run_directory / "run.json", manifest)
        return EvaluationRunSummary(
            metadata=self.metadata,
            results=results,
            aggregates=aggregates,
            run_directory=run_directory,
        )

    def _run_and_write_case(
        self,
        case: EvalCase,
        *,
        run_directory: Path,
    ) -> EvaluationResult:
        component = _safe_case_component(case.case_id)
        result_relative = Path("cases") / component / "result.json"
        artifacts_relative = Path("cases") / component / "artifacts.json"
        started = self.clock()
        retrieval: RetrievalEvalResult | None = None
        evidence: EvidenceEvalResult | None = None
        answer: AnswerEvalResult | None = None
        runtime: RuntimeMetrics | None = None
        artifacts: dict[str, object] = {}
        status = CaseExecutionStatus.COMPLETED
        error: str | None = None

        guard: ContextManager[None] = (
            offline_network_guard()
            if self.metadata.mode is EvalMode.OFFLINE
            else nullcontext()
        )
        try:
            with guard:
                run_retrieval = "retrieval" in case.tags
                run_research = "research" in case.tags
                if not run_retrieval and not run_research:
                    raise ValueError(
                        "case must include an applicable 'retrieval' or 'research' tag"
                    )
                if run_retrieval:
                    if self.retrieval_evaluator is None:
                        raise RuntimeError("retrieval evaluator is not configured")
                    retrieval = self.retrieval_evaluator.evaluate(
                        case,
                        top_k=self.top_k,
                    )
                if run_research:
                    if self.workflow_runner is None:
                        raise RuntimeError("workflow runner is not configured")
                    workflow_output = self.workflow_runner.run(case)
                    runtime = workflow_output.runtime
                    artifacts.update(workflow_output.artifacts)
                    if self.evidence_evaluator is not None:
                        evidence = self.evidence_evaluator.evaluate(
                            case,
                            list(workflow_output.evidence_by_id.values()),
                        )
                    if self.answer_evaluator is not None:
                        answer = self.answer_evaluator.evaluate(
                            case,
                            workflow_output.report,
                            workflow_output.evidence_by_id,
                        )

                if runtime is None:
                    runtime = RuntimeMetrics(
                        completed=True,
                        latency_ms=max(0, round((self.clock() - started) * 1000)),
                        prompt_tokens=None,
                        completion_tokens=None,
                        retrieval_calls=1 if retrieval is not None else None,
                        failed_calls=0 if retrieval is not None else None,
                    )
        except Exception as exc:  # noqa: BLE001 - per-Case isolation boundary
            status = CaseExecutionStatus.FAILED
            detail = str(exc).strip()
            error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            if runtime is None:
                runtime = RuntimeMetrics(
                    completed=False,
                    latency_ms=max(0, round((self.clock() - started) * 1000)),
                    prompt_tokens=None,
                    completion_tokens=None,
                    retrieval_calls=1 if retrieval is not None else None,
                    failed_calls=None,
                )
            else:
                runtime = replace(runtime, completed=False)
            artifacts["execution_error"] = error

        artifact_paths = {
            "result": result_relative.as_posix(),
            "intermediates": artifacts_relative.as_posix(),
        }
        result = EvaluationResult(
            metadata=self.metadata,
            case=case,
            retrieval=retrieval,
            evidence=evidence,
            answer=answer,
            runtime=runtime,
            execution_status=status,
            error=error,
            artifact_paths=artifact_paths,
        )
        write_json_artifact(run_directory / artifacts_relative, artifacts)
        write_evaluation_result(run_directory / result_relative, result)
        return result


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(repo_root: str | Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def configuration_fingerprint(values: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(values),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_metadata(
    *,
    dataset_path: str | Path,
    dataset_version: str,
    repo_root: str | Path,
    model_name: str,
    prompt_version: str,
    index_version: str,
    retriever_configuration: dict[str, object],
    mode: EvalMode,
    evaluation_type: EvaluationType,
    eval_run_id: str | None = None,
) -> EvalRunMetadata:
    dataset_hash = dataset_sha256(dataset_path)
    commit = git_commit(repo_root)
    identity: dict[str, object] = {
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_hash,
        "git_commit": commit,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "index_version": index_version,
        "retriever_configuration": retriever_configuration,
        "mode": mode.value,
        "evaluation_type": evaluation_type.value,
    }
    resolved_run_id = eval_run_id or datetime.now(timezone.utc).strftime(
        "eval-%Y%m%dT%H%M%S%fZ"
    )
    return EvalRunMetadata(
        eval_run_id=resolved_run_id,
        dataset_version=dataset_version,
        dataset_sha256=dataset_hash,
        git_commit=commit,
        model_name=model_name,
        prompt_version=prompt_version,
        index_version=index_version,
        retriever_configuration=dict(retriever_configuration),
        mode=mode,
        evaluation_type=evaluation_type,
        config_fingerprint=configuration_fingerprint(identity),
    )


def _select_cases(cases: list[EvalCase], case_ids: set[str] | None) -> list[EvalCase]:
    if case_ids is None:
        return cases
    known = {case.case_id for case in cases}
    unknown = case_ids - known
    if unknown:
        raise ValueError("unknown case_id(s): " + ", ".join(sorted(unknown)))
    return [case for case in cases if case.case_id in case_ids]


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not component:
        raise ValueError("artifact path component has no safe characters")
    return component


def _safe_case_component(case_id: str) -> str:
    component = _safe_component(case_id)
    if component == case_id:
        return component
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8]
    return f"{component}-{digest}"


def _aggregate_results(
    results: tuple[EvaluationResult, ...],
) -> dict[str, object]:
    completed = [
        result
        for result in results
        if result.execution_status is CaseExecutionStatus.COMPLETED
    ]
    retrieval_results = [
        result.retrieval
        for result in completed
        if result.retrieval is not None
        and result.retrieval.metric_status is MetricStatus.COMPUTED
    ]
    labels = {label.value: 0 for label in JudgeLabel}
    statuses = {status.value: 0 for status in JudgeExecutionStatus}
    for result in completed:
        for judgment in _judge_outcomes(result):
            statuses[judgment.status.value] += 1
            if (
                judgment.status is JudgeExecutionStatus.COMPLETED
                and judgment.result is not None
            ):
                labels[judgment.result.label.value] += 1

    recalls = [
        item.recall_at_k
        for item in retrieval_results
        if item.recall_at_k is not None
    ]
    reciprocal_ranks = [
        item.reciprocal_rank
        for item in retrieval_results
        if item.reciprocal_rank is not None
    ]
    evidence_results = [
        result.evidence
        for result in completed
        if result.evidence is not None
    ]
    evidence_relevance = _completed_judgments(
        [item.relevance for item in evidence_results]
    )
    evidence_coverage = _completed_judgments(
        [item.coverage for item in evidence_results]
    )
    provenance_rates = [
        item.provenance_valid_rate
        for item in evidence_results
        if item.provenance_valid_rate is not None
    ]
    answer_results = [
        result.answer for result in completed if result.answer is not None
    ]
    groundedness = _completed_judgments(
        [
            claim.groundedness
            for item in answer_results
            for claim in item.claim_results
        ]
    )
    answer_coverage = _completed_judgments(
        [item.coverage for item in answer_results]
    )
    constraints = _completed_judgments(
        [item.constraints for item in answer_results]
    )
    runtimes = [result.runtime for result in results if result.runtime is not None]
    latencies = [float(item.latency_ms) for item in runtimes]
    prompt_tokens = [
        item.prompt_tokens for item in runtimes if item.prompt_tokens is not None
    ]
    completion_tokens = [
        item.completion_tokens
        for item in runtimes
        if item.completion_tokens is not None
    ]
    return {
        "cases": {
            "total": len(results),
            "completed": len(completed),
            "failed": len(results) - len(completed),
        },
        "retrieval": {
            "computed_cases": len(retrieval_results),
            "mean_recall_at_k": _mean(recalls),
            "mrr": _mean(reciprocal_ranks),
        },
        "judge": {
            "semantic_labels": labels,
            "execution_statuses": statuses,
            "semantic_denominator": sum(labels.values()),
        },
        "evidence": {
            "provenance_valid_rate": _mean(provenance_rates),
            "provenance_denominator": len(provenance_rates),
            "relevance_pass_rate": _pass_rate(evidence_relevance),
            "relevance_denominator": len(evidence_relevance),
            "coverage_pass_rate": _pass_rate(evidence_coverage),
            "coverage_denominator": len(evidence_coverage),
        },
        "answer": {
            "grounded_claim_pass_rate": _pass_rate(groundedness),
            "grounded_claim_denominator": len(groundedness),
            "coverage_pass_rate": _pass_rate(answer_coverage),
            "coverage_denominator": len(answer_coverage),
            "constraint_pass_rate": _pass_rate(constraints),
            "constraint_denominator": len(constraints),
        },
        "runtime": {
            "completion_rate": (
                sum(item.completed for item in runtimes) / len(results)
                if results
                else None
            ),
            "mean_latency_ms": _mean(latencies),
            "p95_latency_ms": _percentile_95(latencies),
            "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "prompt_token_observations": len(prompt_tokens),
            "total_completion_tokens": (
                sum(completion_tokens) if completion_tokens else None
            ),
            "completion_token_observations": len(completion_tokens),
        },
    }


def _judge_outcomes(result: EvaluationResult) -> list[JudgeEvaluation]:
    outcomes: list[JudgeEvaluation] = []
    if result.evidence is not None:
        outcomes.extend((result.evidence.relevance, result.evidence.coverage))
    if result.answer is not None:
        outcomes.extend(
            item.groundedness for item in result.answer.claim_results
        )
        outcomes.extend((result.answer.coverage, result.answer.constraints))
    return outcomes


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _completed_judgments(
    judgments: list[JudgeEvaluation],
) -> list[JudgeEvaluation]:
    return [
        item
        for item in judgments
        if item.status is JudgeExecutionStatus.COMPLETED and item.result is not None
    ]


def _pass_rate(judgments: list[JudgeEvaluation]) -> float | None:
    if not judgments:
        return None
    return sum(
        item.result is not None and item.result.label is JudgeLabel.PASS
        for item in judgments
    ) / len(judgments)


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
