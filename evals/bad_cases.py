"""Classify persisted Evaluation failures without rerunning any evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from evals.models import (
    CaseExecutionStatus,
    EvaluationResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    MetricStatus,
)
from evals.serialization import write_json_artifact


BAD_CASE_FORMAT_VERSION = "evaluation_bad_cases.v1"


class BadCaseCategory(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    RERANK_DROP = "rerank_drop"
    EVIDENCE_NOISE = "evidence_noise"
    EVIDENCE_GAP = "evidence_gap"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_ERROR = "citation_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    RUNTIME_FAILURE = "runtime_failure"


@dataclass(frozen=True, slots=True)
class BadCase:
    case_id: str
    category: BadCaseCategory
    summary: str
    artifact_path: str
    failure_kind: str = "quality_failure"


def extract_bad_cases(results: list[EvaluationResult] | tuple[EvaluationResult, ...]) -> tuple[BadCase, ...]:
    """Return confirmed failures only; unavailable Judge outcomes stay excluded."""
    bad_cases: list[BadCase] = []
    for result in results:
        if not isinstance(result, EvaluationResult):
            raise TypeError("results must contain EvaluationResult values")
        artifact_path = result.artifact_paths.get("result", "")
        if result.execution_status is CaseExecutionStatus.FAILED or (
            result.runtime is not None and not result.runtime.completed
        ):
            bad_cases.append(
                BadCase(
                    case_id=result.case.case_id,
                    category=BadCaseCategory.RUNTIME_FAILURE,
                    summary=result.error or "Case execution did not complete",
                    artifact_path=artifact_path,
                    failure_kind="execution_failure",
                )
            )
            continue

        retrieval = result.retrieval
        if (
            retrieval is not None
            and retrieval.metric_status is MetricStatus.COMPUTED
            and retrieval.recall_at_k is not None
            and retrieval.recall_at_k < 1.0
        ):
            _append(
                bad_cases,
                result,
                BadCaseCategory.RETRIEVAL_MISS,
                f"Recall@K was {retrieval.recall_at_k:.6g}",
            )
        if (
            retrieval is not None
            and retrieval.pre_rerank_reciprocal_rank is not None
            and retrieval.post_rerank_reciprocal_rank is not None
            and retrieval.post_rerank_reciprocal_rank
            < retrieval.pre_rerank_reciprocal_rank
        ):
            _append(
                bad_cases,
                result,
                BadCaseCategory.RERANK_DROP,
                "Reranking reduced reciprocal rank from "
                f"{retrieval.pre_rerank_reciprocal_rank:.6g} to "
                f"{retrieval.post_rerank_reciprocal_rank:.6g}",
            )

        evidence = result.evidence
        if evidence is not None:
            if _is_fail(evidence.relevance):
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.EVIDENCE_NOISE,
                    _reason(evidence.relevance, "Evidence relevance failed"),
                )
            if _is_fail(evidence.coverage):
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.EVIDENCE_GAP,
                    _reason(evidence.coverage, "Evidence coverage failed"),
                )
            if evidence.provenance_errors:
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.CITATION_ERROR,
                    "; ".join(evidence.provenance_errors),
                )

        answer = result.answer
        if answer is not None:
            failed_claims = [
                item
                for item in answer.claim_results
                if _is_fail(item.groundedness)
            ]
            if failed_claims:
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.UNSUPPORTED_CLAIM,
                    "Unsupported Claim(s): "
                    + ", ".join(item.claim_id for item in failed_claims),
                )
            if _is_fail(answer.coverage):
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.EVIDENCE_GAP,
                    _reason(answer.coverage, "Answer coverage failed"),
                )
            if answer.structural_errors:
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.CITATION_ERROR,
                    "; ".join(answer.structural_errors),
                )
            if _is_fail(answer.constraints):
                _append(
                    bad_cases,
                    result,
                    BadCaseCategory.CONSTRAINT_VIOLATION,
                    _reason(answer.constraints, "Answer constraints failed"),
                )

    return tuple(bad_cases)


def write_bad_cases(
    path: str | Path,
    eval_run_id: str,
    bad_cases: tuple[BadCase, ...] | list[BadCase],
) -> None:
    write_json_artifact(
        path,
        {
            "format_version": BAD_CASE_FORMAT_VERSION,
            "eval_run_id": eval_run_id,
            "bad_cases": list(bad_cases),
        },
    )


def _append(
    output: list[BadCase],
    result: EvaluationResult,
    category: BadCaseCategory,
    summary: str,
) -> None:
    if any(
        item.case_id == result.case.case_id and item.category is category
        for item in output
    ):
        return
    output.append(
        BadCase(
            case_id=result.case.case_id,
            category=category,
            summary=summary,
            artifact_path=result.artifact_paths.get("result", ""),
        )
    )


def _is_fail(judgment: JudgeEvaluation) -> bool:
    return (
        judgment.status is JudgeExecutionStatus.COMPLETED
        and judgment.result is not None
        and judgment.result.label is JudgeLabel.FAIL
    )


def _reason(judgment: JudgeEvaluation, fallback: str) -> str:
    if judgment.result is not None and judgment.result.reason.strip():
        return judgment.result.reason.strip()
    return judgment.reason.strip() or fallback
