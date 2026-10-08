from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from evals import (
    AnswerEvalResult,
    CaseExecutionStatus,
    EvalCase,
    EvalMode,
    EvalRunMetadata,
    EvaluationType,
    EvaluationResult,
    EvidenceEvalResult,
    ClaimEvalResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    JudgeResult,
    MetricStatus,
    RetrievalEvalResult,
    RuntimeMetrics,
)


def test_evaluation_models_form_one_immutable_case_result() -> None:
    case = EvalCase(case_id="case-1", query="为什么需要 overlap？")
    metadata = EvalRunMetadata(
        eval_run_id="eval-1",
        dataset_version="v1",
        dataset_sha256="a" * 64,
        git_commit="abc123",
        model_name="offline-fixture",
        prompt_version="none",
        index_version="index-v1",
        retriever_configuration={"top_k": 5},
        mode=EvalMode.OFFLINE,
        evaluation_type=EvaluationType.MOCK_TEST,
        config_fingerprint="fingerprint",
    )
    retrieval = RetrievalEvalResult(
        case_id=case.case_id,
        retrieved_ids=("chunk-1",),
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        metric_status=MetricStatus.COMPUTED,
    )
    passed = JudgeEvaluation(
        status=JudgeExecutionStatus.COMPLETED,
        result=JudgeResult(JudgeLabel.PASS, "The rubric is satisfied."),
    )
    evidence = EvidenceEvalResult(
        case_id=case.case_id,
        provenance_valid_rate=1.0,
        evidence_count=1,
        relevance=passed,
        coverage=passed,
    )
    answer = AnswerEvalResult(
        case_id=case.case_id,
        total_claims=1,
        grounded_claims=1,
        claim_results=(ClaimEvalResult("T1-C1", passed),),
        coverage=passed,
        constraints=passed,
    )
    runtime = RuntimeMetrics(
        completed=True,
        latency_ms=12,
        prompt_tokens=None,
        completion_tokens=None,
        retrieval_calls=1,
        failed_calls=0,
    )

    result = EvaluationResult(
        metadata=metadata,
        case=case,
        retrieval=retrieval,
        evidence=evidence,
        answer=answer,
        runtime=runtime,
    )

    assert result.case.tags == ()
    assert result.case.metadata == {}
    assert result.metadata.mode is EvalMode.OFFLINE
    assert result.runtime is not None
    assert result.runtime.prompt_tokens is None
    assert result.answer is not None
    assert result.answer.structural_errors == ()

    with pytest.raises(FrozenInstanceError):
        result.case.query = "changed"  # type: ignore[misc]


def test_unlabelled_and_infrastructure_outcomes_have_no_semantic_label() -> None:
    retrieval = RetrievalEvalResult(
        case_id="case-1",
        retrieved_ids=("chunk-1",),
        recall_at_k=None,
        reciprocal_rank=None,
        metric_status=MetricStatus.NOT_COMPUTABLE,
        metric_reason="case has no retrieval Gold Label",
    )
    timed_out = JudgeEvaluation(
        status=JudgeExecutionStatus.TIMEOUT,
        result=None,
        reason="judge exceeded 2 seconds",
    )

    assert retrieval.reciprocal_rank is None
    assert retrieval.metric_status is MetricStatus.NOT_COMPUTABLE
    assert timed_out.result is None


def test_run_metadata_and_case_status_keep_mock_results_explicit() -> None:
    metadata = EvalRunMetadata(
        eval_run_id="eval-1",
        dataset_version="research-smoke-v1",
        dataset_sha256="a" * 64,
        git_commit="abc123",
        model_name="fixture-model",
        prompt_version="judge-v1",
        index_version="smoke-index-v1",
        retriever_configuration={"top_k": 3, "reranker": "fixture"},
        mode=EvalMode.OFFLINE,
        evaluation_type=EvaluationType.MOCK_TEST,
        config_fingerprint="fingerprint",
    )
    result = EvaluationResult(
        metadata=metadata,
        case=EvalCase(case_id="case-1", query="query"),
        retrieval=None,
        evidence=None,
        answer=None,
        runtime=RuntimeMetrics(
            completed=False,
            latency_ms=5,
            prompt_tokens=None,
            completion_tokens=None,
            retrieval_calls=None,
            failed_calls=None,
        ),
        execution_status=CaseExecutionStatus.FAILED,
        error="RuntimeError: fixture failed",
        artifact_paths={"intermediates": "cases/case-1/artifacts.json"},
    )

    assert metadata.evaluation_type is EvaluationType.MOCK_TEST
    assert metadata.retriever_configuration == {"top_k": 3, "reranker": "fixture"}
    assert result.execution_status is CaseExecutionStatus.FAILED
    assert result.runtime is not None and result.runtime.retrieval_calls is None
    assert result.artifact_paths["intermediates"].endswith("artifacts.json")


def test_production_package_does_not_import_evaluation_package() -> None:
    production_root = Path(__file__).resolve().parents[2] / "insight_agent"
    violations: list[str] = []

    for path in production_root.rglob("*.py"):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("from evals") or stripped.startswith(
                "import evals"
            ):
                violations.append(
                    f"{path.relative_to(production_root)}:{line_number}:{stripped}"
                )

    assert violations == []
