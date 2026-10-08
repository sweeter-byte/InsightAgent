from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from evals import (
    AnswerEvalResult,
    EvalCase,
    EvalMode,
    EvalRunMetadata,
    EvaluationResult,
    EvidenceEvalResult,
    JudgeLabel,
    RetrievalEvalResult,
    RuntimeMetrics,
)


def test_evaluation_models_form_one_immutable_case_result() -> None:
    case = EvalCase(case_id="case-1", query="为什么需要 overlap？")
    metadata = EvalRunMetadata(
        eval_run_id="eval-1",
        dataset_version="v1",
        git_commit="abc123",
        model_name="offline-fixture",
        prompt_version="none",
        index_version="index-v1",
        mode=EvalMode.OFFLINE,
        config_fingerprint="fingerprint",
    )
    retrieval = RetrievalEvalResult(
        case_id=case.case_id,
        retrieved_ids=("chunk-1",),
        recall_at_k=1.0,
        reciprocal_rank=1.0,
    )
    evidence = EvidenceEvalResult(
        case_id=case.case_id,
        provenance_valid_rate=1.0,
        evidence_count=1,
        judge_label=JudgeLabel.PASS,
    )
    answer = AnswerEvalResult(
        case_id=case.case_id,
        total_claims=1,
        grounded_claims=1,
        coverage_label=JudgeLabel.PASS,
        constraint_label=JudgeLabel.PASS,
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
