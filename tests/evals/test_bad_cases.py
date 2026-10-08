from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from evals import (
    AnswerEvalResult,
    BadCaseCategory,
    CaseExecutionStatus,
    ClaimEvalResult,
    EvalCase,
    EvalMode,
    EvalRunMetadata,
    EvaluationResult,
    EvaluationType,
    EvidenceEvalResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    JudgeResult,
    MetricStatus,
    RetrievalEvalResult,
    RuntimeMetrics,
    extract_bad_cases,
    write_bad_cases,
)


def _metadata() -> EvalRunMetadata:
    return EvalRunMetadata(
        eval_run_id="run-1",
        dataset_version="dataset-v1",
        dataset_sha256="a" * 64,
        git_commit="abc123",
        model_name="model",
        prompt_version="prompt-v1",
        index_version="index-v1",
        retriever_configuration={"top_k": 5, "fixture_version": "fixed-v1"},
        mode=EvalMode.OFFLINE,
        evaluation_type=EvaluationType.MOCK_TEST,
        config_fingerprint="fingerprint",
    )


def _judgment(
    label: JudgeLabel | None = JudgeLabel.PASS,
    *,
    status: JudgeExecutionStatus = JudgeExecutionStatus.COMPLETED,
    reason: str = "reason",
) -> JudgeEvaluation:
    result = JudgeResult(label, reason) if label is not None else None
    return JudgeEvaluation(status=status, result=result, reason=reason)


def _result(case_id: str = "case-1") -> EvaluationResult:
    passed = _judgment()
    return EvaluationResult(
        metadata=_metadata(),
        case=EvalCase(
            case_id,
            "query",
            tags=("retrieval", "research"),
            gold_retrieval_ids=("gold-1",),
            required_points=("point",),
            constraints=("constraint",),
            fixture_set="fixed-v1",
        ),
        retrieval=RetrievalEvalResult(
            case_id,
            ("other",),
            0.0,
            0.0,
            MetricStatus.COMPUTED,
            pre_rerank_ids=("gold-1", "other"),
            post_rerank_ids=("other", "gold-1"),
            pre_rerank_reciprocal_rank=1.0,
            post_rerank_reciprocal_rank=0.5,
        ),
        evidence=EvidenceEvalResult(
            case_id,
            0.0,
            2,
            _judgment(JudgeLabel.FAIL, reason="irrelevant evidence"),
            _judgment(JudgeLabel.FAIL, reason="required point missing"),
            provenance_errors=("invalid_provenance:E1:source",),
        ),
        answer=AnswerEvalResult(
            case_id,
            1,
            (ClaimEvalResult("C1", _judgment(JudgeLabel.FAIL, reason="unsupported")),),
            _judgment(JudgeLabel.FAIL, reason="answer gap"),
            _judgment(JudgeLabel.FAIL, reason="constraint violated"),
            grounded_claims=0,
            structural_errors=("missing_citation:E1",),
        ),
        runtime=RuntimeMetrics(True, 12, 10, 4, 1, 0),
        artifact_paths={"result": "cases/case-1/result.json"},
    )


def test_extract_bad_cases_classifies_deterministic_and_semantic_failures() -> None:
    bad_cases = extract_bad_cases([_result()])

    assert {item.category for item in bad_cases} == set(BadCaseCategory) - {
        BadCaseCategory.RUNTIME_FAILURE
    }
    assert all(item.case_id == "case-1" for item in bad_cases)
    assert all(item.artifact_path == "cases/case-1/result.json" for item in bad_cases)
    assert all(item.summary for item in bad_cases)
    quality = [item for item in bad_cases if item.category is not BadCaseCategory.RUNTIME_FAILURE]
    assert all(item.failure_kind == "quality_failure" for item in quality)


def test_judge_errors_and_missing_labels_are_not_agent_quality_bad_cases() -> None:
    unavailable = _judgment(
        None,
        status=JudgeExecutionStatus.NOT_COMPUTABLE,
        reason="missing Gold Label",
    )
    timeout = _judgment(
        None,
        status=JudgeExecutionStatus.TIMEOUT,
        reason="judge timeout",
    )
    backend_error = _judgment(
        None,
        status=JudgeExecutionStatus.BACKEND_ERROR,
        reason="judge unavailable",
    )
    invalid = _judgment(
        None,
        status=JudgeExecutionStatus.INVALID_OUTPUT,
        reason="invalid judge JSON",
    )
    result = replace(
        _result("infrastructure-only"),
        retrieval=RetrievalEvalResult(
            "infrastructure-only",
            (),
            None,
            None,
            MetricStatus.NOT_COMPUTABLE,
            "missing retrieval Gold Label",
        ),
        evidence=EvidenceEvalResult(
            "infrastructure-only", None, 0, timeout, unavailable
        ),
        answer=AnswerEvalResult(
            "infrastructure-only",
            2,
            (
                ClaimEvalResult("C1", backend_error),
                ClaimEvalResult("C2", invalid),
            ),
            unavailable,
            timeout,
            grounded_claims=None,
        ),
        artifact_paths={"result": "cases/infrastructure-only/result.json"},
    )

    assert extract_bad_cases([result]) == ()


def test_case_execution_failure_is_an_operational_bad_case() -> None:
    result = replace(
        _result("runtime-failure"),
        retrieval=None,
        evidence=None,
        answer=None,
        runtime=RuntimeMetrics(False, 25, None, None, None, 1),
        execution_status=CaseExecutionStatus.FAILED,
        error="RuntimeError: workflow crashed",
        artifact_paths={"result": "cases/runtime-failure/result.json"},
    )

    assert extract_bad_cases([result]) == (
        replace(
            extract_bad_cases([result])[0],
            category=BadCaseCategory.RUNTIME_FAILURE,
            failure_kind="execution_failure",
        ),
    )
    assert extract_bad_cases([result])[0].summary == "RuntimeError: workflow crashed"


def test_write_bad_cases_uses_a_versioned_machine_readable_format(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "bad_cases.json"

    write_bad_cases(destination, "run-1", extract_bad_cases([_result()]))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["format_version"] == "evaluation_bad_cases.v1"
    assert payload["eval_run_id"] == "run-1"
    assert len(payload["bad_cases"]) == 7
