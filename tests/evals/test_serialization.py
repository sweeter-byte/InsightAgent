from __future__ import annotations

import json
from dataclasses import replace
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
    evaluation_result_to_dict,
    evaluation_result_to_json,
    write_evaluation_result,
)


def _result() -> EvaluationResult:
    case = EvalCase(
        case_id="case-1",
        query="为什么文本分块需要重叠？",
        tags=("local", "citation"),
        gold_retrieval_ids=("chunk-1",),
        required_points=("保留边界上下文",),
        constraints=("简洁",),
        metadata={"owner": "human", "priority": 1},
    )
    return EvaluationResult(
        metadata=EvalRunMetadata(
            eval_run_id="eval-1",
            dataset_version="v1",
            git_commit="abc123",
            model_name="offline-fixture",
            prompt_version="none",
            index_version="index-v1",
            mode=EvalMode.OFFLINE,
            config_fingerprint="fingerprint",
        ),
        case=case,
        retrieval=RetrievalEvalResult(
            case_id=case.case_id,
            retrieved_ids=("chunk-1", "chunk-2"),
            recall_at_k=1.0,
            reciprocal_rank=1.0,
        ),
        evidence=EvidenceEvalResult(
            case_id=case.case_id,
            provenance_valid_rate=1.0,
            evidence_count=1,
            judge_label=JudgeLabel.PASS,
        ),
        answer=AnswerEvalResult(
            case_id=case.case_id,
            total_claims=1,
            grounded_claims=1,
            coverage_label=JudgeLabel.PASS,
            constraint_label=JudgeLabel.PASS,
        ),
        runtime=RuntimeMetrics(
            completed=True,
            latency_ms=15,
            prompt_tokens=None,
            completion_tokens=None,
            retrieval_calls=1,
            failed_calls=0,
        ),
    )


def test_evaluation_result_to_dict_converts_nested_json_types() -> None:
    payload = evaluation_result_to_dict(_result())

    assert payload["metadata"]["mode"] == "offline"
    assert payload["case"]["tags"] == ["local", "citation"]
    assert payload["evidence"]["judge_label"] == "pass"
    assert payload["runtime"]["prompt_tokens"] is None


def test_evaluation_result_to_json_preserves_unicode() -> None:
    payload = evaluation_result_to_dict(_result())

    text = evaluation_result_to_json(_result())

    assert "为什么" in text
    assert json.loads(text) == payload


def test_write_evaluation_result_writes_utf8_json_with_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"

    write_evaluation_result(path, _result())

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == evaluation_result_to_dict(_result())


def test_serialization_rejects_non_finite_numbers() -> None:
    result = _result()
    assert result.retrieval is not None
    invalid = replace(
        result,
        retrieval=replace(result.retrieval, recall_at_k=float("nan")),
    )

    with pytest.raises(ValueError):
        evaluation_result_to_json(invalid)


def test_serialization_rejects_non_string_mapping_keys() -> None:
    result = _result()
    invalid_case = replace(result.case, metadata={1: "invalid"})  # type: ignore[dict-item]

    with pytest.raises(TypeError, match="mapping keys"):
        evaluation_result_to_dict(replace(result, case=invalid_case))
