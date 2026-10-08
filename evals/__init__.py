"""Repository-level, offline Evaluation Harness foundations."""

from evals.dataset import EvalDatasetError, load_eval_cases
from evals.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank
from evals.models import (
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
from evals.serialization import (
    evaluation_result_to_dict,
    evaluation_result_to_json,
    write_evaluation_result,
)
from evals.structural_checks import (
    provenance_valid_rate,
    validate_citations,
    validate_claim_bindings,
    validate_evidence_provenance,
    validate_report_structure,
)

__all__ = [
    "AnswerEvalResult",
    "EvalCase",
    "EvalDatasetError",
    "EvalMode",
    "EvalRunMetadata",
    "EvaluationResult",
    "EvidenceEvalResult",
    "JudgeLabel",
    "RetrievalEvalResult",
    "RuntimeMetrics",
    "evaluation_result_to_dict",
    "evaluation_result_to_json",
    "load_eval_cases",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "provenance_valid_rate",
    "validate_citations",
    "validate_claim_bindings",
    "validate_evidence_provenance",
    "validate_report_structure",
    "write_evaluation_result",
]
