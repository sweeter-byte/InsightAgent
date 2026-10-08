"""Evaluation-owned data contracts kept outside the production Agent package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalMode(str, Enum):
    """Whether an Evaluation Run uses fixed fixtures or live dependencies."""

    OFFLINE = "offline"
    LIVE = "live"


class EvaluationType(str, Enum):
    """Whether outputs came from deterministic fakes or real components."""

    MOCK_TEST = "mock_test"
    LIVE_TEST = "live_test"


class CaseExecutionStatus(str, Enum):
    """Execution outcome kept separate from any quality judgment."""

    COMPLETED = "completed"
    FAILED = "failed"


class JudgeLabel(str, Enum):
    """Finite semantic-evaluation vocabulary returned by a valid Judge."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class MetricStatus(str, Enum):
    """Whether a deterministic metric had the human label it requires."""

    COMPUTED = "computed"
    NOT_COMPUTABLE = "not_computable"


class JudgeExecutionStatus(str, Enum):
    """Execution state kept separate from a semantic Judge label."""

    COMPLETED = "completed"
    NOT_COMPUTABLE = "not_computable"
    INVALID_OUTPUT = "invalid_output"
    BACKEND_ERROR = "backend_error"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """One valid finite-state semantic judgment."""

    label: JudgeLabel
    reason: str


@dataclass(frozen=True, slots=True)
class JudgeEvaluation:
    """Semantic execution outcome, including skips and infrastructure errors."""

    status: JudgeExecutionStatus
    result: JudgeResult | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ClaimEvalResult:
    """Groundedness execution retained for one production Claim."""

    claim_id: str
    groundedness: JudgeEvaluation


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One versioned evaluation input and its human-supplied expectations."""

    case_id: str
    query: str
    tags: tuple[str, ...] = ()
    gold_retrieval_ids: tuple[str, ...] = ()
    required_points: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    fixture_set: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalEvalResult:
    """Deterministic retrieval output and metrics for one Case."""

    case_id: str
    retrieved_ids: tuple[str, ...]
    recall_at_k: float | None
    reciprocal_rank: float | None
    metric_status: MetricStatus
    metric_reason: str = ""
    pre_rerank_ids: tuple[str, ...] = ()
    post_rerank_ids: tuple[str, ...] = ()
    pre_rerank_reciprocal_rank: float | None = None
    post_rerank_reciprocal_rank: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceEvalResult:
    """Deterministic provenance and independent semantic executions."""

    case_id: str
    provenance_valid_rate: float | None
    evidence_count: int
    relevance: JudgeEvaluation
    coverage: JudgeEvaluation
    provenance_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerEvalResult:
    """Structure-first Claim and whole-answer semantic executions."""

    case_id: str
    total_claims: int
    claim_results: tuple[ClaimEvalResult, ...]
    coverage: JudgeEvaluation
    constraints: JudgeEvaluation
    grounded_claims: int | None = None
    structural_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Observable execution cost without estimated usage values."""

    completed: bool
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_calls: int | None
    failed_calls: int | None


@dataclass(frozen=True, slots=True)
class EvalRunMetadata:
    """Configuration identity needed to interpret an Evaluation Run."""

    eval_run_id: str
    dataset_version: str
    dataset_sha256: str
    git_commit: str
    model_name: str
    prompt_version: str
    index_version: str
    retriever_configuration: dict[str, object]
    mode: EvalMode
    evaluation_type: EvaluationType
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Serializable per-Case artifact produced by later evaluation orchestration."""

    metadata: EvalRunMetadata
    case: EvalCase
    retrieval: RetrievalEvalResult | None
    evidence: EvidenceEvalResult | None
    answer: AnswerEvalResult | None
    runtime: RuntimeMetrics | None
    execution_status: CaseExecutionStatus = CaseExecutionStatus.COMPLETED
    error: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
