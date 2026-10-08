"""Evaluation-owned data contracts kept outside the production Agent package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalMode(str, Enum):
    """Whether an Evaluation Run uses fixed fixtures or live dependencies."""

    OFFLINE = "offline"
    LIVE = "live"


class JudgeLabel(str, Enum):
    """Finite semantic-evaluation vocabulary reserved for a future Judge."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


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
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class EvidenceEvalResult:
    """Evidence-level checks, with optional future semantic judgment."""

    case_id: str
    provenance_valid_rate: float | None
    evidence_count: int
    provenance_errors: tuple[str, ...] = ()
    judge_label: JudgeLabel | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AnswerEvalResult:
    """Answer structure checks, with optional future semantic judgments."""

    case_id: str
    total_claims: int
    structural_errors: tuple[str, ...] = ()
    grounded_claims: int | None = None
    coverage_label: JudgeLabel | None = None
    constraint_label: JudgeLabel | None = None


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    """Observable execution cost without estimated usage values."""

    completed: bool
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_calls: int
    failed_calls: int


@dataclass(frozen=True, slots=True)
class EvalRunMetadata:
    """Configuration identity needed to interpret an Evaluation Run."""

    eval_run_id: str
    dataset_version: str
    git_commit: str
    model_name: str
    prompt_version: str
    index_version: str
    mode: EvalMode
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
