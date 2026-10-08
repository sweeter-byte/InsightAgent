"""Independent deterministic and semantic Evaluation of Evidence pools."""

from __future__ import annotations

from evals.judge import (
    EvaluationJudge,
    JudgeRequest,
    JudgeTask,
    evaluate_with_judge,
)
from evals.models import (
    EvalCase,
    EvidenceEvalResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
)
from evals.structural_checks import (
    provenance_valid_rate,
    validate_evidence_provenance,
)
from insight_agent.evidence import Evidence
from insight_agent.routing import RetrievalSource


_RELEVANCE_RUBRIC = (
    "Judge whether the Evidence pool directly serves the Case query.",
    "Treat partially related material as partial and unrelated noise as fail.",
    "Use only the supplied Evidence content.",
)

_COVERAGE_RUBRIC = (
    "Judge whether the Evidence pool covers every supplied required point.",
    "Use partial when meaningful coverage exists but at least one point is missing.",
    "Use only the supplied Evidence content.",
)


class EvidenceEvaluator:
    """Evaluate production Evidence without production grading decisions."""

    def __init__(self, judge: EvaluationJudge) -> None:
        self.judge = judge

    def evaluate(
        self,
        case: EvalCase,
        evidence: list[Evidence],
    ) -> EvidenceEvalResult:
        if not isinstance(case, EvalCase):
            raise TypeError("case must be an EvalCase")
        if not isinstance(evidence, list) or any(
            not isinstance(item, Evidence) for item in evidence
        ):
            raise TypeError("evidence must be a list of Evidence objects")

        provenance_errors = tuple(
            error
            for item in evidence
            for error in validate_evidence_provenance(item)
        )
        evidence_payload = [_evidence_payload(item) for item in evidence]
        relevance = evaluate_with_judge(
            self.judge,
            JudgeRequest(
                task=JudgeTask.EVIDENCE_RELEVANCE,
                case_id=case.case_id,
                query=case.query,
                rubric=_RELEVANCE_RUBRIC,
                payload={"evidence": evidence_payload},
            ),
        )

        if case.required_points:
            coverage = evaluate_with_judge(
                self.judge,
                JudgeRequest(
                    task=JudgeTask.EVIDENCE_COVERAGE,
                    case_id=case.case_id,
                    query=case.query,
                    rubric=_COVERAGE_RUBRIC,
                    payload={
                        "required_points": list(case.required_points),
                        "evidence": evidence_payload,
                    },
                ),
            )
        else:
            coverage = JudgeEvaluation(
                status=JudgeExecutionStatus.NOT_COMPUTABLE,
                result=None,
                reason="case has no required_points Gold Label",
            )

        return EvidenceEvalResult(
            case_id=case.case_id,
            provenance_valid_rate=provenance_valid_rate(evidence),
            evidence_count=len(evidence),
            relevance=relevance,
            coverage=coverage,
            provenance_errors=provenance_errors,
        )


def _evidence_payload(evidence: Evidence) -> dict[str, object]:
    source = evidence.retrieval_source
    return {
        "id": evidence.id,
        "task_id": evidence.task_id,
        "content": evidence.content,
        "retrieval_source": (
            source.value if isinstance(source, RetrievalSource) else str(source)
        ),
        "origin_id": evidence.origin_id,
        "source": evidence.source,
    }
