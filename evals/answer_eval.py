"""Structure-first Evaluation of production Structured Reports."""

from __future__ import annotations

from collections.abc import Mapping

from evals.judge import (
    EvaluationJudge,
    JudgeRequest,
    JudgeTask,
    evaluate_with_judge,
)
from evals.models import (
    AnswerEvalResult,
    ClaimEvalResult,
    EvalCase,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
)
from evals.structural_checks import (
    claim_semantic_precondition_errors,
    validate_report_structure,
)
from insight_agent.evidence import Evidence
from insight_agent.reporting import StructuredReport
from insight_agent.routing import RetrievalSource


_GROUNDEDNESS_RUBRIC = (
    "Judge whether every factual part of the Claim is supported by its bound Evidence.",
    "Do not use model knowledge or unbound Evidence.",
    "Use partial only when a supported core remains but a concrete qualifier is unsupported.",
)

_COVERAGE_RUBRIC = (
    "Judge whether the answer Claims cover every supplied required point.",
    "Use partial when meaningful coverage exists but at least one point is missing.",
)

_CONSTRAINT_RUBRIC = (
    "Judge whether the answer complies with every supplied explicit constraint.",
    "Use partial only when the core constraint is followed with a limited deviation.",
)


class AnswerEvaluator:
    """Evaluate report structure before independent Claim and answer semantics."""

    def __init__(self, judge: EvaluationJudge) -> None:
        self.judge = judge

    def evaluate(
        self,
        case: EvalCase,
        report: StructuredReport,
        evidence_by_id: Mapping[str, Evidence],
    ) -> AnswerEvalResult:
        if not isinstance(case, EvalCase):
            raise TypeError("case must be an EvalCase")
        if not isinstance(report, StructuredReport):
            raise TypeError("report must be a StructuredReport")
        if not isinstance(evidence_by_id, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, Evidence)
            for key, value in evidence_by_id.items()
        ):
            raise TypeError("evidence_by_id must map strings to Evidence")

        structural_errors = validate_report_structure(report, evidence_by_id)
        precondition_errors = claim_semantic_precondition_errors(
            report,
            evidence_by_id,
        )
        claim_results: list[ClaimEvalResult] = []
        for section in report.sections:
            for claim in section.claims:
                blockers = precondition_errors.get(claim.id, ())
                if blockers:
                    groundedness = _not_computable(
                        "Claim has structural prerequisite errors: "
                        + "; ".join(blockers)
                    )
                else:
                    bound_evidence = [
                        evidence_by_id[evidence_id]
                        for evidence_id in claim.evidence_ids
                    ]
                    groundedness = evaluate_with_judge(
                        self.judge,
                        JudgeRequest(
                            task=JudgeTask.CLAIM_GROUNDEDNESS,
                            case_id=case.case_id,
                            query=case.query,
                            rubric=_GROUNDEDNESS_RUBRIC,
                            payload={
                                "claim": {
                                    "id": claim.id,
                                    "task_id": claim.task_id,
                                    "text": claim.text,
                                    "evidence_ids": list(claim.evidence_ids),
                                },
                                "evidence": [
                                    _evidence_payload(item)
                                    for item in bound_evidence
                                ],
                            },
                        ),
                    )
                claim_results.append(ClaimEvalResult(claim.id, groundedness))

        report_payload = _report_payload(report)
        if case.required_points:
            coverage = evaluate_with_judge(
                self.judge,
                JudgeRequest(
                    task=JudgeTask.ANSWER_COVERAGE,
                    case_id=case.case_id,
                    query=case.query,
                    rubric=_COVERAGE_RUBRIC,
                    payload={
                        "required_points": list(case.required_points),
                        "answer": report_payload,
                    },
                ),
            )
        else:
            coverage = _not_computable(
                "case has no required_points Gold Label"
            )

        if case.constraints:
            constraints = evaluate_with_judge(
                self.judge,
                JudgeRequest(
                    task=JudgeTask.ANSWER_CONSTRAINTS,
                    case_id=case.case_id,
                    query=case.query,
                    rubric=_CONSTRAINT_RUBRIC,
                    payload={
                        "constraints": list(case.constraints),
                        "answer": report_payload,
                    },
                ),
            )
        else:
            constraints = _not_computable("case has no constraints Gold Label")

        completed = [
            item.groundedness
            for item in claim_results
            if item.groundedness.status is JudgeExecutionStatus.COMPLETED
            and item.groundedness.result is not None
        ]
        grounded_claims = (
            sum(
                outcome.result is not None
                and outcome.result.label is JudgeLabel.PASS
                for outcome in completed
            )
            if completed
            else None
        )
        return AnswerEvalResult(
            case_id=case.case_id,
            total_claims=len(claim_results),
            grounded_claims=grounded_claims,
            claim_results=tuple(claim_results),
            coverage=coverage,
            constraints=constraints,
            structural_errors=structural_errors,
        )


def _not_computable(reason: str) -> JudgeEvaluation:
    return JudgeEvaluation(
        status=JudgeExecutionStatus.NOT_COMPUTABLE,
        result=None,
        reason=reason,
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


def _report_payload(report: StructuredReport) -> dict[str, object]:
    return {
        "objective": report.objective,
        "sections": [
            {
                "task_id": section.task_id,
                "title": section.title,
                "claims": [
                    {"id": claim.id, "text": claim.text}
                    for claim in section.claims
                ],
            }
            for section in report.sections
        ],
    }
