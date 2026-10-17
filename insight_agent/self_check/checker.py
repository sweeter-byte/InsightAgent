"""LLM-backed semantic checking with deterministic runtime validation."""

from __future__ import annotations

import json
from typing import Any

from insight_agent.evidence import Evidence, EvidenceAssessment
from insight_agent.llm import LLMClient
from insight_agent.planning import ResearchPlan, validate_research_plan
from insight_agent.reporting import Claim, StructuredReport, assemble_report
from insight_agent.self_check.models import (
    SelfCheckError,
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)
from insight_agent.self_check.prompts import SELF_CHECK_SYSTEM_PROMPT


MAX_SELF_CHECK_EVIDENCE_ITEMS = 10
MAX_SELF_CHECK_EVIDENCE_CONTENT_CHARS = 6000
MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS = 40000
SELF_CHECK_CONTENT_OMISSION_MARKER = (
    "\n\n[... CONTENT OMITTED FROM SELF CHECK VIEW ...]\n\n"
)


class ReportSelfChecker:
    """Judge report fidelity without changing research state or retrieving."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def check(
        self,
        *,
        plan: ResearchPlan,
        report: StructuredReport,
        evidence_pool: dict[str, list[Evidence]],
        assessments: dict[str, list[EvidenceAssessment]],
    ) -> SelfCheckResult:
        latest = self._validate_inputs(plan, report, evidence_pool, assessments)
        evidence_by_id = {
            evidence.id: evidence
            for items in evidence_pool.values()
            for evidence in items
        }
        task_by_id = {task.id: task for task in plan.tasks}
        referenced_evidence_ids: list[str] = []
        seen_evidence_ids: set[str] = set()
        for section in report.sections:
            for claim in section.claims:
                for evidence_id in claim.evidence_ids:
                    if evidence_id not in seen_evidence_ids:
                        seen_evidence_ids.add(evidence_id)
                        referenced_evidence_ids.append(evidence_id)

        # First-reference order is deterministic and prioritizes Evidence that the
        # report actually cites. A future optimization may use claim-aware fair
        # allocation if reports routinely exceed this bounded single-pass view.
        selected_evidence_ids = referenced_evidence_ids[
            :MAX_SELF_CHECK_EVIDENCE_ITEMS
        ]
        selected_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in selected_evidence_ids
        ]
        content_budgets = self._allocate_content_budgets(selected_evidence)
        provided_evidence_ids = set(selected_evidence_ids)
        claim_contexts: list[dict[str, Any]] = []
        unprovided_by_claim: dict[str, list[str]] = {}
        for section in report.sections:
            task = task_by_id[section.task_id]
            assessment = latest[section.task_id]
            for claim in section.claims:
                provided_ids = [
                    evidence_id
                    for evidence_id in claim.evidence_ids
                    if evidence_id in provided_evidence_ids
                ]
                unprovided_ids = [
                    evidence_id
                    for evidence_id in claim.evidence_ids
                    if evidence_id not in provided_evidence_ids
                ]
                if unprovided_ids:
                    unprovided_by_claim[claim.id] = unprovided_ids
                claim_contexts.append(
                    {
                        "task": {"id": task.id, "question": task.question},
                        "claim": {
                            "id": claim.id,
                            "text": claim.text,
                            "evidence_ids": list(claim.evidence_ids),
                        },
                        "provided_evidence_ids": provided_ids,
                        "unprovided_evidence_ids": unprovided_ids,
                        "latest_assessment": self._assessment_payload(assessment),
                    }
                )

        payload = {
            "plan": {
                "objective": plan.objective,
                "constraints": list(plan.constraints),
            },
            "report_sections": [
                {
                    "task_id": section.task_id,
                    "title": section.title,
                    "claims": [
                        {
                            "id": claim.id,
                            "text": claim.text,
                            "evidence_ids": list(claim.evidence_ids),
                        }
                        for claim in section.claims
                    ],
                    "sufficient": section.sufficient,
                    "missing_information": list(section.missing_information),
                }
                for section in report.sections
            ],
            "task_assessments": {
                task.id: self._assessment_payload(latest[task.id])
                for task in plan.tasks
            },
            "evidence_view": {
                "bounded": True,
                "selection_rule": (
                    "unique Evidence in first Claim-reference order"
                ),
                "referenced_evidence_count": len(referenced_evidence_ids),
                "provided_evidence_count": len(selected_evidence_ids),
                "max_evidence_items": MAX_SELF_CHECK_EVIDENCE_ITEMS,
                "max_content_chars_per_evidence": (
                    MAX_SELF_CHECK_EVIDENCE_CONTENT_CHARS
                ),
                "max_total_content_chars": (
                    MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS
                ),
            },
            "claim_contexts": claim_contexts,
            "evidence_catalog": {
                evidence.id: {
                    "evidence_id": evidence.id,
                    "content": self._excerpt_content(
                        evidence.content,
                        content_budget,
                    ),
                    "content_is_excerpt": len(evidence.content) > content_budget,
                    "original_content_char_count": len(evidence.content),
                    "retrieval_source": evidence.retrieval_source.value,
                    "origin_id": evidence.origin_id,
                    "source": evidence.source,
                    "metadata": evidence.metadata,
                }
                for evidence, content_budget in zip(
                    selected_evidence,
                    content_budgets,
                    strict=True,
                )
            },
        }
        try:
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SelfCheckError(
                "self-check input must contain JSON-serializable values"
            ) from exc
        response = self.llm.chat(
            [
                {"role": "system", "content": SELF_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": serialized},
            ],
            tools=None,
        )
        raw_text = self._extract_text(response)
        if not raw_text.strip():
            raise SelfCheckError("self-check response did not contain JSON")
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SelfCheckError(
                f"self-check returned invalid JSON: {exc.msg}"
            ) from exc
        parsed = self._parse_result(result, plan=plan, report=report)
        return self._enforce_bounded_view_result(
            parsed,
            report=report,
            unprovided_by_claim=unprovided_by_claim,
        )

    @staticmethod
    def _allocate_content_budgets(evidence: list[Evidence]) -> list[int]:
        """Allocate the total body budget without truncating short items."""
        target_sizes = [
            min(len(item.content), MAX_SELF_CHECK_EVIDENCE_CONTENT_CHARS)
            for item in evidence
        ]
        if sum(target_sizes) <= MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS:
            return target_sizes

        budgets = [0] * len(target_sizes)
        remaining_budget = MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS
        active = set(range(len(target_sizes)))
        while active:
            fair_share = remaining_budget // len(active)
            completed = [
                index for index in active if target_sizes[index] <= fair_share
            ]
            if completed:
                for index in completed:
                    budgets[index] = target_sizes[index]
                    remaining_budget -= budgets[index]
                    active.remove(index)
                continue

            ordered_active = sorted(active)
            for index in ordered_active:
                budgets[index] = fair_share
            remainder = remaining_budget - fair_share * len(ordered_active)
            for index in ordered_active[:remainder]:
                budgets[index] += 1
            break
        return budgets

    @staticmethod
    def _excerpt_content(content: str, limit: int) -> str:
        """Return an exact-size head/tail excerpt when content exceeds limit."""
        if len(content) <= limit:
            return content
        marker_size = len(SELF_CHECK_CONTENT_OMISSION_MARKER)
        if limit <= marker_size:
            return SELF_CHECK_CONTENT_OMISSION_MARKER[:limit]
        retained_chars = limit - marker_size
        head_chars = (retained_chars + 1) // 2
        tail_chars = retained_chars // 2
        return (
            content[:head_chars]
            + SELF_CHECK_CONTENT_OMISSION_MARKER
            + content[-tail_chars:]
        )

    @staticmethod
    def _enforce_bounded_view_result(
        result: SelfCheckResult,
        *,
        report: StructuredReport,
        unprovided_by_claim: dict[str, list[str]],
    ) -> SelfCheckResult:
        """Prevent omitted Evidence from producing a false complete pass."""
        if not unprovided_by_claim:
            return result

        existing_unverified_claims = {
            issue.claim_id
            for issue in result.issues
            if issue.code is SelfCheckIssueCode.UNSUPPORTED_CLAIM
        }
        issues = list(result.issues)
        for section in report.sections:
            for claim in section.claims:
                missing_ids = unprovided_by_claim.get(claim.id)
                if not missing_ids or claim.id in existing_unverified_claims:
                    continue
                issues.append(
                    SelfCheckIssue(
                        code=SelfCheckIssueCode.UNSUPPORTED_CLAIM,
                        reason=(
                            "Claim could not be fully verified because referenced "
                            "Evidence was not provided in the bounded self-check "
                            f"view: {', '.join(missing_ids)}."
                        ),
                        task_id=claim.task_id,
                        claim_id=claim.id,
                    )
                )

        summary = result.summary
        if result.status is SelfCheckStatus.PASS:
            summary = (
                f"{summary} Complete verification was not possible because "
                "some referenced Evidence was outside the bounded self-check view."
            )
        return SelfCheckResult(SelfCheckStatus.REVISE, issues, summary)

    @staticmethod
    def _validate_inputs(
        plan: ResearchPlan,
        report: StructuredReport,
        evidence_pool: dict[str, list[Evidence]],
        assessments: dict[str, list[EvidenceAssessment]],
    ) -> dict[str, EvidenceAssessment]:
        if not isinstance(plan, ResearchPlan):
            raise SelfCheckError("self-check plan must be a ResearchPlan")
        validate_research_plan(plan)
        if not isinstance(report, StructuredReport):
            raise SelfCheckError("self-check report must be a StructuredReport")
        if report.objective != plan.objective:
            raise SelfCheckError("report objective does not match the research plan")
        if not isinstance(evidence_pool, dict):
            raise SelfCheckError("evidence_pool must be task-keyed")
        if not isinstance(assessments, dict):
            raise SelfCheckError("assessments must be task-keyed")

        expected_task_ids = [task.id for task in plan.tasks]
        section_task_ids = [section.task_id for section in report.sections]
        if section_task_ids != expected_task_ids:
            raise SelfCheckError("report sections must follow the research plan")

        latest: dict[str, EvidenceAssessment] = {}
        for task in plan.tasks:
            history = assessments.get(task.id)
            if not isinstance(history, list) or not history:
                raise SelfCheckError(
                    f"task {task.id} has no latest Evidence assessment"
                )
            assessment = history[-1]
            if not isinstance(assessment, EvidenceAssessment):
                raise SelfCheckError("assessment history contains an invalid item")
            if assessment.task_id != task.id:
                raise SelfCheckError("latest assessment belongs to another task")
            latest[task.id] = assessment

        rebuilt = assemble_report(report.objective, report.sections, evidence_pool)
        if rebuilt.citations != report.citations:
            raise SelfCheckError(
                "report citations do not match trusted Evidence provenance"
            )
        return latest

    @staticmethod
    def _assessment_payload(assessment: EvidenceAssessment) -> dict[str, Any]:
        return {
            "sufficient": assessment.sufficient,
            "missing_information": list(assessment.missing_information),
            "reason": assessment.reason,
        }

    @staticmethod
    def _parse_result(
        result: Any,
        *,
        plan: ResearchPlan,
        report: StructuredReport,
    ) -> SelfCheckResult:
        if not isinstance(result, dict):
            raise SelfCheckError("self-check JSON must be an object")
        required = {"status", "issues", "summary"}
        if set(result) != required:
            raise SelfCheckError(
                "self-check JSON must contain exactly status, issues, and summary"
            )
        try:
            status = SelfCheckStatus(result["status"])
        except (TypeError, ValueError) as exc:
            raise SelfCheckError("self-check status is invalid") from exc
        summary = result["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise SelfCheckError("self-check summary must be a non-empty string")
        raw_issues = result["issues"]
        if not isinstance(raw_issues, list):
            raise SelfCheckError("self-check issues must be a list")

        task_ids = {task.id for task in plan.tasks}
        claim_by_id: dict[str, Claim] = {}
        for section in report.sections:
            for claim in section.claims:
                if claim.id in claim_by_id:
                    raise SelfCheckError("report contains duplicate Claim IDs")
                claim_by_id[claim.id] = claim

        issues: list[SelfCheckIssue] = []
        for index, raw_issue in enumerate(raw_issues, start=1):
            if not isinstance(raw_issue, dict):
                raise SelfCheckError(f"self-check issue {index} must be an object")
            if set(raw_issue) != {"code", "reason", "task_id", "claim_id"}:
                raise SelfCheckError(
                    f"self-check issue {index} has an invalid field set"
                )
            try:
                code = SelfCheckIssueCode(raw_issue["code"])
            except (TypeError, ValueError) as exc:
                raise SelfCheckError(
                    f"self-check issue {index} code is invalid"
                ) from exc
            reason = raw_issue["reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise SelfCheckError(
                    f"self-check issue {index} reason must be non-empty"
                )
            task_id = ReportSelfChecker._optional_id(
                raw_issue["task_id"], f"self-check issue {index} task_id"
            )
            claim_id = ReportSelfChecker._optional_id(
                raw_issue["claim_id"], f"self-check issue {index} claim_id"
            )
            if task_id is not None and task_id not in task_ids:
                raise SelfCheckError(
                    f"self-check issue references unknown task ID {task_id!r}"
                )
            claim = claim_by_id.get(claim_id) if claim_id is not None else None
            if claim_id is not None and claim is None:
                raise SelfCheckError(
                    f"self-check issue references unknown Claim ID {claim_id!r}"
                )
            if task_id is not None and claim is not None and claim.task_id != task_id:
                raise SelfCheckError(
                    "self-check issue Claim does not belong to its task"
                )
            issues.append(SelfCheckIssue(code, reason.strip(), task_id, claim_id))

        if status is SelfCheckStatus.PASS and issues:
            raise SelfCheckError("pass self-check result must not contain issues")
        if status is SelfCheckStatus.REVISE and not issues:
            raise SelfCheckError(
                "revise self-check result must contain at least one issue"
            )
        return SelfCheckResult(status, issues, summary.strip())

    @staticmethod
    def _optional_id(value: Any, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SelfCheckError(f"{label} must be a non-empty string or null")
        return value.strip()

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""
