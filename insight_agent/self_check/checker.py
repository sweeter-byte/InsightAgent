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
        claim_contexts: list[dict[str, Any]] = []
        for section in report.sections:
            task = task_by_id[section.task_id]
            assessment = latest[section.task_id]
            for claim in section.claims:
                claim_contexts.append(
                    {
                        "task": {"id": task.id, "question": task.question},
                        "claim": {
                            "id": claim.id,
                            "text": claim.text,
                            "evidence_ids": list(claim.evidence_ids),
                        },
                        "evidence": [
                            {
                                "evidence_id": evidence_by_id[evidence_id].id,
                                "content": evidence_by_id[evidence_id].content,
                                "retrieval_source": evidence_by_id[
                                    evidence_id
                                ].retrieval_source.value,
                            }
                            for evidence_id in claim.evidence_ids
                        ],
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
            "claim_contexts": claim_contexts,
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
        return self._parse_result(result, plan=plan, report=report)

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
