"""Controlled report repair using only existing research state."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from insight_agent.evidence import Evidence, EvidenceAssessment
from insight_agent.llm import LLMClient
from insight_agent.planning import ResearchPlan
from insight_agent.reporting import (
    Claim,
    ReportGenerationError,
    ReportSection,
    StructuredReport,
    assemble_report,
    select_report_candidates,
    validate_claims,
)
from insight_agent.self_check.checker import ReportSelfChecker
from insight_agent.self_check.models import (
    SelfCheckError,
    SelfCheckIssue,
    SelfCheckIssueCode,
    SelfCheckResult,
    SelfCheckStatus,
)
from insight_agent.self_check.prompts import REPAIR_SYSTEM_PROMPT


MAX_REPAIR_EVIDENCE_ITEMS = 10
MAX_REPAIR_EVIDENCE_CONTENT_CHARS = 6000
MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS = 40000
REPAIR_CONTENT_OMISSION_MARKER = (
    "\n\n[... CONTENT OMITTED FROM REPORT REPAIR VIEW ...]\n\n"
)


class ReportRepairer:
    """Repair report expression and bindings without changing research state."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def repair(
        self,
        *,
        plan: ResearchPlan,
        report: StructuredReport,
        self_check_result: SelfCheckResult,
        evidence_pool: dict[str, list[Evidence]],
        assessments: dict[str, list[EvidenceAssessment]],
    ) -> StructuredReport:
        latest = ReportSelfChecker._validate_inputs(
            plan,
            report,
            evidence_pool,
            assessments,
        )
        self._validate_result(self_check_result, plan=plan, report=report)

        candidates_by_task = {
            task.id: select_report_candidates(
                task,
                list(evidence_pool.get(task.id, [])),
                latest[task.id],
            )
            for task in plan.tasks
        }
        selected_evidence = self._select_evidence(
            plan=plan,
            report=report,
            self_check_result=self_check_result,
            candidates_by_task=candidates_by_task,
        )
        content_budgets = self._allocate_content_budgets(selected_evidence)
        provided_ids_by_task = {
            task.id: [
                evidence.id
                for evidence in selected_evidence
                if evidence.task_id == task.id
            ]
            for task in plan.tasks
        }
        payload = {
            "current_report": {
                "objective": report.objective,
                "sections": [self._section_payload(section) for section in report.sections],
            },
            "self_check": {
                "status": self_check_result.status.value,
                "issues": [self._issue_payload(issue) for issue in self_check_result.issues],
                "summary": self_check_result.summary,
            },
            "task_contexts": [
                {
                    "task": {"id": task.id, "question": task.question},
                    "latest_assessment": {
                        "sufficient": latest[task.id].sufficient,
                        "missing_information": list(
                            latest[task.id].missing_information
                        ),
                        "reason": latest[task.id].reason,
                    },
                    "candidate_evidence_count": len(candidates_by_task[task.id]),
                    "provided_evidence_ids": provided_ids_by_task[task.id],
                }
                for task in plan.tasks
            ],
            "evidence_view": {
                "bounded": True,
                "selection_rule": (
                    "unique issue-Claim references, then other report references, "
                    "then issue-Task candidates, then remaining candidates"
                ),
                "candidate_evidence_count": sum(
                    len(items) for items in candidates_by_task.values()
                ),
                "provided_evidence_count": len(selected_evidence),
                "max_evidence_items": MAX_REPAIR_EVIDENCE_ITEMS,
                "max_content_chars_per_evidence": (
                    MAX_REPAIR_EVIDENCE_CONTENT_CHARS
                ),
                "max_total_content_chars": (
                    MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS
                ),
            },
            "evidence_catalog": {
                evidence.id: {
                    "evidence_id": evidence.id,
                    "task_id": evidence.task_id,
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
                "repair input must contain JSON-serializable values"
            ) from exc
        response = self.llm.chat(
            [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": serialized},
            ],
            tools=None,
        )
        raw_text = self._extract_text(response)
        if not raw_text.strip():
            raise SelfCheckError("repair response did not contain JSON")
        try:
            candidate = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise SelfCheckError(f"repair returned invalid JSON: {exc.msg}") from exc
        return self._parse_report(
            candidate,
            plan=plan,
            evidence_pool=evidence_pool,
            latest=latest,
            candidates_by_task=candidates_by_task,
            provided_ids_by_task={
                task_id: set(evidence_ids)
                for task_id, evidence_ids in provided_ids_by_task.items()
            },
            original_report=report,
        )

    @staticmethod
    def _select_evidence(
        *,
        plan: ResearchPlan,
        report: StructuredReport,
        self_check_result: SelfCheckResult,
        candidates_by_task: dict[str, list[Evidence]],
    ) -> list[Evidence]:
        """Select a deterministic, globally deduplicated repair Evidence view."""
        candidate_by_id = {
            evidence.id: evidence
            for task in plan.tasks
            for evidence in candidates_by_task[task.id]
        }
        claim_by_id = {
            claim.id: claim
            for section in report.sections
            for claim in section.claims
        }
        selected_ids: list[str] = []
        seen_ids: set[str] = set()

        def add(evidence_id: str) -> None:
            if evidence_id in candidate_by_id and evidence_id not in seen_ids:
                seen_ids.add(evidence_id)
                selected_ids.append(evidence_id)

        issue_task_ids: list[str] = []
        seen_issue_task_ids: set[str] = set()
        for issue in self_check_result.issues:
            claim = claim_by_id.get(issue.claim_id) if issue.claim_id else None
            if claim is not None:
                for evidence_id in claim.evidence_ids:
                    add(evidence_id)
            task_id = issue.task_id or (claim.task_id if claim is not None else None)
            if task_id is not None and task_id not in seen_issue_task_ids:
                seen_issue_task_ids.add(task_id)
                issue_task_ids.append(task_id)

        for section in report.sections:
            for claim in section.claims:
                for evidence_id in claim.evidence_ids:
                    add(evidence_id)

        for task_id in issue_task_ids:
            for evidence in candidates_by_task[task_id]:
                add(evidence.id)

        for task in plan.tasks:
            for evidence in candidates_by_task[task.id]:
                add(evidence.id)

        return [
            candidate_by_id[evidence_id]
            for evidence_id in selected_ids[:MAX_REPAIR_EVIDENCE_ITEMS]
        ]

    @staticmethod
    def _allocate_content_budgets(evidence: list[Evidence]) -> list[int]:
        """Allocate the total repair body budget without truncating short items."""
        target_sizes = [
            min(len(item.content), MAX_REPAIR_EVIDENCE_CONTENT_CHARS)
            for item in evidence
        ]
        if sum(target_sizes) <= MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS:
            return target_sizes

        budgets = [0] * len(target_sizes)
        remaining_budget = MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS
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
        marker_size = len(REPAIR_CONTENT_OMISSION_MARKER)
        if limit <= marker_size:
            return REPAIR_CONTENT_OMISSION_MARKER[:limit]
        retained_chars = limit - marker_size
        head_chars = (retained_chars + 1) // 2
        tail_chars = retained_chars // 2
        return (
            content[:head_chars]
            + REPAIR_CONTENT_OMISSION_MARKER
            + content[-tail_chars:]
        )

    @staticmethod
    def _parse_report(
        candidate: Any,
        *,
        plan: ResearchPlan,
        evidence_pool: dict[str, list[Evidence]],
        latest: dict[str, EvidenceAssessment],
        candidates_by_task: dict[str, list[Evidence]],
        provided_ids_by_task: dict[str, set[str]],
        original_report: StructuredReport,
    ) -> StructuredReport:
        if not isinstance(candidate, dict) or set(candidate) != {
            "objective",
            "sections",
        }:
            raise SelfCheckError(
                "repair JSON must contain exactly objective and sections"
            )
        if candidate["objective"] != plan.objective:
            raise SelfCheckError("repair must not change the report objective")
        raw_sections = candidate["sections"]
        if not isinstance(raw_sections, list) or len(raw_sections) != len(plan.tasks):
            raise SelfCheckError("repair must preserve every plan-ordered section")

        sections: list[ReportSection] = []
        original_claim_signatures_by_task = {
            section.task_id: Counter(
                (claim.text.strip(), tuple(claim.evidence_ids))
                for claim in section.claims
            )
            for section in original_report.sections
        }
        for section_index, (task, raw_section) in enumerate(
            zip(plan.tasks, raw_sections, strict=True),
            start=1,
        ):
            if not isinstance(raw_section, dict) or set(raw_section) != {
                "task_id",
                "title",
                "claims",
                "sufficient",
                "missing_information",
            }:
                raise SelfCheckError(
                    f"repair section {section_index} has an invalid field set"
                )
            if raw_section["task_id"] != task.id:
                raise SelfCheckError("repair must preserve plan task order and IDs")
            if raw_section["title"] != task.question:
                raise SelfCheckError("repair must preserve section titles")
            assessment = latest[task.id]
            if (
                not isinstance(raw_section["sufficient"], bool)
                or raw_section["sufficient"] is not assessment.sufficient
            ):
                raise SelfCheckError("repair must not change assessment sufficiency")
            missing_information = raw_section["missing_information"]
            if not isinstance(missing_information, list) or any(
                not isinstance(item, str) or not item.strip()
                for item in missing_information
            ):
                raise SelfCheckError(
                    "repair missing_information must be a list of strings"
                )
            if missing_information != assessment.missing_information:
                raise SelfCheckError(
                    "repair must preserve latest assessment missing_information"
                )

            raw_claims = raw_section["claims"]
            if not isinstance(raw_claims, list):
                raise SelfCheckError("repair section claims must be a list")
            claims: list[Claim] = []
            for claim_index, raw_claim in enumerate(raw_claims, start=1):
                if not isinstance(raw_claim, dict) or set(raw_claim) != {
                    "text",
                    "evidence_ids",
                }:
                    raise SelfCheckError(
                        f"repair Claim {claim_index} has an invalid field set"
                    )
                text = raw_claim["text"]
                evidence_ids = raw_claim["evidence_ids"]
                if not isinstance(text, str) or not text.strip():
                    raise SelfCheckError("repair Claim text must be non-empty")
                if not isinstance(evidence_ids, list):
                    raise SelfCheckError("repair Claim evidence_ids must be a list")
                claims.append(
                    Claim(
                        id=f"{task.id}-C{claim_index}",
                        task_id=task.id,
                        text=text.strip(),
                        evidence_ids=list(evidence_ids),
                    )
                )
            validate_claims(task, candidates_by_task[task.id], claims)
            unchanged_signatures = original_claim_signatures_by_task[task.id]
            for claim in claims:
                signature = (claim.text, tuple(claim.evidence_ids))
                if unchanged_signatures[signature] > 0:
                    unchanged_signatures[signature] -= 1
                    continue
                if any(
                    evidence_id not in provided_ids_by_task[task.id]
                    for evidence_id in claim.evidence_ids
                ):
                    raise ReportGenerationError(
                        "new or modified Claim references Evidence outside the "
                        "provided repair Evidence view"
                    )
            sections.append(
                ReportSection(
                    task_id=task.id,
                    title=task.question,
                    claims=claims,
                    sufficient=assessment.sufficient,
                    missing_information=list(assessment.missing_information),
                )
            )
        return assemble_report(plan.objective, sections, evidence_pool)

    @staticmethod
    def _validate_result(
        result: SelfCheckResult,
        *,
        plan: ResearchPlan,
        report: StructuredReport,
    ) -> None:
        if not isinstance(result, SelfCheckResult):
            raise SelfCheckError("repair requires a SelfCheckResult")
        if result.status is not SelfCheckStatus.REVISE or not result.issues:
            raise SelfCheckError("repair requires a revise result with issues")
        if not isinstance(result.summary, str) or not result.summary.strip():
            raise SelfCheckError("repair self-check summary must be non-empty")
        for issue in result.issues:
            if (
                not isinstance(issue, SelfCheckIssue)
                or not isinstance(issue.code, SelfCheckIssueCode)
                or not isinstance(issue.reason, str)
                or not issue.reason.strip()
            ):
                raise SelfCheckError("repair received an invalid self-check issue")
        raw_result = {
            "status": result.status.value,
            "issues": [ReportRepairer._issue_payload(issue) for issue in result.issues],
            "summary": result.summary,
        }
        ReportSelfChecker._parse_result(raw_result, plan=plan, report=report)

    @staticmethod
    def _section_payload(section: ReportSection) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _issue_payload(issue: SelfCheckIssue) -> dict[str, Any]:
        return {
            "code": issue.code.value,
            "reason": issue.reason,
            "task_id": issue.task_id,
            "claim_id": issue.claim_id,
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""
