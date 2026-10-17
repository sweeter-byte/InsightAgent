"""Offline integration coverage for bounded research-context handoffs."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.evidence import (
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceGrader,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
    make_evidence_id,
)
from insight_agent.evidence.grader import (
    MAX_EVIDENCE_CONTENT_CHARS,
    MAX_EVIDENCE_ITEMS,
    MAX_TOTAL_EVIDENCE_CONTENT_CHARS,
)
from insight_agent.ingestion import Document, SourceType
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import (
    Claim,
    ReportGenerationError,
    ReportGenerator,
    ReportSection,
    StructuredReport,
    assemble_report,
)
from insight_agent.reporting.generator import (
    MAX_REPORT_EVIDENCE_CONTENT_CHARS,
    MAX_REPORT_EVIDENCE_ITEMS,
    MAX_TOTAL_REPORT_EVIDENCE_CONTENT_CHARS,
)
from insight_agent.research import ResearchRoutingWorkflow
from insight_agent.research.workflow import MAX_REPAIR_ROUNDS
from insight_agent.routing import RetrievalSource, RouteDecision
from insight_agent.self_check import (
    ReportRepairer,
    ReportSelfChecker,
    SelfCheckError,
    SelfCheckIssueCode,
    SelfCheckStatus,
)
from insight_agent.self_check.checker import (
    MAX_SELF_CHECK_EVIDENCE_CONTENT_CHARS,
    MAX_SELF_CHECK_EVIDENCE_ITEMS,
    MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS,
)
from insight_agent.self_check.repair import (
    MAX_REPAIR_EVIDENCE_CONTENT_CHARS,
    MAX_REPAIR_EVIDENCE_ITEMS,
    MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS,
)
from insight_agent.web_search import WebRetrievalResult, WebSearchHit


class ScriptedLLM:
    """Return fixed offline responses while retaining every request."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append({"messages": deepcopy(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        message = SimpleNamespace(content=self.responses.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class WebOnlyRouter:
    def route(self, **kwargs: Any) -> RouteDecision:
        task = kwargs["task"]
        return RouteDecision(task.id, RetrievalSource.WEB, "Offline web fixture")


class LongWebRetriever:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[str] = []

    def retrieve(self, query: str) -> WebRetrievalResult:
        self.calls.append(query)
        hits: list[WebSearchHit] = []
        documents: list[Document] = []
        for index, content in enumerate(self.contents, start=1):
            url = f"https://example.test/source-{index}"
            hits.append(WebSearchHit(index, f"Source {index}", url, "fixture"))
            documents.append(
                Document(
                    content=content,
                    source=url,
                    source_type=SourceType.URL,
                    metadata={
                        "final_url": url,
                        "search_rank": index,
                        "title": f"Source {index}",
                    },
                )
            )
        return WebRetrievalResult(query, hits, documents, [])


class NeverRepairer:
    def repair(self, **kwargs: Any) -> StructuredReport:
        raise AssertionError("a passing report must not enter repair")


class NeverCalled:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected dependency call: {name}")


def _user_payload(llm: ScriptedLLM, call_index: int = 0) -> dict[str, Any]:
    return json.loads(llm.calls[call_index]["messages"][1]["content"])


def _assert_bounded_contents(
    items: list[dict[str, Any]],
    *,
    max_items: int,
    max_chars_per_item: int,
    max_total_chars: int,
) -> None:
    assert len(items) <= max_items
    assert all(len(item["content"]) <= max_chars_per_item for item in items)
    assert sum(len(item["content"]) for item in items) <= max_total_chars


def test_long_web_evidence_completes_bounded_research_workflow() -> None:
    task = ResearchTask(id="T1", question="What do the sources establish?")
    plan = ResearchPlan(
        objective="Summarize the offline source set",
        constraints=["Use only collected Evidence"],
        tasks=[task],
    )
    contents = [
        f"HEAD-{index}:" + (str(index % 10) * 7000) + f":TAIL-{index}"
        for index in range(1, 13)
    ]
    urls = [f"https://example.test/source-{index}" for index in range(1, 13)]
    evidence_ids = [
        make_evidence_id("T1", RetrievalSource.WEB, url) for url in urls
    ]
    provided_ids = evidence_ids[:MAX_EVIDENCE_ITEMS]

    grader_llm = ScriptedLLM(
        json.dumps(
            {
                "task_id": "T1",
                "evidence_judgments": [
                    {
                        "evidence_id": evidence_id,
                        "relevance": "relevant",
                        "quality": "strong",
                        "reason": "Direct offline source.",
                    }
                    for evidence_id in provided_ids
                ],
                "coverage": "complete",
                "sufficient": True,
                "missing_information": [],
                "reason": "The bounded source view covers the task.",
            }
        )
    )
    report_llm = ScriptedLLM(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "The offline sources establish the requested result.",
                        "evidence_ids": [provided_ids[0], provided_ids[-1]],
                    }
                ]
            }
        )
    )
    checker_llm = ScriptedLLM(
        json.dumps(
            {
                "status": "pass",
                "issues": [],
                "summary": "Every Claim is supported by supplied Evidence.",
            }
        )
    )
    retriever = LongWebRetriever(contents)
    workflow = ResearchRoutingWorkflow(
        router=WebOnlyRouter(),  # type: ignore[arg-type]
        web_retriever=retriever,
        grader=EvidenceGrader(grader_llm),  # type: ignore[arg-type]
        report_generator=ReportGenerator(report_llm),  # type: ignore[arg-type]
        self_checker=ReportSelfChecker(checker_llm),  # type: ignore[arg-type]
        report_repairer=NeverRepairer(),  # type: ignore[arg-type]
        max_retrieval_rounds=1,
    )

    final_state = workflow.run(
        ResearchState(
            query=plan.objective,
            plan=plan,
            available_sources={RetrievalSource.WEB},
        )
    )

    assert retriever.calls == [task.question]
    assert final_state.final_output is not None
    assert final_state.self_check_result is not None
    assert final_state.self_check_result.status is SelfCheckStatus.PASS
    assert final_state.self_check_rounds == 0

    pool = final_state.evidence_pool["T1"]
    assert [item.content for item in pool] == contents
    assert [item.id for item in pool] == evidence_ids

    grader_payload = _user_payload(grader_llm)
    grader_view = grader_payload["evidence"]
    _assert_bounded_contents(
        grader_view,
        max_items=MAX_EVIDENCE_ITEMS,
        max_chars_per_item=MAX_EVIDENCE_CONTENT_CHARS,
        max_total_chars=MAX_TOTAL_EVIDENCE_CONTENT_CHARS,
    )
    assert grader_payload["evidence_view"]["pool_evidence_count"] == 12
    assert grader_payload["evidence_view"]["provided_evidence_count"] == 10

    report_payload = _user_payload(report_llm)
    report_view = report_payload["evidence"]
    _assert_bounded_contents(
        report_view,
        max_items=MAX_REPORT_EVIDENCE_ITEMS,
        max_chars_per_item=MAX_REPORT_EVIDENCE_CONTENT_CHARS,
        max_total_chars=MAX_TOTAL_REPORT_EVIDENCE_CONTENT_CHARS,
    )
    assert report_payload["evidence_view"]["candidate_evidence_count"] == 10

    checker_payload = _user_payload(checker_llm)
    checker_view = list(checker_payload["evidence_catalog"].values())
    _assert_bounded_contents(
        checker_view,
        max_items=MAX_SELF_CHECK_EVIDENCE_ITEMS,
        max_chars_per_item=MAX_SELF_CHECK_EVIDENCE_CONTENT_CHARS,
        max_total_chars=MAX_TOTAL_SELF_CHECK_EVIDENCE_CONTENT_CHARS,
    )
    for llm in (grader_llm, report_llm, checker_llm):
        serialized_request = llm.calls[0]["messages"][1]["content"]
        assert all(content not in serialized_request for content in contents)

    assert final_state.report is not None
    claimed_ids = {
        evidence_id
        for section in final_state.report.sections
        for claim in section.claims
        for evidence_id in claim.evidence_ids
    }
    pool_by_id = {item.id: item for item in pool}
    assert claimed_ids == {provided_ids[0], provided_ids[-1]}
    assert claimed_ids <= pool_by_id.keys()
    assert {citation.evidence_id for citation in final_state.report.citations} == (
        claimed_ids
    )
    assert all(
        citation.source == pool_by_id[citation.evidence_id].source
        for citation in final_state.report.citations
    )


def _over_capacity_research_state() -> ResearchState:
    task = ResearchTask(id="T1", question="Which claims are supported?")
    plan = ResearchPlan(
        objective="Audit an over-capacity report",
        constraints=["Use only collected Evidence"],
        tasks=[task],
    )
    evidence = [
        Evidence(
            id=f"E{index}",
            task_id="T1",
            retrieval_source=RetrievalSource.LOCAL,
            origin_id=f"origin-E{index}",
            content=f"Evidence body for claim {index}.",
            source=f"source-{index}.md",
            metadata={"filename": f"source-{index}.md"},
        )
        for index in range(1, MAX_SELF_CHECK_EVIDENCE_ITEMS + 2)
    ]
    assessment = EvidenceAssessment(
        task_id="T1",
        evidence_judgments=[
            EvidenceJudgment(
                evidence_id=item.id,
                relevance=EvidenceRelevance.RELEVANT,
                quality=EvidenceQuality.STRONG,
                reason="Usable audit fixture.",
            )
            for item in evidence
        ],
        coverage=EvidenceCoverage.COMPLETE,
        sufficient=True,
        missing_information=[],
        reason="The full pool covers the task.",
    )
    report = assemble_report(
        plan.objective,
        [
            ReportSection(
                task_id="T1",
                title=task.question,
                claims=[
                    Claim(
                        id=f"T1-C{index}",
                        task_id="T1",
                        text=f"Claim {index}.",
                        evidence_ids=[f"E{index}"],
                    )
                    for index in range(1, MAX_SELF_CHECK_EVIDENCE_ITEMS + 2)
                ],
                sufficient=True,
                missing_information=[],
            )
        ],
        {"T1": evidence},
    )
    return ResearchState(
        query=plan.objective,
        plan=plan,
        available_sources={RetrievalSource.LOCAL},
        task_index=1,
        evidence_pool={"T1": evidence},
        evidence_assessments={"T1": [assessment]},
        report=report,
    )


def _unchanged_repair_payload(report: StructuredReport) -> str:
    return json.dumps(
        {
            "objective": report.objective,
            "sections": [
                {
                    "task_id": section.task_id,
                    "title": section.title,
                    "claims": [
                        {
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
        }
    )


def test_unprovided_evidence_cannot_false_pass_or_escape_one_repair_limit() -> None:
    state = _over_capacity_research_state()
    assert state.plan is not None
    assert state.report is not None
    pass_response = json.dumps(
        {"status": "pass", "issues": [], "summary": "All Claims pass."}
    )
    checker_llm = ScriptedLLM(pass_response, pass_response)
    checker = ReportSelfChecker(checker_llm)  # type: ignore[arg-type]

    first_result = checker.check(
        plan=state.plan,
        report=state.report,
        evidence_pool=state.evidence_pool,
        assessments=state.evidence_assessments,
    )

    assert first_result.status is SelfCheckStatus.REVISE
    assert [issue.claim_id for issue in first_result.issues] == ["T1-C11"]
    assert first_result.issues[0].code is SelfCheckIssueCode.UNSUPPORTED_CLAIM
    assert "E11" in first_result.issues[0].reason
    first_check_payload = _user_payload(checker_llm)
    assert first_check_payload["claim_contexts"][-1][
        "unprovided_evidence_ids"
    ] == ["E11"]

    malicious_payload = json.loads(_unchanged_repair_payload(state.report))
    malicious_payload["sections"][0]["claims"][9] = {
        "text": "Rewritten claim using omitted Evidence.",
        "evidence_ids": ["E10"],
    }
    malicious_llm = ScriptedLLM(json.dumps(malicious_payload))
    with pytest.raises(
        ReportGenerationError,
        match="outside the provided repair Evidence view",
    ):
        ReportRepairer(malicious_llm).repair(  # type: ignore[arg-type]
            plan=state.plan,
            report=state.report,
            self_check_result=first_result,
            evidence_pool=state.evidence_pool,
            assessments=state.evidence_assessments,
        )
    malicious_view = _user_payload(malicious_llm)["evidence_catalog"]
    assert "E11" in malicious_view
    assert "E10" not in malicious_view

    added_claim_payload = json.loads(_unchanged_repair_payload(state.report))
    added_claim_payload["sections"][0]["claims"].append(
        {
            "text": "New claim using omitted Evidence.",
            "evidence_ids": ["E10"],
        }
    )
    added_claim_llm = ScriptedLLM(json.dumps(added_claim_payload))
    with pytest.raises(
        ReportGenerationError,
        match="outside the provided repair Evidence view",
    ):
        ReportRepairer(added_claim_llm).repair(  # type: ignore[arg-type]
            plan=state.plan,
            report=state.report,
            self_check_result=first_result,
            evidence_pool=state.evidence_pool,
            assessments=state.evidence_assessments,
        )
    assert "E10" not in _user_payload(added_claim_llm)["evidence_catalog"]

    repair_llm = ScriptedLLM(_unchanged_repair_payload(state.report))
    workflow = ResearchRoutingWorkflow(
        router=NeverCalled(),  # type: ignore[arg-type]
        local_retriever=NeverCalled(),  # type: ignore[arg-type]
        grader=NeverCalled(),  # type: ignore[arg-type]
        report_generator=NeverCalled(),  # type: ignore[arg-type]
        self_checker=checker,
        report_repairer=ReportRepairer(repair_llm),  # type: ignore[arg-type]
        max_retrieval_rounds=1,
    )
    state.self_check_result = first_result
    repair_update = workflow.repair_report(state)
    state.report = repair_update["report"]
    state.self_check_rounds = repair_update["self_check_rounds"]

    assert state.self_check_rounds == MAX_REPAIR_ROUNDS == 1
    assert len(repair_llm.calls) == 1
    assert state.report.sections[0].claims[-1].text == "Claim 11."
    repair_view = list(_user_payload(repair_llm)["evidence_catalog"].values())
    _assert_bounded_contents(
        repair_view,
        max_items=MAX_REPAIR_EVIDENCE_ITEMS,
        max_chars_per_item=MAX_REPAIR_EVIDENCE_CONTENT_CHARS,
        max_total_chars=MAX_TOTAL_REPAIR_EVIDENCE_CONTENT_CHARS,
    )

    second_result = workflow.self_check_report(state)["self_check_result"]
    state.self_check_result = second_result
    assert second_result.status is SelfCheckStatus.REVISE
    assert [issue.claim_id for issue in second_result.issues] == ["T1-C11"]
    assert workflow.choose_after_self_check(state) == "failed"
    with pytest.raises(SelfCheckError, match="failed after 1 repair"):
        workflow.fail_self_check(state)
    assert len(repair_llm.calls) == 1
