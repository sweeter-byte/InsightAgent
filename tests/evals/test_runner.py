from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from evals import (
    AnswerEvalResult,
    CaseExecutionStatus,
    EvalCase,
    EvalMode,
    EvalRunMetadata,
    EvaluationResult,
    EvaluationRunner,
    EvaluationType,
    EvidenceEvalResult,
    JudgeEvaluation,
    JudgeExecutionStatus,
    JudgeLabel,
    JudgeResult,
    MetricStatus,
    OfflineDependencyError,
    RetrievalEvalResult,
    RuntimeMetrics,
    WorkflowCaseOutput,
    offline_network_guard,
)
from insight_agent.evidence import Evidence
from insight_agent.planning import ResearchPlan, ResearchState, ResearchTask
from insight_agent.reporting import StructuredReport
from insight_agent.routing import RetrievalSource


def _metadata(run_id: str = "eval-test") -> EvalRunMetadata:
    return EvalRunMetadata(
        eval_run_id=run_id,
        dataset_version="dataset-v1",
        dataset_sha256="a" * 64,
        git_commit="abc123",
        model_name="fixture-model",
        prompt_version="judge-v1",
        index_version="index-v1",
        retriever_configuration={"top_k": 5},
        mode=EvalMode.OFFLINE,
        evaluation_type=EvaluationType.MOCK_TEST,
        config_fingerprint="fingerprint",
    )


class FakeRetrievalEvaluator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate(self, case: EvalCase, *, top_k: int = 5) -> RetrievalEvalResult:
        self.calls.append(case.case_id)
        return RetrievalEvalResult(
            case_id=case.case_id,
            retrieved_ids=("gold-1",),
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            metric_status=MetricStatus.COMPUTED,
        )


class FakeWorkflowRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, case: EvalCase) -> WorkflowCaseOutput:
        self.calls.append(case.case_id)
        if case.case_id == "broken/research":
            raise RuntimeError("fixture workflow exploded")
        evidence = Evidence(
            id="E1",
            task_id="T1",
            retrieval_source=RetrievalSource.LOCAL,
            origin_id="gold-1",
            content="fixed evidence",
            source="fixture.md",
        )
        report = StructuredReport(objective=case.query)
        state = ResearchState(
            query=case.query,
            plan=ResearchPlan(
                objective=case.query,
                tasks=[ResearchTask("T1", case.query)],
            ),
            available_sources={RetrievalSource.LOCAL},
            evidence_pool={"T1": [evidence]},
            report=report,
            final_output="fixed output",
        )
        return WorkflowCaseOutput(
            state=state,
            report=report,
            evidence_by_id={"E1": evidence},
            runtime=RuntimeMetrics(True, 7, None, None, 1, 0),
            artifacts={"final_output": "fixed output", "raw_marker": case.case_id},
        )


class FakeEvidenceEvaluator:
    def evaluate(
        self,
        case: EvalCase,
        evidence: list[Evidence],
    ) -> EvidenceEvalResult:
        return EvidenceEvalResult(
            case_id=case.case_id,
            provenance_valid_rate=1.0,
            evidence_count=len(evidence),
            relevance=JudgeEvaluation(
                JudgeExecutionStatus.TIMEOUT,
                None,
                "judge deadline",
            ),
            coverage=JudgeEvaluation(
                JudgeExecutionStatus.NOT_COMPUTABLE,
                None,
                "no required points",
            ),
        )


class FakeAnswerEvaluator:
    def evaluate(
        self,
        case: EvalCase,
        report: StructuredReport,
        evidence_by_id: dict[str, Evidence],
    ) -> AnswerEvalResult:
        failed = JudgeEvaluation(
            JudgeExecutionStatus.COMPLETED,
            JudgeResult(JudgeLabel.FAIL, "fixed constraint was violated"),
        )
        unavailable = JudgeEvaluation(
            JudgeExecutionStatus.NOT_COMPUTABLE,
            None,
            "no Gold Label",
        )
        return AnswerEvalResult(
            case_id=case.case_id,
            total_claims=0,
            claim_results=(),
            grounded_claims=None,
            coverage=unavailable,
            constraints=failed,
            structural_errors=("missing_section:T1",),
        )


def test_runner_isolates_cases_and_writes_raw_results(tmp_path: Path) -> None:
    retrieval = FakeRetrievalEvaluator()
    workflow = FakeWorkflowRunner()
    runner = EvaluationRunner(
        metadata=_metadata(),
        output_root=tmp_path,
        retrieval_evaluator=retrieval,
        workflow_runner=workflow,
        evidence_evaluator=FakeEvidenceEvaluator(),
        answer_evaluator=FakeAnswerEvaluator(),
    )
    cases = [
        EvalCase(
            case_id="retrieval-only",
            query="find chunk",
            tags=("retrieval",),
            gold_retrieval_ids=("gold-1",),
        ),
        EvalCase(
            case_id="research case",
            query="write answer",
            tags=("research",),
            constraints=("fixed constraint",),
        ),
        EvalCase(
            case_id="broken/research",
            query="explode",
            tags=("research",),
        ),
    ]

    summary = runner.run(cases)

    assert retrieval.calls == ["retrieval-only"]
    assert workflow.calls == ["research case", "broken/research"]
    assert [result.execution_status for result in summary.results] == [
        CaseExecutionStatus.COMPLETED,
        CaseExecutionStatus.COMPLETED,
        CaseExecutionStatus.FAILED,
    ]
    assert summary.results[1].answer is not None
    assert summary.results[1].answer.structural_errors == ("missing_section:T1",)
    assert summary.results[1].evidence is not None
    assert summary.results[1].evidence.relevance.status is JudgeExecutionStatus.TIMEOUT
    assert summary.results[2].error == "RuntimeError: fixture workflow exploded"
    assert summary.aggregates["cases"] == {
        "total": 3,
        "completed": 2,
        "failed": 1,
    }
    assert summary.aggregates["retrieval"]["computed_cases"] == 1
    assert summary.aggregates["judge"]["semantic_labels"] == {
        "pass": 0,
        "partial": 0,
        "fail": 1,
    }
    assert summary.aggregates["judge"]["execution_statuses"]["timeout"] == 1
    assert (
        summary.aggregates["judge"]["execution_statuses"]["not_computable"]
        == 2
    )
    assert summary.aggregates["evidence"]["relevance_pass_rate"] is None
    assert summary.aggregates["evidence"]["coverage_pass_rate"] is None
    assert summary.aggregates["answer"]["constraint_pass_rate"] == 0.0
    assert summary.aggregates["answer"]["constraint_denominator"] == 1
    assert summary.aggregates["runtime"]["completion_rate"] == pytest.approx(
        2 / 3
    )
    assert summary.aggregates["runtime"]["total_prompt_tokens"] is None

    run_dir = tmp_path / "eval-test"
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "evaluation_run.v1"
    assert manifest["metadata"]["evaluation_type"] == "mock_test"
    assert len(manifest["cases"]) == 3
    for result in summary.results:
        result_path = run_dir / result.artifact_paths["result"]
        artifact_path = run_dir / result.artifact_paths["intermediates"]
        assert result_path.is_file()
        assert artifact_path.is_file()
        assert run_dir in result_path.parents
    stored_research = json.loads(
        (
            run_dir
            / summary.results[1].artifact_paths["intermediates"]
        ).read_text(encoding="utf-8")
    )
    assert stored_research["raw_marker"] == "research case"
    bad_cases = json.loads(
        (run_dir / "bad_cases.json").read_text(encoding="utf-8")
    )
    assert bad_cases["format_version"] == "evaluation_bad_cases.v1"
    assert {
        item["category"] for item in bad_cases["bad_cases"]
    } == {"constraint_violation", "citation_error", "runtime_failure"}


def test_runner_filters_requested_case_ids(tmp_path: Path) -> None:
    runner = EvaluationRunner(
        metadata=_metadata("filtered"),
        output_root=tmp_path,
        retrieval_evaluator=FakeRetrievalEvaluator(),
    )

    summary = runner.run(
        [
            EvalCase("one", "q1", tags=("retrieval",)),
            EvalCase("two", "q2", tags=("retrieval",)),
        ],
        case_ids={"two"},
    )

    assert [result.case.case_id for result in summary.results] == ["two"]


def test_runner_rejects_duplicate_case_ids_before_writing(tmp_path: Path) -> None:
    runner = EvaluationRunner(
        metadata=_metadata("duplicate-case-run"),
        output_root=tmp_path,
        retrieval_evaluator=FakeRetrievalEvaluator(),
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        runner.run(
            [
                EvalCase("duplicate", "q1", tags=("retrieval",)),
                EvalCase("duplicate", "q2", tags=("retrieval",)),
            ]
        )

    assert not (tmp_path / "duplicate-case-run").exists()


def test_runner_refuses_to_overwrite_existing_run_directory(tmp_path: Path) -> None:
    runner = EvaluationRunner(
        metadata=_metadata("stable-run-id"),
        output_root=tmp_path,
        retrieval_evaluator=FakeRetrievalEvaluator(),
    )
    cases = [EvalCase("one", "q1", tags=("retrieval",))]
    first = runner.run(cases)
    original_manifest = (first.run_directory / "run.json").read_bytes()

    with pytest.raises(FileExistsError):
        runner.run(cases)

    assert (first.run_directory / "run.json").read_bytes() == original_manifest


def test_offline_network_guard_blocks_socket_connections() -> None:
    with offline_network_guard():
        with pytest.raises(OfflineDependencyError, match="Offline Evaluation"):
            socket.create_connection(("127.0.0.1", 9))


def test_evaluation_result_defaults_remain_serializable() -> None:
    result = EvaluationResult(
        metadata=_metadata(),
        case=EvalCase("case", "query"),
        retrieval=None,
        evidence=None,
        answer=None,
        runtime=None,
    )

    assert result.execution_status is CaseExecutionStatus.COMPLETED
