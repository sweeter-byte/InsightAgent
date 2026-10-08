from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from evals import (
    AnswerEvaluator,
    EvalMode,
    EvaluationRunner,
    EvaluationType,
    EvidenceEvaluator,
    JudgeExecutionStatus,
    RetrievalEvaluator,
    build_run_metadata,
    BaselineKind,
    RegressionStatus,
    compare_regression,
    create_baseline,
    load_evaluation_run,
    load_regression_policy,
    write_regression_report,
    load_eval_cases,
)
from evals.offline import load_offline_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "eval_data" / "research_eval_smoke_v1.jsonl"
FIXTURE = PROJECT_ROOT / "evals" / "fixtures" / "smoke_v1.json"
POLICY = PROJECT_ROOT / "evals" / "policies" / "default_v1.json"


def _run(tmp_path: Path, run_id: str):
    fixture = load_offline_fixture(FIXTURE, project_root=PROJECT_ROOT)
    metadata = build_run_metadata(
        dataset_path=DATASET,
        dataset_version="research-eval-smoke-v1",
        repo_root=PROJECT_ROOT,
        model_name="recorded-fixture",
        prompt_version="fixture-judge-v1",
        index_version=fixture.index_version,
        retriever_configuration={
            "top_k": 5,
            "fixture_version": fixture.fixture_version,
        },
        mode=EvalMode.OFFLINE,
        evaluation_type=EvaluationType.MOCK_TEST,
        eval_run_id=run_id,
    )
    runner = EvaluationRunner(
        metadata=metadata,
        output_root=tmp_path,
        retrieval_evaluator=RetrievalEvaluator(fixture.retriever),
        workflow_runner=fixture.workflow_runner,
        evidence_evaluator=EvidenceEvaluator(fixture.judge),
        answer_evaluator=AnswerEvaluator(fixture.judge),
    )
    return runner.run(load_eval_cases(DATASET))


def _deterministic_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"].pop("eval_run_id")
    payload["runtime"].pop("latency_ms")
    return payload


def test_fixed_dataset_completes_twice_with_deterministic_quality_results(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, "offline-a")
    second = _run(tmp_path, "offline-b")

    assert first.metadata.config_fingerprint == second.metadata.config_fingerprint
    assert first.aggregates == second.aggregates
    assert len(first.results) == 3
    for first_result, second_result in zip(first.results, second.results, strict=True):
        first_path = first.run_directory / first_result.artifact_paths["result"]
        second_path = second.run_directory / second_result.artifact_paths["result"]
        assert _deterministic_result(first_path) == _deterministic_result(second_path)

    web = next(result for result in first.results if result.case.case_id == "recorded_web_research")
    web_artifacts = json.loads(
        (first.run_directory / web.artifact_paths["intermediates"]).read_text(
            encoding="utf-8"
        )
    )
    assert web_artifacts["web_result"]["documents"][0]["content"].startswith(
        "Offline web evaluation"
    )

    vision = next(result for result in first.results if result.case.case_id == "fixed_vision_research")
    assert vision.answer is not None
    assert vision.answer.coverage.status is JudgeExecutionStatus.NOT_COMPUTABLE
    assert vision.answer.constraints.status is JudgeExecutionStatus.NOT_COMPUTABLE
    vision_artifacts = json.loads(
        (first.run_directory / vision.artifact_paths["intermediates"]).read_text(
            encoding="utf-8"
        )
    )
    image_path = Path(vision_artifacts["vision_result"]["analyses"][0]["source"])
    assert image_path == (PROJECT_ROOT / "examples" / "vision_workflow.png").resolve()
    assert image_path.is_file()


def test_smoke_dataset_labels_point_to_fixed_versioned_sources() -> None:
    cases = {case.case_id: case for case in load_eval_cases(DATASET)}
    fixture = load_offline_fixture(FIXTURE, project_root=PROJECT_ROOT)

    retrieval = cases["chunk_overlap_retrieval"]
    retrieved_ids = {
        item.chunk_id
        for item in fixture.retriever.retrieve(retrieval.query, top_k=5)
    }
    assert fixture.index_version == "smoke-index-v1"
    assert set(retrieval.gold_retrieval_ids) <= retrieved_ids
    assert retrieval.metadata["annotation_source"].startswith(
        "evals/fixtures/smoke_v1.json#"
    )

    web = cases["recorded_web_research"]
    assert web.required_points
    assert web.metadata["annotation_source"].startswith(
        "evals/fixtures/smoke_v1.json#"
    )

    vision = cases["fixed_vision_research"]
    assert vision.required_points == ()
    assert vision.metadata["label_status"] == "semantic_gold_unavailable"


def test_fixed_fixture_closes_the_baseline_regression_report_loop(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path / "runs", "baseline-run")
    second = _run(tmp_path / "runs", "candidate-run")
    baseline = create_baseline(
        first.run_directory,
        tmp_path / "baselines" / "smoke.json",
        confirmed=True,
        kind=BaselineKind.MOCK_ONLY,
    )
    candidate = load_evaluation_run(second.run_directory)
    policy = replace(
        load_regression_policy(POLICY),
        runtime_max_increase_ratios={},
    )

    comparison = compare_regression(baseline, candidate, policy)
    json_report, markdown_report = write_regression_report(
        comparison,
        baseline,
        candidate,
        tmp_path / "reports",
    )

    assert comparison.status is RegressionStatus.PASS
    assert json_report.is_file()
    assert markdown_report.is_file()
