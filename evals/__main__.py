"""CLI for explicit Offline Mock Evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.answer_eval import AnswerEvaluator
from evals.baseline import BaselineKind, create_baseline, load_baseline
from evals.dataset import load_eval_cases
from evals.evidence_eval import EvidenceEvaluator
from evals.models import EvalMode, EvaluationType
from evals.offline import load_offline_fixture
from evals.regression import (
    RegressionStatus,
    compare_regression,
    load_regression_policy,
)
from evals.report import write_regression_report
from evals.results import load_evaluation_run
from evals.retrieval_eval import RetrievalEvaluator
from evals.runner import EvaluationRunner, build_run_metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run InsightAgent Evaluation")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--mode", choices=[mode.value for mode in EvalMode], default="offline")
    parser.add_argument("--output", type=Path, default=Path("eval_results/runs"))
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--dataset-version")
    parser.add_argument("--model", default="recorded-fixture")
    parser.add_argument("--prompt-version", default="fixture-judge-v1")
    parser.add_argument("--index-version")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-id")
    parser.add_argument("--register-baseline", type=Path, metavar="RUN_DIRECTORY")
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument(
        "--baseline-kind",
        choices=[kind.value for kind in BaselineKind],
    )
    parser.add_argument("--confirm-baseline", action="store_true")
    parser.add_argument("--overwrite-baseline", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("eval_results/reports"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.register_baseline is not None:
        return _register_baseline(args)
    if args.baseline is not None or args.candidate is not None:
        return _compare_runs(args)

    mode = EvalMode(args.mode)
    if mode is EvalMode.LIVE:
        print(
            "[evaluation config error] Live Test requires explicit real components; "
            "the fixture CLI never substitutes Mock Test components.",
            file=sys.stderr,
        )
        return 2
    if args.dataset is None:
        print(
            "[evaluation config error] --dataset is required to run Evaluation",
            file=sys.stderr,
        )
        return 2
    if args.fixture is None:
        print(
            "[evaluation config error] --fixture is required in offline mode",
            file=sys.stderr,
        )
        return 2

    project_root = Path(__file__).resolve().parents[1]
    try:
        fixture = load_offline_fixture(args.fixture, project_root=project_root)
        index_version = args.index_version or fixture.index_version
        if index_version != fixture.index_version:
            raise ValueError(
                "--index-version does not match the recorded fixture index_version"
            )
        cases = load_eval_cases(args.dataset)
        metadata = build_run_metadata(
            dataset_path=args.dataset,
            dataset_version=args.dataset_version or args.dataset.stem,
            repo_root=project_root,
            model_name=args.model,
            prompt_version=args.prompt_version,
            index_version=index_version,
            retriever_configuration={
                "top_k": args.top_k,
                "fixture_version": fixture.fixture_version,
            },
            mode=mode,
            evaluation_type=EvaluationType.MOCK_TEST,
            eval_run_id=args.run_id,
        )
        summary = EvaluationRunner(
            metadata=metadata,
            output_root=args.output,
            retrieval_evaluator=RetrievalEvaluator(fixture.retriever),
            workflow_runner=fixture.workflow_runner,
            evidence_evaluator=EvidenceEvaluator(fixture.judge),
            answer_evaluator=AnswerEvaluator(fixture.judge),
            top_k=args.top_k,
        ).run(cases, case_ids=set(args.case_ids) if args.case_ids else None)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        detail = str(exc).strip() or type(exc).__name__
        print(f"[evaluation error] {detail}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "run_directory": str(summary.run_directory),
                "config_fingerprint": summary.metadata.config_fingerprint,
                "aggregates": summary.aggregates,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print("MOCK TEST: fixture results do not measure real-model research quality.")
    failed = summary.aggregates["cases"]["failed"]  # type: ignore[index]
    return 1 if failed else 0


def _register_baseline(args: argparse.Namespace) -> int:
    if args.baseline_output is None or args.baseline_kind is None:
        print(
            "[evaluation config error] --baseline-output and --baseline-kind "
            "are required with --register-baseline",
            file=sys.stderr,
        )
        return 2
    if not args.confirm_baseline:
        print(
            "[evaluation config error] --confirm-baseline is required; "
            "Baseline registration is never automatic",
            file=sys.stderr,
        )
        return 2
    try:
        record = create_baseline(
            args.register_baseline,
            args.baseline_output,
            confirmed=True,
            kind=BaselineKind(args.baseline_kind),
            overwrite=args.overwrite_baseline,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        detail = str(exc).strip() or type(exc).__name__
        print(f"[baseline error] {detail}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "baseline": str(args.baseline_output),
                "kind": record.kind.value,
                "source_run_id": record.source_run.metadata["eval_run_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if record.kind is BaselineKind.MOCK_ONLY:
        print("MOCK-ONLY BASELINE: this does not establish real-model quality.")
    return 0


def _compare_runs(args: argparse.Namespace) -> int:
    if args.baseline is None or args.candidate is None:
        print(
            "[evaluation config error] --baseline and --candidate must be "
            "provided together",
            file=sys.stderr,
        )
        return 2
    default_policy = Path(__file__).with_name("policies") / "default_v1.json"
    try:
        baseline = load_baseline(args.baseline)
        candidate = load_evaluation_run(args.candidate)
        policy = load_regression_policy(args.policy or default_policy)
        comparison = compare_regression(baseline, candidate, policy)
        json_path, markdown_path = write_regression_report(
            comparison,
            baseline,
            candidate,
            args.report_output,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        detail = str(exc).strip() or type(exc).__name__
        print(f"[regression error] {detail}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": comparison.status.value,
                "baseline_run_id": comparison.baseline_run_id,
                "candidate_run_id": comparison.candidate_run_id,
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return {
        RegressionStatus.PASS: 0,
        RegressionStatus.REGRESSION: 1,
        RegressionStatus.INCOMPATIBLE: 2,
        RegressionStatus.INCONCLUSIVE: 3,
    }[comparison.status]


if __name__ == "__main__":
    raise SystemExit(main())
