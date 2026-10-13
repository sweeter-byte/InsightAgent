from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.beir_scifact.report import regenerate_report, render_summary


def _metrics():
    return {
        name: {
            "ndcg_at_10": value,
            "recall_at_5": value,
            "recall_at_10": value,
            "mrr_at_10": value,
            "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0},
        }
        for name, value in {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}.items()
    }


def test_summary_is_chinese_and_discloses_deltas_and_limitations() -> None:
    summary = render_summary(
        {
            "run_id": "run-1",
            "smoke": False,
            "dataset": {"query_count": 300},
            "config": {
                "embedding_model": "embed/model",
                "reranker_model": "rerank/model",
                "chunk_size": 777,
                "chunk_overlap": 77,
                "dense_k": 61,
                "sparse_k": 62,
                "rerank_k": 90,
                "document_top_k": 10,
                "rrf_k": 55,
                "embedding_batch_size": 8,
                "document_batch_size": 16,
                "reranker_batch_size": 4,
            },
            "index_fingerprint": "fingerprint-123",
            "reproduce": "scripts/benchmarks/scifact.sh full --dense-k 61",
        },
        _metrics(),
        {"improvements": [{"query_id": "q1"}], "regressions": [{"query_id": "q2"}]},
    )

    assert "SciFact 检索评测" in summary
    assert "D 相比 A" in summary
    assert "+0.300000" in summary
    assert "q1" in summary and "q2" in summary
    assert "Chunk 索引" in summary
    assert "777" in summary and "61" in summary and "fingerprint-123" in summary
    assert "严格递减" in summary
    assert "scripts/benchmarks/scifact.sh full --dense-k 61" in summary


def test_regenerate_report_refuses_incomplete_run(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "failed"}))

    with pytest.raises(ValueError, match="completed run"):
        regenerate_report(tmp_path)


def test_regenerate_report_uses_persisted_metrics_without_inference(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "complete", "run_id": "run-1", "smoke": True})
    )
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics()))
    (tmp_path / "bad_cases.json").write_text(
        json.dumps({"improvements": [], "regressions": []})
    )

    output = regenerate_report(tmp_path)

    assert output == tmp_path / "summary.md"
    assert "Smoke" in output.read_text()
