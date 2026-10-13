"""Human-readable report generation from completed persisted artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def render_summary(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    bad_cases: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    smoke = bool(manifest.get("smoke"))
    label = "Smoke（非正式 Benchmark）" if smoke else "完整 Test Split"
    dataset = manifest.get("dataset", {})
    query_count = (
        dataset.get("query_count", manifest.get("query_count", "未知"))
        if isinstance(dataset, Mapping)
        else manifest.get("query_count", "未知")
    )
    config = manifest.get("config", {})
    embedding = config.get("embedding_model", "BAAI/bge-m3") if isinstance(config, Mapping) else "未知"
    reranker = config.get("reranker_model", "BAAI/bge-reranker-v2-m3") if isinstance(config, Mapping) else "未知"
    lines = [
        "# BEIR SciFact 检索评测",
        "",
        f"- Run ID：`{manifest.get('run_id', 'unknown')}`",
        f"- 运行类型：{label}",
        f"- Test Query 数：{query_count}",
        f"- Embedding：`{embedding}`",
        f"- Reranker：`{reranker}`",
        f"- Index Fingerprint：`{manifest.get('index_fingerprint', '未知')}`",
        "",
        "## 检索配置",
        "",
    ]
    if isinstance(config, Mapping):
        for name in (
            "chunk_size",
            "chunk_overlap",
            "embedding_batch_size",
            "document_batch_size",
            "dense_k",
            "sparse_k",
            "rerank_k",
            "document_top_k",
            "rrf_k",
            "reranker_batch_size",
        ):
            lines.append(f"- `{name}`：`{config.get(name, '未知')}`")
    lines.extend([
        "",
        "## 指标对比",
        "",
        "| 实验 | NDCG@10 | Recall@5 | Recall@10 | MRR@10 | Mean ms | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("A", "B", "C", "D"):
        values = metrics[name]
        latency = values["latency_ms"]
        lines.append(
            f"| {name} | {values['ndcg_at_10']:.6f} | "
            f"{values['recall_at_5']:.6f} | {values['recall_at_10']:.6f} | "
            f"{values['mrr_at_10']:.6f} | {latency['mean']:.3f} | "
            f"{latency['p50']:.3f} | {latency['p95']:.3f} |"
        )
    lines.extend(["", "## D 的指标变化", ""])
    for baseline in ("A", "B", "C"):
        deltas = [
            f"{metric} {metrics['D'][metric] - metrics[baseline][metric]:+.6f}"
            for metric in ("ndcg_at_10", "recall_at_5", "recall_at_10", "mrr_at_10")
        ]
        lines.append(f"- D 相比 {baseline}：" + "，".join(deltas))

    lines.extend(["", "## 代表性 Case", ""])
    improvements = bad_cases.get("improvements", [])
    regressions = bad_cases.get("regressions", [])
    lines.append(
        "- 改进："
        + (", ".join(str(item["query_id"]) for item in improvements[:5]) or "无")
    )
    lines.append(
        "- 退化："
        + (", ".join(str(item["query_id"]) for item in regressions[:5]) or "无")
    )
    lines.extend(
        [
            "",
            "## 限制与复现",
            "",
            "本实验使用 InsightAgent 的 Chunk 索引和检索组件，但按原始 BEIR Document ID 去重后计算文档级指标。它并非 BEIR 官方模型配置；Chunk 切分、候选深度和模型版本均会影响结果。A/B 延迟为实际检索阶段，C 为已取得候选后的 RRF，D 为相同 RRF 候选池上的 Cross-Encoder 重排。",
            "",
            "写入 BEIR 的分数由文档名次转换为严格递减的唯一值，只用于保持 InsightAgent 原始排序；Qrels 仅在指标与 Bad Case 阶段读取。",
            "",
            "```bash",
            str(manifest.get("reproduce", "scripts/benchmarks/scifact.sh full")),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def regenerate_report(run_directory: str | Path) -> Path:
    directory = Path(run_directory)
    manifest = _load(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("report regeneration requires a completed run")
    metrics = _load(directory / "metrics.json")
    bad_cases = _load(directory / "bad_cases.json")
    output = directory / "summary.md"
    output.write_text(render_summary(manifest, metrics, bad_cases), encoding="utf-8")
    return output


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value
