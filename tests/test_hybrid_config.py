"""Configuration tests for the hybrid retrieval pipeline."""

from __future__ import annotations

import pytest

from insight_agent.retrieval.config import HybridRetrievalConfig


def test_hybrid_config_requires_explicit_reranker_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="RERANKER_MODEL"):
        HybridRetrievalConfig.from_env()


def test_hybrid_config_reads_all_stage_depths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "HYBRID_DENSE_K": "31",
        "HYBRID_SPARSE_K": "29",
        "HYBRID_RERANK_K": "17",
        "HYBRID_FINAL_TOP_K": "7",
        "HYBRID_RRF_K": "41",
        "RERANKER_MODEL": "example/bilingual-reranker",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = HybridRetrievalConfig.from_env()

    assert config == HybridRetrievalConfig(
        dense_k=31,
        sparse_k=29,
        rerank_k=17,
        final_top_k=7,
        rrf_k=41,
        reranker_model="example/bilingual-reranker",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HYBRID_DENSE_K", "0"),
        ("HYBRID_SPARSE_K", "-2"),
        ("HYBRID_RERANK_K", "many"),
        ("HYBRID_RRF_K", "1.5"),
    ],
)
def test_hybrid_config_rejects_invalid_positive_integer(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "configured/model")
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=name):
        HybridRetrievalConfig.from_env()


@pytest.mark.parametrize("value", ["0", "9", "invalid"])
def test_hybrid_config_rejects_final_top_k_outside_tool_contract(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_MODEL", "configured/model")
    monkeypatch.setenv("HYBRID_FINAL_TOP_K", value)

    with pytest.raises(RuntimeError, match="HYBRID_FINAL_TOP_K"):
        HybridRetrievalConfig.from_env()


@pytest.mark.parametrize(
    "overrides",
    [
        {"dense_k": 0},
        {"sparse_k": -1},
        {"rerank_k": True},
        {"rrf_k": 0},
        {"final_top_k": 9},
        {"reranker_model": ""},
    ],
)
def test_direct_config_construction_enforces_invariants(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "dense_k": 20,
        "sparse_k": 20,
        "rerank_k": 20,
        "final_top_k": 5,
        "rrf_k": 60,
        "reranker_model": "configured/model",
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        HybridRetrievalConfig(**values)  # type: ignore[arg-type]
