"""Unit tests for tokenization and the in-memory BM25 retriever."""

from __future__ import annotations

import threading

import pytest
from rank_bm25 import BM25Okapi

from insight_agent.indexing import Chunk
from insight_agent.ingestion import SourceType
from insight_agent.retrieval.sparse import BM25Retriever, tokenize


def _chunk(
    chunk_id: str,
    content: str,
    *,
    source: str = "notes.md",
    metadata: dict[str, object] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        source=source,
        source_type=SourceType.MARKDOWN,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
        metadata=dict(metadata or {}),
    )


def test_tokenizer_preserves_technical_identifiers_and_paths() -> None:
    tokens = tokenize(
        "设置 DEEPSEEK_API_KEY 后处理 APITimeoutError，"
        "再调用 search_knowledge_base 读取 /path/to/file。"
    )

    assert "deepseek_api_key" in tokens
    assert "apitimeouterror" in tokens
    assert "search_knowledge_base" in tokens
    assert "/path/to/file" in tokens
    assert any(token in tokens for token in ("设置", "处理", "调用", "读取"))


def test_bm25_exact_identifier_ranks_matching_chunk_first_and_keeps_provenance() -> None:
    chunks = [
        _chunk(
            "api-key",
            "将 DEEPSEEK_API_KEY 写入环境变量。",
            source="config/guide.md",
            metadata={"section": "configuration"},
        ),
        _chunk("timeout", "APITimeoutError 表示请求超时。"),
        _chunk("natural", "普通自然语言介绍检索系统。"),
    ]
    retriever = BM25Retriever(lambda: chunks)

    results = retriever.retrieve("DEEPSEEK_API_KEY", top_k=2)

    assert results[0].chunk_id == "api-key"
    assert results[0].source == "config/guide.md"
    assert results[0].metadata == {"section": "configuration"}
    assert results[0].document_id == "doc-api-key"


def test_bm25_exact_identifier_wins_in_two_chunk_corpus() -> None:
    chunks = [
        _chunk("ordinary", "普通自然语言说明。"),
        _chunk("api-key", "使用 DEEPSEEK_API_KEY 配置服务。"),
    ]
    retriever = BM25Retriever(lambda: chunks)

    results = retriever.retrieve("DEEPSEEK_API_KEY", top_k=2)

    assert results[0].chunk_id == "api-key"
    assert results[0].score > results[1].score


def test_bm25_skips_tokenless_chunks_without_crashing() -> None:
    chunks = [
        _chunk("emoji", "😀 !!!"),
        _chunk("searchable", "DEEPSEEK_API_KEY"),
    ]
    retriever = BM25Retriever(lambda: chunks)

    results = retriever.retrieve("DEEPSEEK_API_KEY", top_k=2)

    assert [result.chunk_id for result in results] == ["searchable"]


def test_bm25_all_tokenless_chunks_publish_empty_snapshot() -> None:
    retriever = BM25Retriever(lambda: [_chunk("emoji", "😀 !!!")])

    assert retriever.retrieve("query") == []


def test_bm25_refresh_failure_preserves_previous_snapshot() -> None:
    state: dict[str, object] = {"chunks": [_chunk("old", "stable_token")]}

    def load_chunks() -> list[Chunk]:
        error = state.get("error")
        if isinstance(error, Exception):
            raise error
        return list(state["chunks"])  # type: ignore[arg-type]

    retriever = BM25Retriever(load_chunks)
    state["error"] = RuntimeError("qdrant unavailable")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        retriever.refresh()

    assert retriever.retrieve("stable_token", top_k=1)[0].chunk_id == "old"


def test_bm25_refresh_builds_before_atomically_swapping_snapshot() -> None:
    old_chunks = [_chunk("old", "old_unique_token")]
    new_chunks = [_chunk("new", "new_unique_token")]
    current = {"chunks": old_chunks}
    build_started = threading.Event()
    allow_build = threading.Event()
    factory_calls = 0

    def bm25_factory(corpus: list[list[str]]) -> BM25Okapi:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 2:
            build_started.set()
            assert allow_build.wait(timeout=5)
        return BM25Okapi(corpus)

    retriever = BM25Retriever(
        lambda: current["chunks"],
        bm25_factory=bm25_factory,
    )
    current["chunks"] = new_chunks
    thread = threading.Thread(target=retriever.refresh)
    thread.start()
    assert build_started.wait(timeout=5)

    while_rebuilding = retriever.retrieve("old_unique_token", top_k=1)
    assert while_rebuilding[0].chunk_id == "old"

    allow_build.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert retriever.retrieve("new_unique_token", top_k=1)[0].chunk_id == "new"


def test_bm25_refresh_to_tokenless_corpus_atomically_publishes_empty_snapshot() -> None:
    current = {"chunks": [_chunk("old", "stable_token")]}
    retriever = BM25Retriever(lambda: current["chunks"])
    current["chunks"] = [_chunk("emoji", "😀 !!!")]

    retriever.refresh()

    assert retriever.retrieve("stable_token") == []


def test_bm25_empty_snapshot_returns_no_results() -> None:
    retriever = BM25Retriever(lambda: [])

    assert retriever.retrieve("anything") == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_bm25_rejects_non_positive_top_k(top_k: int) -> None:
    retriever = BM25Retriever(lambda: [])

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("query", top_k=top_k)
