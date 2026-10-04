"""Tests for the Document -> Chunk -> Vector -> Qdrant write pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.ingestion import Document, SourceType, ingest_file
from insight_agent.indexing import QdrantVectorStore, TextChunker, index_documents


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FixedEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectors


class RecordingStore:
    def __init__(self) -> None:
        self.ensure_calls: list[int] = []
        self.upsert_calls: list[tuple[object, object]] = []

    def ensure_collection(self, vector_size: int) -> None:
        self.ensure_calls.append(vector_size)

    def upsert(self, chunks, vectors) -> None:
        self.upsert_calls.append((chunks, vectors))


def _documents() -> list[Document]:
    return [
        Document(
            content="0123456789abcdefghij",
            source="notes.txt",
            source_type=SourceType.TEXT,
            metadata={"filename": "notes.txt"},
        )
    ]


def test_pipeline_indexes_ingested_file_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("0123456789abcdefghij", encoding="utf-8")
    documents = ingest_file(str(source))
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)
    embedder = FakeEmbedder()
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="pipeline_documents",
    )
    try:
        count = index_documents(documents, chunker, embedder, store)

        point_count = store.client.count(
            collection_name="pipeline_documents", exact=True
        ).count
        assert count == len(embedder.calls[0]) == point_count
        assert count > 1
    finally:
        store.close()


def test_empty_documents_short_circuit_without_embedding_or_store_calls() -> None:
    embedder = FakeEmbedder()
    store = RecordingStore()

    count = index_documents([], TextChunker(), embedder, store)

    assert count == 0
    assert embedder.calls == []
    assert store.ensure_calls == []
    assert store.upsert_calls == []


def test_pipeline_rejects_embedding_count_mismatch() -> None:
    store = RecordingStore()
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    with pytest.raises(
        ValueError, match="embedding count mismatch.*chunks=.*vectors=1"
    ):
        index_documents(_documents(), chunker, FixedEmbedder([[1.0, 2.0]]), store)

    assert store.ensure_calls == []
    assert store.upsert_calls == []


def test_pipeline_rejects_empty_vector() -> None:
    store = RecordingStore()

    with pytest.raises(ValueError, match="embedding vector at index 0 is empty"):
        index_documents(
            [Document("short", "s", SourceType.TEXT, {})],
            TextChunker(),
            FixedEmbedder([[]]),
            store,
        )

    assert store.ensure_calls == []
    assert store.upsert_calls == []


def test_pipeline_rejects_inconsistent_vector_dimensions() -> None:
    store = RecordingStore()
    documents = [Document("abcdefghij", "s", SourceType.TEXT, {})]

    with pytest.raises(
        ValueError,
        match="inconsistent embedding dimensions.*expected=2.*index 1 has dimension 1",
    ):
        index_documents(
            documents,
            TextChunker(chunk_size=6, chunk_overlap=2),
            FixedEmbedder([[1.0, 2.0], [3.0]]),
            store,
        )

    assert store.ensure_calls == []
    assert store.upsert_calls == []


def test_pipeline_ensures_collection_before_upsert() -> None:
    events: list[str] = []

    class OrderedStore:
        def ensure_collection(self, vector_size: int) -> None:
            assert vector_size == 2
            events.append("ensure")

        def upsert(self, chunks, vectors) -> None:
            assert len(chunks) == len(vectors) == 1
            events.append("upsert")

    count = index_documents(
        [Document("short", "s", SourceType.TEXT, {})],
        TextChunker(),
        FixedEmbedder([[1.0, 2.0]]),
        OrderedStore(),
    )

    assert count == 1
    assert events == ["ensure", "upsert"]
