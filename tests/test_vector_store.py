"""Qdrant Local Mode tests; every database lives under pytest's tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.models import Distance, VectorParams

from insight_agent.ingestion import Document, SourceType
from insight_agent.indexing import QdrantVectorStore, TextChunker


def _chunk():
    document = Document(
        content="A PDF page about deterministic indexing.",
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"filename": "paper.pdf", "page": 3, "page_count": 10},
    )
    return TextChunker(chunk_size=100, chunk_overlap=10).split_documents([document])[0]


def _store(tmp_path: Path) -> QdrantVectorStore:
    return QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_documents",
    )


def test_ensure_collection_creates_cosine_collection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.ensure_collection(vector_size=3)

        info = store.client.get_collection("test_documents")
        vectors = info.config.params.vectors
        assert vectors.size == 3
        assert str(vectors.distance).lower().endswith("cosine")
    finally:
        store.close()


def test_existing_collection_accepts_same_dimension(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.ensure_collection(vector_size=2)
        store.ensure_collection(vector_size=2)

        assert store.client.collection_exists("test_documents")
    finally:
        store.close()


def test_existing_collection_rejects_incompatible_dimension(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.ensure_collection(vector_size=2)

        with pytest.raises(
            ValueError,
            match="Embedding model / vector dimension incompatibility.*existing=2.*requested=3",
        ):
            store.ensure_collection(vector_size=3)
    finally:
        store.close()


def test_existing_collection_rejects_non_cosine_distance(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.client.create_collection(
            collection_name="test_documents",
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
        )

        with pytest.raises(ValueError, match="distance incompatibility.*Cosine"):
            store.ensure_collection(vector_size=2)
    finally:
        store.close()


def test_upsert_writes_vector_and_complete_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunk = _chunk()
    # Qdrant normalizes vectors in a COSINE collection; use a unit vector so
    # retrieval can assert the stored values directly.
    vector = [1.0, 0.0, 0.0]
    try:
        store.ensure_collection(vector_size=len(vector))
        store.upsert([chunk], [vector])

        points = store.client.retrieve(
            collection_name="test_documents",
            ids=[chunk.id],
            with_payload=True,
            with_vectors=True,
        )
        assert len(points) == 1
        point = points[0]
        assert point.id == chunk.id
        assert point.vector == pytest.approx(vector)
        assert point.payload == {
            "content": chunk.content,
            "document_id": chunk.document_id,
            "source": "paper.pdf",
            "source_type": "pdf",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(chunk.content),
            "metadata": chunk.metadata,
        }
    finally:
        store.close()


def test_repeated_upsert_of_stable_id_does_not_increase_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    chunk = _chunk()
    vectors = [[0.25, 0.75]]
    try:
        store.ensure_collection(vector_size=2)
        store.upsert([chunk], vectors)
        store.upsert([chunk], vectors)

        count = store.client.count(
            collection_name="test_documents", exact=True
        ).count
        assert count == 1
    finally:
        store.close()


def test_upsert_rejects_chunk_vector_count_mismatch(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="chunk/vector count mismatch"):
            store.upsert([_chunk()], [])
    finally:
        store.close()
