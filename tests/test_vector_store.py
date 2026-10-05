"""Qdrant Local Mode tests; every database lives under pytest's tmp_path."""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

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


def test_search_returns_ranked_payloads_without_stored_vectors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    documents = [
        Document(
            content="Project Helios Alpha scored 31.7.",
            source="helios.txt",
            source_type=SourceType.TEXT,
            metadata={"method": "Alpha"},
        ),
        Document(
            content="Project Helios Beta scored 48.9.",
            source="helios.txt",
            source_type=SourceType.TEXT,
            metadata={"method": "Beta"},
        ),
        Document(
            content="Project Helios Gamma scored 42.3.",
            source="helios.txt",
            source_type=SourceType.TEXT,
            metadata={"method": "Gamma"},
        ),
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(documents)
    vectors = [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]]
    try:
        store.ensure_collection(vector_size=2)
        store.upsert(chunks, vectors)

        points = store.search([1.0, 0.0], limit=2)

        assert len(points) == 2
        assert points[0].payload == {
            "content": "Project Helios Beta scored 48.9.",
            "document_id": chunks[1].document_id,
            "source": "helios.txt",
            "source_type": "text",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len("Project Helios Beta scored 48.9."),
            "metadata": chunks[1].metadata,
        }
        assert points[0].score == pytest.approx(1.0)
        assert all(point.payload is not None for point in points)
        assert all(point.vector is None for point in points)
    finally:
        store.close()


def test_search_filters_source_types_before_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    documents = [
        Document(
            content="A text result with the closest vector.",
            source="notes.txt",
            source_type=SourceType.TEXT,
        ),
        Document(
            content="An indexed image description.",
            source="diagram.png",
            source_type=SourceType.IMAGE,
        ),
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(documents)
    try:
        store.ensure_collection(vector_size=2)
        store.upsert(chunks, [[1.0, 0.0], [0.8, 0.2]])

        points = store.search(
            [1.0, 0.0],
            limit=1,
            source_types={"image"},
        )

        assert len(points) == 1
        assert points[0].payload["source"] == "diagram.png"
        assert points[0].payload["source_type"] == "image"
    finally:
        store.close()


def test_load_chunks_can_filter_image_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)
    documents = [
        Document("Text", "notes.txt", SourceType.TEXT),
        Document("Diagram", "diagram.png", SourceType.IMAGE),
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(documents)
    try:
        store.ensure_collection(vector_size=2)
        store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

        restored = store.load_chunks(source_types={"image"})

        assert [chunk.source for chunk in restored] == ["diagram.png"]
    finally:
        store.close()


def test_iter_chunks_can_lazily_filter_image_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)
    documents = [
        Document("Text", "notes.txt", SourceType.TEXT),
        Document("Diagram", "diagram.png", SourceType.IMAGE),
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(documents)
    try:
        store.ensure_collection(vector_size=2)
        store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

        restored = list(store.iter_chunks(batch_size=1, source_types={"image"}))

        assert [chunk.source for chunk in restored] == ["diagram.png"]
    finally:
        store.close()


def test_load_chunks_scrolls_all_pages_and_restores_domain_models(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    documents = [
        Document(
            content=f"Chunk payload {index}",
            source=f"notes-{index}.txt",
            source_type=SourceType.TEXT,
            metadata={"index": index},
        )
        for index in range(3)
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(documents)
    try:
        store.ensure_collection(vector_size=2)
        store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])

        restored = store.load_chunks(batch_size=1)

        assert {chunk.id: chunk for chunk in restored} == {
            chunk.id: chunk for chunk in chunks
        }
    finally:
        store.close()


def test_load_chunks_rejects_malformed_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.ensure_collection(vector_size=2)
        store.client.upsert(
            collection_name="test_documents",
            points=[
                PointStruct(
                    id="7c789742-8e02-49c6-9d6d-850827f27a4d",
                    vector=[1.0, 0.0],
                    payload={"content": "missing provenance"},
                )
            ],
            wait=True,
        )

        with pytest.raises(ValueError, match="document_id"):
            store.load_chunks()
    finally:
        store.close()


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_load_chunks_rejects_invalid_batch_size(
    batch_size: int,
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            store.load_chunks(batch_size=batch_size)
    finally:
        store.close()
