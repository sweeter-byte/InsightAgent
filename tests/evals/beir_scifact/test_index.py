from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.beir_scifact.config import BenchmarkConfig, BenchmarkPaths
from evals.beir_scifact.index import (
    build_or_reuse_index,
    index_fingerprint,
    validate_completed_index,
)
from insight_agent.ingestion import Document, SourceType
from insight_agent.indexing import TextChunker


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    def __init__(self) -> None:
        self.points: dict[str, object] = {}
        self.dimensions: list[int] = []

    def ensure_collection(self, vector_size: int) -> None:
        self.dimensions.append(vector_size)

    def upsert(self, chunks, vectors) -> None:
        self.points.update({chunk.id: chunk for chunk in chunks})

    def point_count(self) -> int:
        return len(self.points)


def _documents() -> list[Document]:
    return [
        Document(
            content=f"document {index} content",
            source=f"beir://scifact/{index}",
            source_type=SourceType.TEXT,
            metadata={"beir_doc_id": str(index)},
        )
        for index in range(3)
    ]


def test_benchmark_paths_are_isolated_from_production_defaults(tmp_path: Path) -> None:
    paths = BenchmarkPaths(root=tmp_path / "bench")

    assert paths.qdrant_path == tmp_path / "bench" / "qdrant"
    assert paths.collection_name == "beir_scifact_chunks"
    assert paths.qdrant_path != Path(".data/qdrant")


def test_index_fingerprint_changes_with_model_or_chunking() -> None:
    base = BenchmarkConfig()

    assert index_fingerprint(base, corpus_md5="abc") == index_fingerprint(
        base, corpus_md5="abc"
    )
    assert index_fingerprint(base, corpus_md5="abc") != index_fingerprint(
        BenchmarkConfig(embedding_model="other/model"), corpus_md5="abc"
    )
    assert index_fingerprint(base, corpus_md5="abc") != index_fingerprint(
        BenchmarkConfig(chunk_size=512), corpus_md5="abc"
    )
    assert index_fingerprint(
        base, corpus_md5="abc", collection_name="collection-a"
    ) != index_fingerprint(
        base, corpus_md5="abc", collection_name="collection-b"
    )


def test_index_batches_documents_and_writes_completed_manifest(tmp_path: Path) -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    manifest = tmp_path / "index-manifest.json"

    result = build_or_reuse_index(
        _documents(),
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
        embedder=embedder,
        vector_store=store,
        manifest_path=manifest,
        fingerprint="fingerprint-1",
        document_batch_size=2,
    )

    saved = json.loads(manifest.read_text())
    assert result.reused is False
    assert result.document_count == 3
    assert result.chunk_count == 3
    assert [len(batch) for batch in embedder.batches] == [2, 1]
    assert saved["status"] == "complete"
    assert saved["fingerprint"] == "fingerprint-1"
    assert saved["completed_document_ids"] == ["0", "1", "2"]


def test_matching_complete_index_is_reused_without_embedding(tmp_path: Path) -> None:
    documents = _documents()
    store = FakeStore()
    first_embedder = FakeEmbedder()
    manifest = tmp_path / "index-manifest.json"
    build_or_reuse_index(
        documents,
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
        embedder=first_embedder,
        vector_store=store,
        manifest_path=manifest,
        fingerprint="same",
        document_batch_size=2,
    )
    second_embedder = FakeEmbedder()

    result = build_or_reuse_index(
        documents,
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
        embedder=second_embedder,
        vector_store=store,
        manifest_path=manifest,
        fingerprint="same",
        document_batch_size=2,
    )

    assert result.reused is True
    assert second_embedder.batches == []


def test_index_rejects_mismatched_existing_configuration(tmp_path: Path) -> None:
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps({"status": "complete", "fingerprint": "old", "chunk_count": 0})
    )

    with pytest.raises(ValueError, match="configuration mismatch"):
        build_or_reuse_index(
            _documents(),
            chunker=TextChunker(chunk_size=100, chunk_overlap=10),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            manifest_path=manifest,
            fingerprint="new",
            document_batch_size=2,
        )


def test_index_rejects_unidentified_existing_points_without_manifest(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    chunk = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(
        [_documents()[0]]
    )[0]
    store.points[chunk.id] = chunk

    with pytest.raises(ValueError, match="points but no index manifest"):
        build_or_reuse_index(
            _documents(),
            chunker=TextChunker(chunk_size=100, chunk_overlap=10),
            embedder=FakeEmbedder(),
            vector_store=store,
            manifest_path=tmp_path / "missing.json",
            fingerprint="new",
            document_batch_size=2,
        )


def test_incomplete_index_resumes_completed_document_batches(tmp_path: Path) -> None:
    documents = _documents()
    store = FakeStore()
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    first_chunk = chunker.split_documents([documents[0]])[0]
    store.points[first_chunk.id] = first_chunk
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "running",
                "fingerprint": "same",
                "completed_document_ids": ["0"],
                "chunk_count": 1,
            }
        )
    )
    embedder = FakeEmbedder()

    result = build_or_reuse_index(
        documents,
        chunker=chunker,
        embedder=embedder,
        vector_store=store,
        manifest_path=manifest,
        fingerprint="same",
        document_batch_size=2,
    )

    assert result.chunk_count == 3
    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 2


def test_resume_tolerates_points_written_before_progress_manifest(tmp_path: Path) -> None:
    documents = _documents()
    store = FakeStore()
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    crash_window_chunk = chunker.split_documents([documents[0]])[0]
    store.points[crash_window_chunk.id] = crash_window_chunk
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "running",
                "fingerprint": "same",
                "completed_document_ids": [],
                "chunk_count": 0,
            }
        )
    )

    result = build_or_reuse_index(
        documents,
        chunker=chunker,
        embedder=FakeEmbedder(),
        vector_store=store,
        manifest_path=manifest,
        fingerprint="same",
        document_batch_size=2,
    )

    assert result.chunk_count == 3
    assert store.point_count() == 3


def test_completed_index_validation_rejects_extra_or_replaced_points(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    manifest = tmp_path / "index-manifest.json"
    build_or_reuse_index(
        _documents(),
        chunker=TextChunker(chunk_size=100, chunk_overlap=10),
        embedder=FakeEmbedder(),
        vector_store=store,
        manifest_path=manifest,
        fingerprint="same",
        document_batch_size=2,
    )
    payload = json.loads(manifest.read_text())
    validate_completed_index(store, payload, expected_fingerprint="same")

    original_id = next(iter(store.points))
    original = store.points.pop(original_id)
    store.points["replacement-id"] = original
    with pytest.raises(ValueError, match="chunk identity"):
        validate_completed_index(store, payload, expected_fingerprint="same")

    store.points[original_id] = original
    with pytest.raises(ValueError, match="point count"):
        validate_completed_index(store, payload, expected_fingerprint="same")


def test_partial_manifest_rejects_unknown_completed_document_id(tmp_path: Path) -> None:
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "running",
                "fingerprint": "same",
                "completed_document_ids": ["not-in-corpus"],
                "chunk_count": 0,
            }
        )
    )

    with pytest.raises(ValueError, match="unknown completed document"):
        build_or_reuse_index(
            _documents(),
            chunker=TextChunker(chunk_size=100, chunk_overlap=10),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(),
            manifest_path=manifest,
            fingerprint="same",
            document_batch_size=2,
        )


def test_failed_final_identity_check_never_publishes_complete_manifest(
    tmp_path: Path,
) -> None:
    documents = _documents()
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.split_documents(documents)
    store = FakeStore()
    # Document 0 was declared complete but its point was lost; document 1 was
    # upserted just before a crash and is not yet recorded as complete.
    store.points[chunks[1].id] = chunks[1]
    manifest = tmp_path / "index-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "running",
                "fingerprint": "same",
                "completed_document_ids": ["0"],
                "chunk_count": 1,
            }
        )
    )

    with pytest.raises(ValueError, match="exact expected chunk identities"):
        build_or_reuse_index(
            documents,
            chunker=chunker,
            embedder=FakeEmbedder(),
            vector_store=store,
            manifest_path=manifest,
            fingerprint="same",
            document_batch_size=2,
        )

    assert json.loads(manifest.read_text())["status"] == "running"
