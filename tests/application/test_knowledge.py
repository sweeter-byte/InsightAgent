from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from insight_agent.application.knowledge import (
    KnowledgeService,
    MaterialProcessingError,
    MaterialStorage,
    MaterialTooLargeError,
    MaterialValidationError,
)
from insight_agent.indexing import TextChunker, make_document_id
from insight_agent.ingestion import Document, SourceType
from insight_agent.retrieval import HybridRetrievalConfig, HybridRetriever
from insight_agent.retrieval.retriever import VectorRetriever
from insight_agent.retrieval.sparse import BM25Retriever


class Upload:
    def __init__(self, filename: str | None, data: bytes) -> None:
        self.filename = filename
        self._data = data
        self._offset = 0

    async def read(self, size: int) -> bytes:
        block = self._data[self._offset : self._offset + size]
        self._offset += len(block)
        return block


@pytest.mark.asyncio
async def test_storage_uses_content_addressed_name_inside_controlled_dir(
    tmp_path: Path,
) -> None:
    root = tmp_path / "materials"
    storage = MaterialStorage(root, max_bytes=100)

    stored = await storage.store(Upload("../../Secrets/Notes.MD", b"hello"))

    digest = hashlib.sha256(b"hello").hexdigest()
    assert stored.material_id == digest
    assert stored.path == root / f"{digest}.md"
    assert stored.path.resolve().parent == root.resolve()
    assert stored.path.read_bytes() == b"hello"
    assert stored.deduplicated is False


@pytest.mark.asyncio
async def test_storage_reports_repeated_content_without_replacing_file(
    tmp_path: Path,
) -> None:
    storage = MaterialStorage(tmp_path / "materials", max_bytes=100)
    first = await storage.store(Upload("first.txt", b"same"))
    second = await storage.store(Upload("renamed.txt", b"same"))

    assert second.path == first.path
    assert second.material_id == first.material_id
    assert second.deduplicated is True
    assert list((tmp_path / "materials").iterdir()) == [first.path]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", [None, "", "notes.exe", "notes.text"])
async def test_storage_rejects_missing_or_unsupported_filename(
    tmp_path: Path, filename: str | None
) -> None:
    storage = MaterialStorage(tmp_path / "materials", max_bytes=100)
    with pytest.raises(MaterialValidationError):
        await storage.store(Upload(filename, b"content"))


@pytest.mark.asyncio
async def test_storage_rejects_empty_and_oversize_files_without_temp_leaks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "materials"
    storage = MaterialStorage(root, max_bytes=4, chunk_bytes=3)

    with pytest.raises(MaterialValidationError, match="empty"):
        await storage.store(Upload("empty.txt", b""))
    with pytest.raises(MaterialTooLargeError, match="4"):
        await storage.store(Upload("large.txt", b"12345"))

    assert list(root.iterdir()) == []


@pytest.mark.asyncio
async def test_storage_requires_vision_configuration_for_images(tmp_path: Path) -> None:
    disabled = MaterialStorage(tmp_path / "disabled", max_bytes=100)
    with pytest.raises(MaterialValidationError, match="Vision.*unavailable"):
        await disabled.store(Upload("chart.png", b"image"))

    enabled = MaterialStorage(
        tmp_path / "enabled", max_bytes=100, image_ingestion_enabled=True
    )
    stored = await enabled.store(Upload("chart.PNG", b"image"))
    assert stored.path.suffix == ".png"


class Embedder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.events.append("embed")
        return [[float(len(text)), 1.0] for text in texts]


class Store:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.point_ids: list[list[str]] = []

    def ensure_collection(self, vector_size: int) -> None:
        assert vector_size == 2
        self.events.append("ensure")

    def upsert(self, chunks, vectors) -> None:
        assert len(chunks) == len(vectors)
        self.events.append("upsert")
        self.point_ids.append([chunk.id for chunk in chunks])


class Refresher:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def refresh(self) -> None:
        self.events.append("refresh")
        if self.fail:
            raise RuntimeError("private refresh failure /tmp/secret")


def test_knowledge_service_indexes_then_refreshes_with_stable_ids(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    document = Document(
        content="alpha beta gamma",
        source=str(tmp_path / "material.txt"),
        source_type=SourceType.TEXT,
        metadata={"filename": "material.txt"},
    )

    def ingest(path: str) -> list[Document]:
        assert path == document.source
        events.append("ingest")
        return [document]

    store = Store(events)
    service = KnowledgeService(
        ingestor=ingest,
        chunker=TextChunker(chunk_size=8, chunk_overlap=0),
        embedder=Embedder(events),
        vector_store=store,
        refresher=Refresher(events),
    )
    material = SimpleNamespace(
        material_id="material-id", path=Path(document.source), deduplicated=False
    )

    first = service.import_material(material)
    material.deduplicated = True
    second = service.import_material(material)

    assert first.material_id == "material-id"
    assert first.document_ids == [make_document_id(document)]
    assert first.status.value == "indexed"
    assert first.chunk_count == 2
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert store.point_ids[0] == store.point_ids[1]
    assert events == [
        "ingest", "embed", "ensure", "upsert", "refresh",
        "ingest", "embed", "ensure", "upsert", "refresh",
    ]


def test_knowledge_service_does_not_refresh_after_index_failure(tmp_path: Path) -> None:
    events: list[str] = []

    class FailingStore(Store):
        def upsert(self, chunks, vectors) -> None:
            super().upsert(chunks, vectors)
            raise RuntimeError("private /tmp/path")

    service = KnowledgeService(
        ingestor=lambda path: [
            Document(path, path, SourceType.TEXT, {})
        ],
        chunker=TextChunker(chunk_size=100, chunk_overlap=0),
        embedder=Embedder(events),
        vector_store=FailingStore(events),
        refresher=Refresher(events),
    )
    material = SimpleNamespace(
        material_id="known-id", path=tmp_path / "source.txt", deduplicated=False
    )

    with pytest.raises(MaterialProcessingError) as caught:
        service.import_material(material)

    assert caught.value.stage == "indexing"
    assert caught.value.material_id == "known-id"
    assert "/tmp" not in str(caught.value)
    assert events == ["embed", "ensure", "upsert"]


def test_knowledge_service_reports_refresh_failure_as_processing_failure(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    path = tmp_path / "source.txt"
    service = KnowledgeService(
        ingestor=lambda _: [Document("text", str(path), SourceType.TEXT, {})],
        chunker=TextChunker(chunk_size=100, chunk_overlap=0),
        embedder=Embedder(events),
        vector_store=Store(events),
        refresher=Refresher(events, fail=True),
    )
    material = SimpleNamespace(
        material_id="known-id", path=path, deduplicated=False
    )

    with pytest.raises(MaterialProcessingError) as caught:
        service.import_material(material)

    assert caught.value.stage == "refresh"
    assert "/tmp/secret" not in str(caught.value)
    assert events[-1] == "refresh"


def test_import_is_visible_to_existing_hybrid_retriever(tmp_path: Path) -> None:
    class MemoryStore:
        def __init__(self) -> None:
            self.chunks = {}

        def ensure_collection(self, vector_size: int) -> None:
            assert vector_size == 2

        def upsert(self, chunks, vectors) -> None:
            assert len(chunks) == len(vectors)
            self.chunks.update({chunk.id: chunk for chunk in chunks})

        def load_chunks(self):
            return list(self.chunks.values())

        def search(self, vector, limit=5, *, source_types=None):
            del vector, source_types
            return [
                SimpleNamespace(
                    id=chunk.id,
                    score=1.0,
                    payload={
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "source_type": chunk.source_type.value,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": chunk.metadata,
                    },
                )
                for chunk in list(self.chunks.values())[:limit]
            ]

    class StableEmbedder:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class StableReranker:
        def rerank(self, query, candidates, top_k):
            del query
            return [replace(item, rerank_score=1.0) for item in candidates[:top_k]]

    store = MemoryStore()
    embedder = StableEmbedder()
    sparse = BM25Retriever(store.load_chunks)
    hybrid = HybridRetriever(
        VectorRetriever(embedder, store),
        sparse,
        StableReranker(),
        config=HybridRetrievalConfig(
            dense_k=5,
            sparse_k=5,
            rerank_k=5,
            final_top_k=5,
            rrf_k=60,
            reranker_model="unused",
        ),
    )
    source = tmp_path / "visible.txt"
    service = KnowledgeService(
        ingestor=lambda _: [
            Document("uniqueterm knowledge", str(source), SourceType.TEXT, {})
        ],
        chunker=TextChunker(chunk_size=100, chunk_overlap=0),
        embedder=embedder,
        vector_store=store,
        refresher=hybrid,
    )

    assert hybrid.retrieve("uniqueterm") == []
    service.import_material(
        SimpleNamespace(material_id="id", path=source, deduplicated=False)
    )

    assert sparse.retrieve("uniqueterm")[0].content == "uniqueterm knowledge"
    assert hybrid.retrieve("uniqueterm")[0].content == "uniqueterm knowledge"
