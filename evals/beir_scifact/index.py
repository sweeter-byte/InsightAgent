"""Resumable benchmark-only index construction and identity checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from evals.beir_scifact.config import BenchmarkConfig
from insight_agent.ingestion import Document
from insight_agent.indexing import Chunk, Embedder, TextChunker


class BenchmarkVectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    reused: bool
    document_count: int
    chunk_count: int
    fingerprint: str


def index_fingerprint(
    config: BenchmarkConfig,
    *,
    corpus_md5: str,
    collection_name: str = "beir_scifact_chunks",
) -> str:
    payload = {
        "corpus_md5": corpus_md5,
        "collection_name": collection_name,
        **config.index_identity(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_chunk_id_digest(
    documents: Sequence[Document],
    chunker: TextChunker,
) -> str:
    return _id_digest({chunk.id for chunk in chunker.split_documents(list(documents))})


def build_or_reuse_index(
    documents: Sequence[Document],
    *,
    chunker: TextChunker,
    embedder: Embedder,
    vector_store: BenchmarkVectorStore,
    manifest_path: str | Path,
    fingerprint: str,
    document_batch_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> IndexBuildResult:
    """Build in document batches and resume only an identical partial index."""
    if document_batch_size <= 0:
        raise ValueError("document_batch_size must be positive")
    target = Path(manifest_path)
    manifest = _read_manifest(target)
    document_ids = [_beir_doc_id(document) for document in documents]
    expected_chunks = chunker.split_documents(list(documents))
    expected_chunk_ids = {chunk.id for chunk in expected_chunks}
    expected_digest = _id_digest(expected_chunk_ids)
    if manifest is not None and manifest.get("fingerprint") != fingerprint:
        raise ValueError(
            "existing SciFact index configuration mismatch; choose a new "
            "benchmark root or remove the old benchmark index explicitly"
        )

    if manifest is not None and manifest.get("status") == "complete":
        validate_completed_index(
            vector_store,
            manifest,
            expected_fingerprint=fingerprint,
        )
        if manifest.get("chunk_id_digest") != expected_digest:
            raise ValueError("completed index chunk identity does not match current corpus")
        chunk_count = _required_int(manifest, "chunk_count")
        return IndexBuildResult(
            reused=True,
            document_count=_required_int(manifest, "document_count"),
            chunk_count=chunk_count,
            fingerprint=fingerprint,
        )

    completed = set(_completed_ids(manifest))
    unknown_completed = completed - set(document_ids)
    if unknown_completed:
        raise ValueError(
            "index manifest contains unknown completed document ID "
            f"{sorted(unknown_completed)[0]!r}"
        )
    saved_chunk_count = int(manifest.get("chunk_count", 0)) if manifest else 0
    actual_chunk_count = _point_count(vector_store)
    if manifest is None and actual_chunk_count:
        raise ValueError(
            "SciFact collection contains points but no index manifest; "
            "use a clean benchmark root or collection"
        )
    if manifest is not None and actual_chunk_count < saved_chunk_count:
        raise ValueError("partial index has fewer Qdrant points than its manifest")
    actual_ids = (
        set() if actual_chunk_count == 0 else _stored_chunk_ids(vector_store)
    )
    unknown_points = actual_ids - expected_chunk_ids
    if unknown_points:
        raise ValueError(
            "partial index contains chunk identities outside the current corpus"
        )

    pending = [
        document
        for document, document_id in zip(documents, document_ids, strict=True)
        if document_id not in completed
    ]
    # Qdrant upserts are idempotent by stable chunk ID. If a process stopped
    # after an upsert but before its manifest update, replaying that batch is safe.
    chunk_count = actual_chunk_count
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        target,
        _manifest_payload(
            status="running",
            fingerprint=fingerprint,
            completed=completed,
            document_count=len(documents),
            chunk_count=chunk_count,
            chunk_id_digest=expected_digest,
        ),
    )

    for offset in range(0, len(pending), document_batch_size):
        batch = pending[offset : offset + document_batch_size]
        chunks = chunker.split_documents(list(batch))
        if chunks:
            vectors = embedder.embed_documents([chunk.content for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError("embedding count does not match chunk count")
            dimensions = {len(vector) for vector in vectors if vector}
            if len(dimensions) != 1 or any(not vector for vector in vectors):
                raise ValueError("embedding vectors must be non-empty and same-sized")
            vector_store.ensure_collection(dimensions.pop())
            vector_store.upsert(chunks, vectors)
            chunk_count = _point_count(vector_store)
        completed.update(_beir_doc_id(document) for document in batch)
        _write_manifest(
            target,
            _manifest_payload(
                status="running",
                fingerprint=fingerprint,
                completed=completed,
                document_count=len(documents),
                chunk_count=chunk_count,
                chunk_id_digest=expected_digest,
            ),
        )
        if progress is not None:
            progress(len(completed), len(documents))

    actual_count = _point_count(vector_store)
    if actual_count != chunk_count:
        raise ValueError(
            f"indexed chunk count mismatch: manifest={chunk_count}, qdrant={actual_count}"
        )
    if completed != set(document_ids):
        raise ValueError("index did not complete every expected document")
    if _stored_chunk_ids(vector_store) != expected_chunk_ids:
        raise ValueError("index does not contain the exact expected chunk identities")
    final = _manifest_payload(
        status="complete",
        fingerprint=fingerprint,
        completed=completed,
        document_count=len(documents),
        chunk_count=chunk_count,
        chunk_id_digest=expected_digest,
    )
    _write_manifest(target, final)
    return IndexBuildResult(False, len(documents), chunk_count, fingerprint)


def validate_completed_index(
    vector_store: BenchmarkVectorStore,
    manifest: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> None:
    """Verify ownership, count, and exact stable chunk identity before reuse."""
    if manifest.get("status") != "complete":
        raise ValueError("SciFact index is incomplete")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("SciFact index configuration fingerprint mismatch")
    expected_count = _required_int(manifest, "chunk_count")
    actual_ids = _stored_chunk_ids(vector_store)
    if len(actual_ids) != expected_count:
        raise ValueError(
            "completed index point count mismatch: "
            f"manifest={expected_count}, qdrant={len(actual_ids)}"
        )
    expected_digest = manifest.get("chunk_id_digest")
    if not isinstance(expected_digest, str) or _id_digest(actual_ids) != expected_digest:
        raise ValueError("completed index chunk identity digest mismatch")


def _beir_doc_id(document: Document) -> str:
    value = document.metadata.get("beir_doc_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every indexed document must contain metadata['beir_doc_id']")
    return value


def _point_count(store: Any) -> int:
    direct = getattr(store, "point_count", None)
    if callable(direct):
        return int(direct())
    client = getattr(store, "client", None)
    collection_name = getattr(store, "collection_name", None)
    if client is None or not isinstance(collection_name, str):
        raise TypeError("benchmark vector store must expose point_count or Qdrant client")
    exists = getattr(store, "collection_exists", lambda: True)()
    if not exists:
        return 0
    return int(client.count(collection_name=collection_name, exact=True).count)


def _stored_chunk_ids(store: Any) -> set[str]:
    points = getattr(store, "points", None)
    if isinstance(points, Mapping):
        return {str(point_id) for point_id in points}
    iterator = getattr(store, "iter_chunks", None)
    if callable(iterator):
        return {str(chunk.id) for chunk in iterator()}
    raise TypeError("benchmark vector store must expose iter_chunks for integrity checks")


def _id_digest(chunk_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(chunk_ids)).encode("utf-8")).hexdigest()


def _completed_ids(manifest: Mapping[str, Any] | None) -> list[str]:
    if manifest is None:
        return []
    values = manifest.get("completed_document_ids", [])
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("invalid completed_document_ids in index manifest")
    return values


def _required_int(manifest: Mapping[str, Any], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {name} in index manifest")
    return value


def _manifest_payload(
    *,
    status: str,
    fingerprint: str,
    completed: set[str],
    document_count: int,
    chunk_count: int,
    chunk_id_digest: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "fingerprint": fingerprint,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "chunk_id_digest": chunk_id_digest,
        "completed_document_ids": sorted(completed),
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("index manifest must be a JSON object")
    return value


def _write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
