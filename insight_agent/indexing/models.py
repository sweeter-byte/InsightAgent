"""Data models and deterministic identifiers for the indexing layer."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from insight_agent.ingestion import Document, SourceType


_DOCUMENT_NAMESPACE = uuid.UUID("a1501ea5-7b55-5c0c-8665-72b87666d4f7")
_CHUNK_NAMESPACE = uuid.UUID("3f4c3d7d-497b-5a2b-b83f-05688fcada18")


@dataclass(slots=True)
class Chunk:
    """A position-aware text segment derived from one ingested Document."""

    id: str
    document_id: str
    content: str
    source: str
    source_type: SourceType
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    """Serialize JSON-compatible provenance with stable key ordering."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def make_document_id(document: Document) -> str:
    """Return a deterministic UUID for all identity-bearing Document fields."""
    identity = {
        "content": document.content,
        "metadata": document.metadata,
        "source": document.source,
        "source_type": document.source_type.value,
    }
    return str(uuid.uuid5(_DOCUMENT_NAMESPACE, _canonical_json(identity)))


def make_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Return a deterministic UUID for one chunk within a Document."""
    identity = {
        "content": content,
        "document_id": document_id,
        "chunk_index": chunk_index,
    }
    return str(uuid.uuid5(_CHUNK_NAMESPACE, _canonical_json(identity)))
