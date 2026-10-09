"""Vector-store protocol and Qdrant Local Mode implementation."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from insight_agent.indexing.models import Chunk
from insight_agent.ingestion import SourceType
from insight_agent.ingestion.models import normalize_source_types


DEFAULT_QDRANT_PATH = ".data/qdrant"
DEFAULT_QDRANT_COLLECTION = "insight_documents"


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """Validated local Qdrant location and collection name."""

    path: Path = Path(DEFAULT_QDRANT_PATH)
    collection_name: str = DEFAULT_QDRANT_COLLECTION

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("QDRANT_PATH must not be empty")
        if (
            not isinstance(self.collection_name, str)
            or not self.collection_name.strip()
        ):
            raise ValueError("QDRANT_COLLECTION must not be empty")
        object.__setattr__(self, "path", Path(self.path))

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> QdrantConfig:
        """Resolve the existing Qdrant variables from a mapping."""
        values = os.environ if environ is None else environ
        path = values.get("QDRANT_PATH", DEFAULT_QDRANT_PATH).strip()
        collection = values.get(
            "QDRANT_COLLECTION",
            DEFAULT_QDRANT_COLLECTION,
        ).strip()
        return cls(
            path=Path(path or DEFAULT_QDRANT_PATH),
            collection_name=collection or DEFAULT_QDRANT_COLLECTION,
        )


class VectorStore(Protocol):
    """Minimal vector persistence and nearest-neighbor search dependency."""

    def ensure_collection(self, vector_size: int) -> None:
        """Create or validate storage for vectors of the requested size."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Persist chunks and their positionally corresponding vectors."""
        ...

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> list[ScoredPoint]:
        """Return nearest points with payloads, excluding stored vectors."""
        ...


class QdrantVectorStore:
    """Persistent local Qdrant collection using stable Chunk UUIDs."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
        *,
        config: QdrantConfig | None = None,
    ) -> None:
        if config is not None and (
            path is not None or collection_name is not None
        ):
            raise ValueError(
                "config cannot be combined with path or collection_name"
            )
        resolved = config if config is not None else QdrantConfig.from_env()
        self.path = Path(path) if path else resolved.path
        self.collection_name = collection_name or resolved.collection_name
        self.client = QdrantClient(path=str(self.path))

    def ensure_collection(self, vector_size: int) -> None:
        """Create a COSINE collection or reject an incompatible dimension."""
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than 0")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return

        collection = self.client.get_collection(self.collection_name)
        vectors_config = collection.config.params.vectors
        if isinstance(vectors_config, dict):
            raise ValueError(
                f"Collection {self.collection_name!r} uses named vectors, but "
                "this indexing pipeline requires one unnamed vector."
            )
        existing_size = vectors_config.size
        if existing_size != vector_size:
            raise ValueError(
                "Embedding model / vector dimension incompatibility for "
                f"collection {self.collection_name!r}: existing={existing_size}, "
                f"requested={vector_size}. Use a compatible embedding model or "
                "a different collection."
            )
        if vectors_config.distance != Distance.COSINE:
            raise ValueError(
                f"Vector distance incompatibility for collection "
                f"{self.collection_name!r}: existing={vectors_config.distance.value}, "
                f"required={Distance.COSINE.value}. Use a Cosine collection."
            )

    def collection_exists(self) -> bool:
        """Return whether this store's configured collection already exists."""
        return self.client.collection_exists(self.collection_name)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert one Qdrant point per Chunk without positional truncation."""
        if len(chunks) != len(vectors):
            raise ValueError(
                "chunk/vector count mismatch: "
                f"received {len(chunks)} chunks and {len(vectors)} vectors"
            )
        if not chunks:
            return

        points = [
            PointStruct(
                id=chunks[index].id,
                vector=vectors[index],
                payload=self._payload(chunks[index]),
            )
            for index in range(len(chunks))
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> list[ScoredPoint]:
        """Query the existing collection and return payload-bearing hits."""
        if not vector:
            raise ValueError("search vector must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("search limit must be a positive integer")
        normalized_types = normalize_source_types(source_types)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=_source_type_filter(normalized_types),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

    def load_chunks(
        self,
        batch_size: int = 100,
        *,
        source_types: set[str] | None = None,
    ) -> list[Chunk]:
        """Restore every stored point as a project-owned ``Chunk``."""
        return list(
            self.iter_chunks(
                batch_size=batch_size,
                source_types=source_types,
            )
        )

    def iter_chunks(
        self,
        batch_size: int = 100,
        *,
        source_types: set[str] | None = None,
    ) -> Iterator[Chunk]:
        """Yield stored chunks page by page, optionally filtering at Qdrant."""
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        normalized_types = normalize_source_types(source_types)

        offset: Any | None = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=_source_type_filter(normalized_types),
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            yield from (self._chunk_from_point(point) for point in points)
            if next_offset is None:
                return
            offset = next_offset

    def close(self) -> None:
        """Release local storage resources held by the Qdrant client."""
        self.client.close()

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, object]:
        return {
            "content": chunk.content,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "source_type": chunk.source_type.value,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _chunk_from_point(point: Any) -> Chunk:
        point_id = getattr(point, "id", None)
        payload = getattr(point, "payload", None)
        if point_id is None:
            raise ValueError("stored point is missing chunk id")
        if not isinstance(payload, Mapping):
            raise ValueError(f"stored point {point_id!r} is missing a payload")

        required = (
            "content",
            "document_id",
            "source",
            "source_type",
            "chunk_index",
            "start_char",
            "end_char",
            "metadata",
        )
        missing = [
            name for name in required if name not in payload or payload[name] is None
        ]
        if missing:
            raise ValueError(
                f"stored point {point_id!r} is missing required payload field(s): "
                f"{', '.join(missing)}"
            )

        for name in ("content", "document_id", "source", "source_type"):
            if not isinstance(payload[name], str):
                raise ValueError(
                    f"stored point {point_id!r} has invalid payload field {name!r}"
                )
        for name in ("chunk_index", "start_char", "end_char"):
            if isinstance(payload[name], bool) or not isinstance(payload[name], int):
                raise ValueError(
                    f"stored point {point_id!r} has invalid payload field {name!r}"
                )
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError(
                f"stored point {point_id!r} has invalid payload field 'metadata'"
            )
        try:
            source_type = SourceType(payload["source_type"])
        except ValueError as exc:
            raise ValueError(
                f"stored point {point_id!r} has unknown source_type "
                f"{payload['source_type']!r}"
            ) from exc

        return Chunk(
            id=str(point_id),
            document_id=payload["document_id"],
            content=payload["content"],
            source=payload["source"],
            source_type=source_type,
            chunk_index=payload["chunk_index"],
            start_char=payload["start_char"],
            end_char=payload["end_char"],
            metadata=dict(payload["metadata"]),
        )


def _source_type_filter(source_types: set[str] | None) -> Filter | None:
    if source_types is None:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="source_type",
                match=MatchAny(any=sorted(source_types)),
            )
        ]
    )
