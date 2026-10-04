"""Vector-store protocol and Qdrant Local Mode implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from insight_agent.indexing.models import Chunk


DEFAULT_QDRANT_PATH = ".data/qdrant"
DEFAULT_QDRANT_COLLECTION = "insight_documents"


class VectorStore(Protocol):
    """Minimal vector persistence and nearest-neighbor search dependency."""

    def ensure_collection(self, vector_size: int) -> None:
        """Create or validate storage for vectors of the requested size."""
        ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Persist chunks and their positionally corresponding vectors."""
        ...

    def search(self, vector: list[float], limit: int = 5) -> list[ScoredPoint]:
        """Return nearest points with payloads, excluding stored vectors."""
        ...


class QdrantVectorStore:
    """Persistent local Qdrant collection using stable Chunk UUIDs."""

    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        configured_path = os.getenv("QDRANT_PATH", DEFAULT_QDRANT_PATH).strip()
        configured_collection = os.getenv(
            "QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION
        ).strip()
        self.path = Path(path or configured_path or DEFAULT_QDRANT_PATH)
        self.collection_name = (
            collection_name or configured_collection or DEFAULT_QDRANT_COLLECTION
        )
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

    def search(self, vector: list[float], limit: int = 5) -> list[ScoredPoint]:
        """Query the existing collection and return payload-bearing hits."""
        if not vector:
            raise ValueError("search vector must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("search limit must be a positive integer")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

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
