"""Query embedding, vector search, and payload restoration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from insight_agent.indexing import Embedder
from insight_agent.ingestion import SourceType
from insight_agent.retrieval.models import RetrievalPayloadError, RetrievalResult


class SearchableVectorStore(Protocol):
    """Read-only vector-store capability consumed by ``VectorRetriever``."""

    def search(self, vector: list[float], limit: int = 5) -> list[Any]:
        """Return the nearest stored points with payloads attached."""
        ...


class VectorRetriever:
    """Retrieve indexed chunks from the embedding space used at index time."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: SearchableVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed one non-empty query and return up to ``top_k`` domain results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        vectors = self.embedder.embed_documents([query])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("query embedding must contain exactly one non-empty vector")

        points = self.vector_store.search(vectors[0], limit=top_k)
        return [self._to_result(point) for point in points]

    @staticmethod
    def _to_result(point: Any) -> RetrievalResult:
        payload = getattr(point, "payload", None)
        point_id = getattr(point, "id", None)
        score = getattr(point, "score", None)
        if point_id is None:
            raise RetrievalPayloadError("retrieved point is missing chunk id")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RetrievalPayloadError(
                f"retrieved point {point_id!r} has an invalid score"
            )
        if not isinstance(payload, Mapping):
            raise RetrievalPayloadError(
                f"retrieved point {point_id!r} is missing a payload"
            )

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
            raise RetrievalPayloadError(
                f"retrieved point {point_id!r} is missing required payload field(s): "
                f"{', '.join(missing)}"
            )

        for name in ("content", "document_id", "source", "source_type"):
            if not isinstance(payload[name], str):
                raise RetrievalPayloadError(
                    f"retrieved point {point_id!r} has invalid payload field {name!r}"
                )
        for name in ("chunk_index", "start_char", "end_char"):
            if isinstance(payload[name], bool) or not isinstance(payload[name], int):
                raise RetrievalPayloadError(
                    f"retrieved point {point_id!r} has invalid payload field {name!r}"
                )
        if not isinstance(payload["metadata"], Mapping):
            raise RetrievalPayloadError(
                f"retrieved point {point_id!r} has invalid payload field 'metadata'"
            )
        try:
            source_type = SourceType(payload["source_type"])
        except ValueError as exc:
            raise RetrievalPayloadError(
                f"retrieved point {point_id!r} has unknown source_type "
                f"{payload['source_type']!r}"
            ) from exc

        return RetrievalResult(
            chunk_id=str(point_id),
            score=float(score),
            content=payload["content"],
            document_id=payload["document_id"],
            source=payload["source"],
            source_type=source_type,
            chunk_index=payload["chunk_index"],
            start_char=payload["start_char"],
            end_char=payload["end_char"],
            metadata=dict(payload["metadata"]),
        )
