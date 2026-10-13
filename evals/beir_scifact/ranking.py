"""Collapse chunk rankings into deterministic BEIR document rankings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from insight_agent.retrieval import RetrievalResult


@dataclass(frozen=True, slots=True)
class DocumentRankingEntry:
    document_id: str
    chunk_id: str
    rank: int
    chunk_score: float


def collapse_chunks_to_documents(
    chunks: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> tuple[DocumentRankingEntry, ...]:
    """Keep the first (highest-ranked) chunk for each official document ID."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    seen: set[str] = set()
    documents: list[DocumentRankingEntry] = []
    for chunk in chunks:
        doc_id = chunk.metadata.get("beir_doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(
                f"retrieved chunk {chunk.chunk_id!r} is missing metadata['beir_doc_id']"
            )
        if doc_id in seen:
            continue
        seen.add(doc_id)
        documents.append(
            DocumentRankingEntry(
                document_id=doc_id,
                chunk_id=chunk.chunk_id,
                rank=len(documents) + 1,
                chunk_score=float(chunk.score),
            )
        )
        if len(documents) == top_k:
            break
    return tuple(documents)


def rankings_to_beir_results(
    rankings: Mapping[str, Sequence[DocumentRankingEntry]],
) -> dict[str, dict[str, float]]:
    """Encode rank as unique descending scores so BEIR cannot reorder ties."""
    converted: dict[str, dict[str, float]] = {}
    for query_id, entries in rankings.items():
        count = len(entries)
        converted[query_id] = {
            entry.document_id: float(count - index)
            for index, entry in enumerate(entries)
        }
    return converted
