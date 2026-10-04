"""Orchestration for the indexing write path only."""

from __future__ import annotations

from insight_agent.ingestion import Document
from insight_agent.indexing.chunker import TextChunker
from insight_agent.indexing.embedder import Embedder
from insight_agent.indexing.vector_store import VectorStore


def index_documents(
    documents: list[Document],
    chunker: TextChunker,
    embedder: Embedder,
    vector_store: VectorStore,
) -> int:
    """Chunk, embed, validate, and persist Documents; return chunks processed."""
    chunks = chunker.split_documents(documents)
    if not chunks:
        return 0

    texts = [chunk.content for chunk in chunks]
    vectors = embedder.embed_documents(texts)
    if len(chunks) != len(vectors):
        raise ValueError(
            "embedding count mismatch: "
            f"chunks={len(chunks)}, vectors={len(vectors)}"
        )

    vector_size: int | None = None
    for index, vector in enumerate(vectors):
        if not vector:
            raise ValueError(f"embedding vector at index {index} is empty")
        if vector_size is None:
            vector_size = len(vector)
        elif len(vector) != vector_size:
            raise ValueError(
                "inconsistent embedding dimensions: "
                f"expected={vector_size}, index {index} has dimension {len(vector)}"
            )

    # ``chunks`` is non-empty and counts match, so at least one non-empty vector
    # was inspected above.
    assert vector_size is not None
    vector_store.ensure_collection(vector_size)
    vector_store.upsert(chunks, vectors)
    return len(chunks)
