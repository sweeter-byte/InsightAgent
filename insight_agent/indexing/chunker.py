"""Simple character-based chunking that preserves Document boundaries."""

from __future__ import annotations

import os
from copy import deepcopy

from insight_agent.ingestion import Document
from insight_agent.indexing.models import Chunk, make_chunk_id, make_document_id


_BOUNDARIES = ("\n\n", "\n", "。", ".")


def _configured_int(name: str, fallback: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


DEFAULT_CHUNK_SIZE = _configured_int("CHUNK_SIZE", 1200)
DEFAULT_CHUNK_OVERLAP = _configured_int("CHUNK_OVERLAP", 200)


class TextChunker:
    """Split each Document independently into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split Documents in order without merging their provenance boundaries."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        document_id = make_document_id(document)
        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            target_end = min(start + self.chunk_size, len(text))
            end = self._choose_end(text, start, target_end)
            content = text[start:end]
            metadata = deepcopy(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunks.append(
                Chunk(
                    id=make_chunk_id(document_id, chunk_index, content),
                    document_id=document_id,
                    content=content,
                    source=document.source,
                    source_type=document.source_type,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata,
                )
            )

            if end == len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
            chunk_index += 1

        return chunks

    def _choose_end(self, text: str, start: int, target_end: int) -> int:
        if target_end == len(text):
            return target_end

        minimum_end = start + max(1, self.chunk_size // 2)
        for boundary in _BOUNDARIES:
            position = text.rfind(boundary, minimum_end, target_end)
            if position != -1:
                candidate_end = position + len(boundary)
                if candidate_end - start > self.chunk_overlap:
                    return candidate_end
        return target_end
