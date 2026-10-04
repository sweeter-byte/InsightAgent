"""Tests for deterministic document chunking and Chunk provenance."""

from __future__ import annotations

import importlib

import pytest

from insight_agent.ingestion import Document, SourceType
from insight_agent.indexing import TextChunker, make_document_id


def _document(
    content: str,
    *,
    source: str = "notes.txt",
    source_type: SourceType = SourceType.TEXT,
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        content=content,
        source=source,
        source_type=source_type,
        metadata=dict(metadata or {}),
    )


def test_short_document_produces_one_chunk() -> None:
    document = _document("short text")

    chunks = TextChunker(chunk_size=100, chunk_overlap=10).split_documents(
        [document]
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content == "short text"
    assert chunk.source == document.source
    assert chunk.source_type is SourceType.TEXT
    assert chunk.chunk_index == 0
    assert (chunk.start_char, chunk.end_char) == (0, len(document.content))


def test_long_document_is_split_with_overlap_and_valid_offsets() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = TextChunker(chunk_size=10, chunk_overlap=3).split_documents(
        [_document(text)]
    )

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert 0 <= chunk.start_char < chunk.end_char <= len(text)
        assert chunk.content == text[chunk.start_char : chunk.end_char]
        assert len(chunk.content) <= 10
    for previous, current in zip(chunks, chunks[1:]):
        assert current.start_char == previous.end_char - 3
        assert current.start_char > previous.start_char


def test_documents_are_split_independently_and_chunk_indexes_restart() -> None:
    documents = [
        _document("A" * 15, source="a.txt"),
        _document("B" * 15, source="b.txt"),
    ]

    chunks = TextChunker(chunk_size=10, chunk_overlap=2).split_documents(documents)

    a_chunks = [chunk for chunk in chunks if chunk.source == "a.txt"]
    b_chunks = [chunk for chunk in chunks if chunk.source == "b.txt"]
    assert [chunk.chunk_index for chunk in a_chunks] == list(range(len(a_chunks)))
    assert [chunk.chunk_index for chunk in b_chunks] == list(range(len(b_chunks)))
    assert a_chunks[0].document_id != b_chunks[0].document_id


def test_split_prefers_paragraph_boundary_without_tiny_chunks() -> None:
    text = "A" * 12 + "\n\n" + "B" * 12

    chunks = TextChunker(chunk_size=20, chunk_overlap=2).split_documents(
        [_document(text)]
    )

    assert chunks[0].end_char == 14
    assert chunks[0].content.endswith("\n\n")
    assert len(chunks[0].content) >= 10


def test_chunker_makes_progress_with_near_size_overlap() -> None:
    text = "0123456789" * 4

    chunks = TextChunker(chunk_size=10, chunk_overlap=9).split_documents(
        [_document(text)]
    )

    assert chunks[-1].end_char == len(text)
    assert all(
        current.start_char > previous.start_char
        for previous, current in zip(chunks, chunks[1:])
    )


def test_boundary_fallback_preserves_near_size_overlap() -> None:
    text = "aaaaa." + "b" * 20

    chunks = TextChunker(chunk_size=10, chunk_overlap=9).split_documents(
        [_document(text)]
    )

    assert chunks[0].end_char == 10
    assert chunks[1].start_char == 1
    assert chunks[0].end_char - chunks[1].start_char == 9


def test_default_chunk_configuration_can_come_from_environment(monkeypatch) -> None:
    import insight_agent.indexing.chunker as chunker_module

    monkeypatch.setenv("CHUNK_SIZE", "64")
    monkeypatch.setenv("CHUNK_OVERLAP", "8")
    reloaded = importlib.reload(chunker_module)
    try:
        chunker = reloaded.TextChunker()
        assert (chunker.chunk_size, chunker.chunk_overlap) == (64, 8)
    finally:
        monkeypatch.delenv("CHUNK_SIZE")
        monkeypatch.delenv("CHUNK_OVERLAP")
        importlib.reload(chunker_module)


@pytest.mark.parametrize("content", ["", "   \n\n\t"])
def test_blank_document_produces_no_chunks(content: str) -> None:
    assert TextChunker().split_documents([_document(content)]) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_invalid_chunk_size_is_rejected(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        TextChunker(chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 10, 11])
def test_invalid_overlap_is_rejected(overlap: int) -> None:
    with pytest.raises(
        ValueError, match="chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
    ):
        TextChunker(chunk_size=10, chunk_overlap=overlap)


def test_metadata_is_copied_with_chunk_offsets() -> None:
    document = _document(
        "abcdefghijklmno",
        metadata={"filename": "notes.txt", "nested": {"author": "Ada"}},
    )

    chunk = TextChunker(chunk_size=10, chunk_overlap=2).split_documents([document])[0]

    assert chunk.metadata is not document.metadata
    assert chunk.metadata["filename"] == "notes.txt"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["start_char"] == chunk.start_char
    assert chunk.metadata["end_char"] == chunk.end_char
    chunk.metadata["filename"] = "changed.txt"
    assert document.metadata["filename"] == "notes.txt"
    nested = chunk.metadata["nested"]
    assert isinstance(nested, dict)
    nested["author"] = "Grace"
    assert document.metadata["nested"] == {"author": "Ada"}


def test_pdf_metadata_is_preserved() -> None:
    document = _document(
        "page content " * 10,
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"filename": "paper.pdf", "page": 3, "page_count": 10},
    )

    chunks = TextChunker(chunk_size=30, chunk_overlap=5).split_documents([document])

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["filename"] == "paper.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["page_count"] == 10


def test_document_id_uses_canonical_metadata_and_distinguishes_pdf_pages() -> None:
    first_order = _document(
        "same",
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"page": 1, "filename": "paper.pdf", "page_count": 2},
    )
    second_order = _document(
        "same",
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"page_count": 2, "filename": "paper.pdf", "page": 1},
    )
    next_page = _document(
        "same",
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"filename": "paper.pdf", "page": 2, "page_count": 2},
    )

    assert make_document_id(first_order) == make_document_id(second_order)
    assert make_document_id(first_order) != make_document_id(next_page)


def test_same_input_and_configuration_produce_stable_chunk_ids() -> None:
    document = _document("0123456789" * 4, metadata={"filename": "notes.txt"})
    chunker = TextChunker(chunk_size=12, chunk_overlap=3)

    first = chunker.split_documents([document])
    second = chunker.split_documents([document])

    assert [chunk.document_id for chunk in first] == [
        chunk.document_id for chunk in second
    ]
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
