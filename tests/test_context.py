"""Tests for the temporary context adapter (``documents_to_context``).

``documents_to_context`` lives in ``insight_agent.ingestion.context`` and is
imported through the package's public surface to also assert that the adapter is
exported where upper-layer code expects it.

These are pure-string tests: no loaders, no network, no vision/LLM API. They pin
down the *contract* of the adapter — verbatim, budget-free rendering — which is
exactly what distinguishes it from (and proves it is not) RAG.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion import (
    Document,
    SourceType,
    documents_to_context,
    ingest_file,
)


def _doc(content: str, source: str = "s", st: SourceType = SourceType.TEXT, **meta: object) -> Document:
    return Document(content=content, source=source, source_type=st, metadata=dict(meta))


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_list_renders_empty_string() -> None:
    assert documents_to_context([]) == ""


# ---------------------------------------------------------------------------
# Single document: exact, deterministic block
# ---------------------------------------------------------------------------


def test_single_document_exact_format() -> None:
    doc = _doc("hello world", source="notes.txt", filename="notes.txt")
    result = documents_to_context([doc])
    assert result == (
        "[Document 1]\n"
        "source: notes.txt\n"
        "source_type: text\n"
        "metadata: filename=notes.txt\n"
        "hello world"
    )


def test_source_type_uses_enum_value_not_repr() -> None:
    # A str-enum must render as its value ("pdf"), not "SourceType.PDF".
    doc = Document(content="c", source="x.pdf", source_type=SourceType.PDF, metadata={})
    assert "source_type: pdf\n" in documents_to_context([doc])


def test_empty_metadata_renders_empty_braces() -> None:
    doc = Document(content="body", source="inline:text", source_type=SourceType.TEXT, metadata={})
    assert "metadata: {}\nbody" in documents_to_context([doc])


# ---------------------------------------------------------------------------
# Metadata rendering: deterministic, non-string values supported
# ---------------------------------------------------------------------------


def test_metadata_keys_are_sorted_and_ints_render() -> None:
    doc = Document(
        content="page two text",
        source="paper.pdf",
        source_type=SourceType.PDF,
        metadata={"page": 2, "filename": "paper.pdf", "page_count": 3},
    )
    result = documents_to_context([doc])
    # Sorted keys → filename, page, page_count; int values stringify cleanly.
    assert "metadata: filename=paper.pdf, page=2, page_count=3" in result


# ---------------------------------------------------------------------------
# Multiple documents: numbering, order, separation
# ---------------------------------------------------------------------------


def test_multiple_documents_are_numbered_in_order() -> None:
    docs = [_doc("first", source="a.txt"), _doc("second", source="b.txt")]
    result = documents_to_context(docs)

    blocks = result.split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].startswith("[Document 1]\nsource: a.txt")
    assert blocks[1].startswith("[Document 2]\nsource: b.txt")
    # Original order is preserved (no ranking/reordering).
    assert result.index("[Document 1]") < result.index("[Document 2]")


# ---------------------------------------------------------------------------
# The "NOT RAG" contract: content is verbatim, never truncated or compressed
# ---------------------------------------------------------------------------


def test_content_is_preserved_verbatim_without_truncation() -> None:
    long_content = "x" * 5000 + "\nmulti\nline\ntail"
    doc = _doc(long_content)
    result = documents_to_context([doc])

    # No token budget / no compression: the entire body survives, byte for byte.
    assert result.endswith(long_content)
    assert "x" * 5000 in result


# ---------------------------------------------------------------------------
# Integration: real ingested Documents render through the adapter
# ---------------------------------------------------------------------------


def test_ingested_txt_file_renders_with_source_and_content(tmp_path: Path) -> None:
    f = tmp_path / "result.txt"
    f.write_text("Method B: 87.6%\n", encoding="utf-8")

    documents = ingest_file(str(f))
    result = documents_to_context(documents)

    assert f"[Document 1]\nsource: {f}\nsource_type: text" in result
    assert "metadata: filename=result.txt" in result
    assert "Method B: 87.6%" in result
