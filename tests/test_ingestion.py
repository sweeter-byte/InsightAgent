"""Tests for the Chapter 3 Multimodal Ingestion layer.

Covers: Document model, Text Loader, Markdown Loader, PDF Loader, URL Loader,
and the unified ``ingest()`` entry point.

No real network calls, no LLM API — all HTTP interactions are mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.loaders.markdown_loader import load_markdown
from insight_agent.ingestion.loaders.pdf_loader import load_pdf
from insight_agent.ingestion.loaders.text_loader import load_text, load_text_file
from insight_agent.ingestion.loaders.url_loader import load_url
from insight_agent.ingestion.models import Document, SourceType


# ---------------------------------------------------------------------------
# 1. Document model
# ---------------------------------------------------------------------------


def test_document_creation_and_fields() -> None:
    doc = Document(
        content="hello world",
        source="test.txt",
        source_type=SourceType.TEXT,
        metadata={"filename": "test.txt"},
    )
    assert doc.content == "hello world"
    assert doc.source == "test.txt"
    assert doc.source_type is SourceType.TEXT
    assert doc.metadata == {"filename": "test.txt"}


def test_document_metadata_defaults_to_empty_dict() -> None:
    doc = Document(content="x", source="s", source_type=SourceType.TEXT)
    assert doc.metadata == {}


def test_source_type_enum_values() -> None:
    assert SourceType.TEXT == "text"
    assert SourceType.MARKDOWN == "markdown"
    assert SourceType.PDF == "pdf"
    assert SourceType.URL == "url"
    assert SourceType.IMAGE == "image"


# ---------------------------------------------------------------------------
# 2. Text Loader
# ---------------------------------------------------------------------------


def test_load_text_returns_document() -> None:
    docs = load_text("InsightAgent is great")
    assert len(docs) == 1
    assert docs[0].content == "InsightAgent is great"
    assert docs[0].source == "inline:text"
    assert docs[0].source_type is SourceType.TEXT
    assert docs[0].metadata == {}


def test_load_text_custom_source() -> None:
    docs = load_text("some text", source="custom:label")
    assert docs[0].source == "custom:label"


def test_load_text_file_reads_content(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("Method A: 81.3%\nMethod B: 87.6%\n", encoding="utf-8")

    docs = load_text_file(str(f))
    assert len(docs) == 1
    assert "87.6%" in docs[0].content
    assert docs[0].source_type is SourceType.TEXT
    assert docs[0].metadata["filename"] == "sample.txt"


def test_load_text_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="File not found"):
        load_text_file(str(tmp_path / "nope.txt"))


def test_load_text_file_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Not a regular file"):
        load_text_file(str(tmp_path))


# ---------------------------------------------------------------------------
# 3. Markdown Loader
# ---------------------------------------------------------------------------


def test_load_markdown_preserves_structure(tmp_path: Path) -> None:
    md_content = (
        "# Title\n\n"
        "## Section\n\n"
        "- item 1\n"
        "- item 2\n\n"
        "```python\nprint('hello')\n```\n"
    )
    f = tmp_path / "readme.md"
    f.write_text(md_content, encoding="utf-8")

    docs = load_markdown(str(f))
    assert len(docs) == 1
    assert docs[0].content == md_content
    assert docs[0].source_type is SourceType.MARKDOWN
    assert docs[0].metadata["filename"] == "readme.md"
    # Verify markdown structure is NOT stripped
    assert "# Title" in docs[0].content
    assert "```python" in docs[0].content
    assert "- item 1" in docs[0].content


def test_load_markdown_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="File not found"):
        load_markdown(str(tmp_path / "absent.md"))


# ---------------------------------------------------------------------------
# 4. PDF Loader
# ---------------------------------------------------------------------------


def _make_pdf(pages: list[str], path: Path) -> None:
    """Create a simple PDF with the given text on each page."""
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_load_pdf_multi_page_produces_documents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "multi.pdf"
    _make_pdf(["Page One Content", "Page Two Content", "Page Three Content"], pdf_path)

    docs = load_pdf(str(pdf_path))
    assert len(docs) == 3
    assert "Page One" in docs[0].content
    assert "Page Two" in docs[1].content
    assert "Page Three" in docs[2].content


def test_load_pdf_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "meta.pdf"
    _make_pdf(["First page", "Second page"], pdf_path)

    docs = load_pdf(str(pdf_path))
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["page_count"] == 2
    assert docs[0].metadata["filename"] == "meta.pdf"
    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["page_count"] == 2


def test_load_pdf_empty_pages_skipped(tmp_path: Path) -> None:
    """Pages with no text content must not generate a Document."""
    pdf_path = tmp_path / "sparse.pdf"
    _make_pdf(["Has text", "", "Also has text"], pdf_path)

    docs = load_pdf(str(pdf_path))
    assert len(docs) == 2
    # The page numbers should reflect the original position (1 and 3)
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 3


def test_load_pdf_all_empty_pages(tmp_path: Path) -> None:
    """A PDF with no text at all should return an empty list."""
    pdf_path = tmp_path / "blank.pdf"
    _make_pdf(["", "", ""], pdf_path)

    docs = load_pdf(str(pdf_path))
    assert docs == []


def test_load_pdf_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="File not found"):
        load_pdf(str(tmp_path / "ghost.pdf"))


def test_load_pdf_source_type(tmp_path: Path) -> None:
    pdf_path = tmp_path / "stype.pdf"
    _make_pdf(["content"], pdf_path)

    docs = load_pdf(str(pdf_path))
    assert docs[0].source_type is SourceType.PDF


# ---------------------------------------------------------------------------
# 5. URL Loader (mocked — no real network)
# ---------------------------------------------------------------------------


_SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head>
    <title>Test Page Title</title>
    <style>body { color: red; }</style>
    <script>var x = "should not appear";</script>
    <noscript>Enable JavaScript</noscript>
</head>
<body>
    <h1>Welcome</h1>
    <p>This is the main content paragraph.</p>
</body>
</html>
"""


class _FakeResponse:
    """Mimics the shape of httpx.Response for our loader's needs."""

    def __init__(self, text: str, url: str, status_code: int = 200) -> None:
        self.text = text
        self.url = url
        self.status_code = status_code
        self.headers: dict[str, str] = {"content-type": "text/html; charset=utf-8"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"Client error {self.status_code}",
                request=httpx.Request("GET", self.url),
                response=httpx.Response(self.status_code),
            )


def test_load_url_extracts_title_and_text() -> None:
    fake_resp = _FakeResponse(_SAMPLE_HTML, "https://example.com/page")

    with patch("insight_agent.ingestion.loaders.url_loader.httpx.get", return_value=fake_resp):
        docs = load_url("https://example.com/page")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_type is SourceType.URL
    assert doc.metadata["title"] == "Test Page Title"
    assert doc.metadata["final_url"] == "https://example.com/page"
    assert "content_type" in doc.metadata
    # Visible text is extracted
    assert "Welcome" in doc.content
    assert "main content paragraph" in doc.content
    # Script/style/noscript content is excluded
    assert "should not appear" not in doc.content
    assert "color: red" not in doc.content
    assert "Enable JavaScript" not in doc.content


def test_load_url_sets_correct_source() -> None:
    fake_resp = _FakeResponse("<html><body><p>hi</p></body></html>", "https://a.com/x")

    with patch("insight_agent.ingestion.loaders.url_loader.httpx.get", return_value=fake_resp):
        docs = load_url("https://a.com/x")

    assert docs[0].source == "https://a.com/x"


def test_load_url_http_error_raises_ingestion_error() -> None:
    import httpx

    fake_resp = _FakeResponse("", "https://example.com/missing", status_code=404)

    with patch("insight_agent.ingestion.loaders.url_loader.httpx.get", return_value=fake_resp):
        with pytest.raises(IngestionError, match="HTTP 404"):
            load_url("https://example.com/missing")


def test_load_url_network_error_raises_ingestion_error() -> None:
    import httpx

    def _connection_error(*args: Any, **kwargs: Any) -> Any:
        raise httpx.ConnectError("Connection refused")

    with patch("insight_agent.ingestion.loaders.url_loader.httpx.get", side_effect=_connection_error):
        with pytest.raises(IngestionError, match="Request failed"):
            load_url("https://unreachable.test/")


# ---------------------------------------------------------------------------
# 6. Unified ingest() dispatch
# ---------------------------------------------------------------------------


def test_ingest_text_file(tmp_path: Path) -> None:
    from insight_agent.ingestion.ingest import ingest

    f = tmp_path / "doc.txt"
    f.write_text("hello from ingest", encoding="utf-8")

    docs = ingest(str(f))
    assert len(docs) == 1
    assert docs[0].content == "hello from ingest"
    assert docs[0].source_type is SourceType.TEXT


def test_ingest_markdown_file(tmp_path: Path) -> None:
    from insight_agent.ingestion.ingest import ingest

    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nbody text", encoding="utf-8")

    docs = ingest(str(f))
    assert docs[0].source_type is SourceType.MARKDOWN


def test_ingest_pdf_file(tmp_path: Path) -> None:
    from insight_agent.ingestion.ingest import ingest

    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(["ingest pdf content"], pdf_path)

    docs = ingest(str(pdf_path))
    assert len(docs) == 1
    assert "ingest pdf content" in docs[0].content
    assert docs[0].source_type is SourceType.PDF


def test_ingest_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from insight_agent.ingestion.ingest import ingest

    fake_resp = _FakeResponse(
        "<html><head><title>URL</title></head><body>text</body></html>",
        "https://example.org/page",
    )
    monkeypatch.setattr(
        "insight_agent.ingestion.loaders.url_loader.httpx.get",
        lambda *a, **kw: fake_resp,
    )

    docs = ingest("https://example.org/page")
    assert docs[0].source_type is SourceType.URL
    assert docs[0].metadata["title"] == "URL"


def test_ingest_unknown_source_raises(tmp_path: Path) -> None:
    from insight_agent.ingestion.ingest import ingest

    with pytest.raises(IngestionError, match="Cannot ingest source"):
        ingest(str(tmp_path / "nonexistent.xyz"))


# ---------------------------------------------------------------------------
# 7. IngestionError hierarchy
# ---------------------------------------------------------------------------


def test_ingestion_error_is_runtime_error() -> None:
    assert issubclass(IngestionError, RuntimeError)


# ---------------------------------------------------------------------------
# 8. Image Loader (VLM call is always mocked — no real vision API)
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "chart.png") -> Path:
    """Create a placeholder image file. Content is irrelevant because the VLM
    call is mocked and MIME inference is extension-based."""
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return f


def test_load_image_returns_document(tmp_path: Path) -> None:
    from insight_agent.ingestion.loaders.image_loader import load_image

    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="A bar chart comparing method A (81.3%) and B (87.6%).",
    ):
        docs = load_image(str(png))

    assert len(docs) == 1
    assert docs[0].source_type is SourceType.IMAGE
    # describe_image() result lands in Document.content
    assert "87.6%" in docs[0].content


def test_load_image_source_and_metadata(tmp_path: Path) -> None:
    from insight_agent.ingestion.loaders.image_loader import load_image

    png = _make_png(tmp_path, name="fig.jpg")
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="desc",
    ):
        docs = load_image(str(png))

    doc = docs[0]
    # source preserves the original image path for traceability
    assert doc.source == str(png)
    assert doc.metadata["filename"] == "fig.jpg"
    assert doc.metadata["mime_type"] == "image/jpeg"


def test_load_image_does_not_touch_original(tmp_path: Path) -> None:
    """The description is derived; the original file must survive untouched."""
    from insight_agent.ingestion.loaders.image_loader import load_image

    png = _make_png(tmp_path)
    before = png.read_bytes()
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="desc",
    ):
        docs = load_image(str(png))

    assert png.exists() and png.is_file()
    assert png.read_bytes() == before
    assert docs[0].source == str(png)  # still traceable to the real file


def test_load_image_empty_response_does_not_crash(tmp_path: Path) -> None:
    from insight_agent.ingestion.loaders.image_loader import load_image

    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="",
    ):
        docs = load_image(str(png))

    assert len(docs) == 1
    assert docs[0].content == ""
    assert docs[0].source_type is SourceType.IMAGE


def test_load_image_missing_file_raises(tmp_path: Path) -> None:
    from insight_agent.ingestion.loaders.image_loader import load_image

    with pytest.raises(IngestionError, match="File not found"):
        load_image(str(tmp_path / "absent.png"))


def test_load_image_directory_raises(tmp_path: Path) -> None:
    from insight_agent.ingestion.loaders.image_loader import load_image

    with pytest.raises(IngestionError, match="Not a regular file"):
        load_image(str(tmp_path))


def test_load_image_failure_propagates(tmp_path: Path) -> None:
    """A vision-model failure must bubble up as an IngestionError."""
    from insight_agent.ingestion.loaders.image_loader import load_image

    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        side_effect=IngestionError("Vision model call failed"),
    ):
        with pytest.raises(IngestionError, match="Vision model call failed"):
            load_image(str(png))


def test_ingest_image_dispatch(tmp_path: Path) -> None:
    from insight_agent.ingestion.ingest import ingest

    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="an image",
    ):
        docs = ingest(str(png))

    assert docs[0].source_type is SourceType.IMAGE
