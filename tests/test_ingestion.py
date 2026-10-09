"""Tests for the Chapter 3 Multimodal Ingestion layer.

Covers: Document model, per-format loaders, and the converged public entry
points (``ingest``, ``ingest_file``, ``ingest_text_file``, ``infer_file_type``,
``safe_ingest``).

Entry-point tests import from ``insight_agent.ingestion`` on purpose — that is
the package surface upper-layer code is allowed to depend on.

No real network calls, no LLM/vision API — all HTTP and VLM interactions are
mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insight_agent.ingestion import (
    Document,
    IngestionError,
    SourceType,
    ingest,
    ingest_file,
    ingest_text_file,
    infer_file_type,
    load_image,
    load_markdown,
    load_pdf,
    load_text,
    load_text_file,
    load_url,
    safe_ingest,
)


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


def test_load_pdf_page_text_failure_is_wrapped_and_document_closed(tmp_path: Path) -> None:
    pdf_path = tmp_path / "page-error.pdf"
    pdf_path.write_bytes(b"placeholder")
    original_error = RuntimeError("page text extraction failed")

    page = MagicMock()
    page.get_text.side_effect = original_error
    pdf_document = MagicMock()
    pdf_document.page_count = 1
    pdf_document.__getitem__.return_value = page

    with patch("pymupdf.open", return_value=pdf_document):
        with pytest.raises(IngestionError, match="Failed to parse PDF") as excinfo:
            load_pdf(str(pdf_path))

    assert excinfo.value.__cause__ is original_error
    pdf_document.close.assert_called_once_with()


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
# 6. Unified entry point: ingest(source_type, source)
# ---------------------------------------------------------------------------


def test_ingest_explicit_text_is_inline_content() -> None:
    """``text`` dispatches to load_text: the payload *is* the content."""
    docs = ingest(SourceType.TEXT, "InsightAgent chapter three")
    assert len(docs) == 1
    assert docs[0].content == "InsightAgent chapter three"
    assert docs[0].source_type is SourceType.TEXT


def test_ingest_text_accepts_string_source_type() -> None:
    docs = ingest("text", "plain string type")
    assert docs[0].source_type is SourceType.TEXT
    assert docs[0].content == "plain string type"


def test_ingest_explicit_markdown_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nbody text", encoding="utf-8")

    docs = ingest(SourceType.MARKDOWN, str(f))
    assert docs[0].source_type is SourceType.MARKDOWN
    assert docs[0].content.startswith("# Heading")


def test_ingest_explicit_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(["ingest pdf content"], pdf_path)

    docs = ingest(SourceType.PDF, str(pdf_path))
    assert len(docs) == 1
    assert "ingest pdf content" in docs[0].content
    assert docs[0].source_type is SourceType.PDF


def test_ingest_explicit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """URL ingestion stays mocked — no real network traffic."""
    fake_resp = _FakeResponse(
        "<html><head><title>URL</title></head><body>text</body></html>",
        "https://example.org/page",
    )
    monkeypatch.setattr(
        "insight_agent.ingestion.loaders.url_loader.httpx.get",
        lambda *a, **kw: fake_resp,
    )

    docs = ingest(SourceType.URL, "https://example.org/page")
    assert docs[0].source_type is SourceType.URL
    assert docs[0].metadata["title"] == "URL"


def test_ingest_explicit_image(tmp_path: Path) -> None:
    """Image ingestion keeps the VLM call mocked — no real vision API."""
    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="a bar chart",
    ):
        docs = ingest(SourceType.IMAGE, str(png))

    assert docs[0].source_type is SourceType.IMAGE
    assert docs[0].content == "a bar chart"


def test_ingest_unknown_source_type_raises() -> None:
    with pytest.raises(IngestionError, match="Unknown source_type"):
        ingest("audio", "some payload")


@pytest.mark.parametrize("bad_type", ["", "TEXT", "docx", 42, None])
def test_ingest_rejects_non_source_type_values(bad_type: Any) -> None:
    with pytest.raises(IngestionError, match="Unknown source_type"):
        ingest(bad_type, "payload")


def test_ingest_propagates_loader_error(tmp_path: Path) -> None:
    """A missing file is reported by the loader, unchanged by the dispatcher."""
    with pytest.raises(IngestionError, match="File not found"):
        ingest(SourceType.PDF, str(tmp_path / "ghost.pdf"))


# ---------------------------------------------------------------------------
# 7. File-type inference (extension only — no magic-byte sniffing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.txt", SourceType.TEXT),
        ("README.md", SourceType.MARKDOWN),
        ("README.markdown", SourceType.MARKDOWN),
        ("paper.pdf", SourceType.PDF),
        ("chart.png", SourceType.IMAGE),
        ("photo.jpg", SourceType.IMAGE),
        ("photo.jpeg", SourceType.IMAGE),
        ("banner.webp", SourceType.IMAGE),
    ],
)
def test_infer_file_type_by_extension(filename: str, expected: SourceType) -> None:
    assert infer_file_type(f"/tmp/{filename}") is expected


def test_infer_file_type_is_case_insensitive() -> None:
    assert infer_file_type("PAPER.PDF") is SourceType.PDF


def test_infer_file_type_unknown_extension_raises() -> None:
    with pytest.raises(IngestionError, match="Unrecognized file extension"):
        infer_file_type("/tmp/data.xyz")


def test_infer_file_type_without_extension_raises() -> None:
    with pytest.raises(IngestionError, match="Unrecognized file extension"):
        infer_file_type("/tmp/Makefile")


@pytest.mark.parametrize("extension", ["gif", "bmp", "svg"])
def test_infer_file_type_rejects_unsupported_image_formats(extension: str) -> None:
    with pytest.raises(IngestionError, match="Unrecognized file extension"):
        infer_file_type(f"/tmp/image.{extension}")


# ---------------------------------------------------------------------------
# 8. ingest_file() — path in, type inferred from the suffix
# ---------------------------------------------------------------------------


def test_ingest_file_txt_reads_file_content(tmp_path: Path) -> None:
    """A .txt path must go through load_text_file, not be treated as text."""
    f = tmp_path / "doc.txt"
    f.write_text("hello from ingest_file", encoding="utf-8")

    docs = ingest_file(str(f))
    assert len(docs) == 1
    assert docs[0].content == "hello from ingest_file"
    assert docs[0].content != str(f)
    assert docs[0].source_type is SourceType.TEXT
    assert docs[0].metadata["filename"] == "doc.txt"


def test_ingest_file_markdown(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Heading\n\nbody", encoding="utf-8")

    docs = ingest_file(str(f))
    assert docs[0].source_type is SourceType.MARKDOWN
    assert docs[0].content.startswith("# Heading")


def test_ingest_file_markdown_long_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.markdown"
    f.write_text("long ext", encoding="utf-8")

    assert ingest_file(str(f))[0].source_type is SourceType.MARKDOWN


def test_ingest_file_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(["page one", "page two"], pdf_path)

    docs = ingest_file(str(pdf_path))
    assert len(docs) == 2
    assert all(d.source_type is SourceType.PDF for d in docs)


@pytest.mark.parametrize("name", ["fig.png", "fig.jpg", "fig.jpeg", "fig.webp"])
def test_ingest_file_image_suffixes(tmp_path: Path, name: str) -> None:
    f = _make_png(tmp_path, name=name)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        return_value="an image",
    ):
        docs = ingest_file(str(f))

    assert len(docs) == 1
    assert docs[0].source_type is SourceType.IMAGE
    assert docs[0].source == str(f)


def test_ingest_file_missing_vision_config_raises_ingestion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    for name in ("VISION_API_KEY", "VISION_BASE_URL", "VISION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    image = _make_png(tmp_path)

    with pytest.raises(IngestionError, match="Vision configuration error") as excinfo:
        ingest_file(str(image))

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_ingest_file_unknown_extension_raises(tmp_path: Path) -> None:
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK\x03\x04")

    with pytest.raises(IngestionError, match="Unrecognized file extension"):
        ingest_file(str(f))


def test_ingest_file_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="File not found"):
        ingest_file(str(tmp_path / "gone.txt"))


def test_ingest_text_file_matches_load_text_file(tmp_path: Path) -> None:
    f = tmp_path / "plain.txt"
    f.write_text("text file body", encoding="utf-8")

    docs = ingest_text_file(str(f))
    assert docs[0].content == "text file body"
    assert docs[0].source_type is SourceType.TEXT


# ---------------------------------------------------------------------------
# 9. safe_ingest() — minimal error boundary, original cause preserved
# ---------------------------------------------------------------------------


def test_safe_ingest_passes_through_on_success() -> None:
    docs = safe_ingest(SourceType.TEXT, "unchanged")
    assert docs[0].content == "unchanged"


def test_safe_ingest_wraps_unexpected_error_and_keeps_cause() -> None:
    boom = TypeError("unexpected loader crash")

    with patch("insight_agent.ingestion.ingest.load_text", side_effect=boom):
        with pytest.raises(IngestionError, match="Ingestion failed for") as excinfo:
            safe_ingest(SourceType.TEXT, "payload")

    # exception chaining keeps the original context reachable
    assert excinfo.value.__cause__ is boom


def test_safe_ingest_wraps_third_party_error(tmp_path: Path) -> None:
    """A raw OSError from below the loader layer still becomes IngestionError."""
    with patch("insight_agent.ingestion.ingest.load_pdf", side_effect=OSError("disk")):
        with pytest.raises(IngestionError) as excinfo:
            safe_ingest(SourceType.PDF, str(tmp_path / "doc.pdf"))

    assert isinstance(excinfo.value.__cause__, OSError)


def test_safe_ingest_does_not_double_wrap_ingestion_error() -> None:
    with patch(
        "insight_agent.ingestion.ingest.load_text",
        side_effect=IngestionError("loader-level failure"),
    ):
        with pytest.raises(IngestionError, match="loader-level failure") as excinfo:
            safe_ingest(SourceType.TEXT, "payload")

    assert "Ingestion failed for" not in str(excinfo.value)


def test_safe_ingest_unknown_source_type_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError, match="Unknown source_type"):
        safe_ingest("audio", "payload")


# ---------------------------------------------------------------------------
# 10. IngestionError hierarchy
# ---------------------------------------------------------------------------


def test_ingestion_error_is_runtime_error() -> None:
    assert issubclass(IngestionError, RuntimeError)


# ---------------------------------------------------------------------------
# 11. Image Loader (VLM call is always mocked — no real vision API)
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "chart.png") -> Path:
    """Create a placeholder image file. Content is irrelevant because the VLM
    call is mocked and MIME inference is extension-based."""
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return f


def test_load_image_returns_document(tmp_path: Path) -> None:
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
    with pytest.raises(IngestionError, match="File not found"):
        load_image(str(tmp_path / "absent.png"))


def test_load_image_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Not a regular file"):
        load_image(str(tmp_path))


def test_load_image_failure_propagates(tmp_path: Path) -> None:
    """A vision-model failure must bubble up as an IngestionError."""
    png = _make_png(tmp_path)
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        side_effect=IngestionError("Vision model call failed"),
    ):
        with pytest.raises(IngestionError, match="Vision model call failed"):
            load_image(str(png))


def test_safe_ingest_wraps_image_failure(tmp_path: Path) -> None:
    """Through the error boundary, a VLM crash keeps its original cause."""
    png = _make_png(tmp_path)
    crash = RuntimeError("vision endpoint unreachable")
    with patch(
        "insight_agent.ingestion.loaders.image_loader.describe_image",
        side_effect=crash,
    ):
        with pytest.raises(IngestionError) as excinfo:
            safe_ingest(SourceType.IMAGE, str(png))

    assert excinfo.value.__cause__ is crash
