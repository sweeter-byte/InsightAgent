"""Unified ingestion entry points.

This module is the *only* ingestion surface upper-layer code should reach for:

  * :func:`ingest`       — dispatch by explicit :class:`SourceType`.
  * :func:`ingest_file`  — dispatch a local path by its file extension.
  * :func:`ingest_text_file` — convenience for plain-text files.
  * :func:`safe_ingest`  — minimal error boundary around :func:`ingest`.

Everything here returns ``list[Document]``. Loader modules under
``insight_agent.ingestion.loaders`` are internal implementation details: upper
layers should import from ``insight_agent.ingestion``, never from
``insight_agent.ingestion.loaders.pdf_loader`` and friends.

This module does NOT do chunking, embedding, retrieval, retries, or any form of
task-state management.
"""

from __future__ import annotations

from insight_agent.ingestion.context import infer_file_type
from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.loaders.image_loader import load_image
from insight_agent.ingestion.loaders.markdown_loader import load_markdown
from insight_agent.ingestion.loaders.pdf_loader import load_pdf
from insight_agent.ingestion.loaders.text_loader import load_text, load_text_file
from insight_agent.ingestion.loaders.url_loader import load_url
from insight_agent.ingestion.models import Document, SourceType

#: Accepted values for the ``source_type`` argument — the enum itself or its
#: string value (``SourceType`` is a ``str`` enum, so ``"pdf"`` works too).
SourceTypeLike = SourceType | str


def _coerce_source_type(source_type: SourceTypeLike) -> SourceType:
    """Normalize a caller-supplied source type into :class:`SourceType`.

    Raises:
        IngestionError: if the value is not a known source type.
    """
    if isinstance(source_type, SourceType):
        return source_type
    try:
        return SourceType(source_type)
    except ValueError as exc:
        supported = ", ".join(member.value for member in SourceType)
        raise IngestionError(
            f"Unknown source_type: {source_type!r}; supported types: {supported}"
        ) from exc


def ingest(source_type: SourceTypeLike, source: str) -> list[Document]:
    """Ingest one source with an explicitly given type.

    Dispatch table:
      * ``text``     → :func:`load_text`      (*source* is the inline text)
      * ``markdown`` → :func:`load_markdown`  (*source* is a file path)
      * ``pdf``      → :func:`load_pdf`       (*source* is a file path)
      * ``url``      → :func:`load_url`       (*source* is a URL)
      * ``image``    → :func:`load_image`     (*source* is a file path)

    For local files whose type is not known upfront, prefer :func:`ingest_file`.

    Args:
        source_type: the :class:`SourceType` (or its string value).
        source:      the payload — inline text, a file path, or a URL, depending
                     on ``source_type``.

    Returns:
        The Documents produced by the selected loader.

    Raises:
        IngestionError: if ``source_type`` is unknown, or if the selected loader
            fails.
    """
    st = _coerce_source_type(source_type)

    if st is SourceType.TEXT:
        return load_text(source)
    if st is SourceType.MARKDOWN:
        return load_markdown(source)
    if st is SourceType.PDF:
        return load_pdf(source)
    if st is SourceType.URL:
        return load_url(source)
    if st is SourceType.IMAGE:
        return load_image(source)

    # Unreachable while SourceType has exactly the five members above; kept so a
    # new enum member cannot silently fall through to a bare ``return []``.
    raise IngestionError(f"Unsupported source_type: {st.value!r}")


def ingest_file(path: str) -> list[Document]:
    """Ingest a local file, picking the loader from its extension.

    Text files go to :func:`load_text_file`, so the file's *contents* become the
    Document content — the path is never mistaken for inline text.

    Args:
        path: filesystem path to a supported file.

    Returns:
        The Documents produced by the selected loader.

    Raises:
        IngestionError: if the extension is unrecognized or the loader fails.
    """
    st = infer_file_type(path)

    if st is SourceType.TEXT:
        return load_text_file(path)
    if st is SourceType.MARKDOWN:
        return load_markdown(path)
    if st is SourceType.PDF:
        return load_pdf(path)
    if st is SourceType.IMAGE:
        return load_image(path)

    # SourceType.URL has no file-extension representation.
    raise IngestionError(f"Cannot ingest {path!r} as a file (type {st.value!r})")


def ingest_text_file(path: str) -> list[Document]:
    """Ingest a plain-text file (thin alias over :func:`load_text_file`)."""
    return load_text_file(path)


def safe_ingest(source_type: SourceTypeLike, source: str) -> list[Document]:
    """:func:`ingest` with a minimal error boundary.

    Any exception escaping the loaders — including unexpected ones from third-
    party libraries — is normalized into :class:`IngestionError` while the
    original exception is preserved as ``__cause__``, so callers can catch a
    single type without losing the traceback context.

    No retry, no backoff, no state recovery: failures surface immediately.

    Args:
        source_type: the :class:`SourceType` (or its string value).
        source:      see :func:`ingest`.

    Returns:
        Whatever :func:`ingest` returns.

    Raises:
        IngestionError: for every failure mode, chained to the root cause.
    """
    try:
        return ingest(source_type, source)
    except IngestionError:
        # Already the ingestion error type; re-raise to keep the loader's own
        # message and chain intact.
        raise
    except Exception as exc:
        raise IngestionError(f"Ingestion failed for {source!r}: {exc}") from exc
