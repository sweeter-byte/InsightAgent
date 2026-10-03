"""Unified ingestion entry point.

``ingest()`` is the single function external code should call. It dispatches
to the appropriate loader based on input type or file extension and always
returns ``list[Document]``.

This module does NOT do chunking, embedding, or retrieval.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.context import detect_source_type
from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.loaders.markdown_loader import load_markdown
from insight_agent.ingestion.loaders.pdf_loader import load_pdf
from insight_agent.ingestion.loaders.text_loader import load_text_file
from insight_agent.ingestion.loaders.url_loader import load_url
from insight_agent.ingestion.models import Document, SourceType


def ingest(source: str) -> list[Document]:
    """Ingest a single input and return its Document list.

    Dispatch logic:
      * If *source* looks like a URL (starts with ``http://`` or ``https://``),
        fetch via URL loader.
      * If *source* is an existing local file path, dispatch by extension.
      * Otherwise raise IngestionError.

    Args:
        source: a URL string or a local filesystem path.

    Returns:
        A list of Documents extracted from the source.

    Raises:
        IngestionError: if the source cannot be resolved or loaded.
    """
    # URL
    if source.startswith(("http://", "https://")):
        return load_url(source)

    # Local file
    p = Path(source)
    if p.exists() and p.is_file():
        try:
            st = detect_source_type(source)
        except ValueError as exc:
            raise IngestionError(str(exc)) from exc

        if st is SourceType.TEXT:
            return load_text_file(source)
        if st is SourceType.MARKDOWN:
            return load_markdown(source)
        if st is SourceType.PDF:
            return load_pdf(source)
        if st is SourceType.IMAGE:
            from insight_agent.ingestion.loaders.image_loader import load_image  # noqa: PLC0415

            return load_image(source)

    raise IngestionError(f"Cannot ingest source: {source!r}")
