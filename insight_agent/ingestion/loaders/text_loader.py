"""Plain-text ingestion loader.

Two entry points:
  * ``load_text``  — accept a raw string (e.g. pasted from clipboard).
  * ``load_text_file`` — read a ``.txt`` file from disk.

Both return ``list[Document]`` with ``source_type=TEXT``.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType


def load_text(text: str, source: str = "inline:text") -> list[Document]:
    """Wrap a raw text string into a single-element Document list.

    Args:
        text:   the text content.
        source: an identifier for provenance tracking.

    Returns:
        ``[Document]`` with one entry.
    """
    return [
        Document(
            content=text,
            source=source,
            source_type=SourceType.TEXT,
            metadata={},
        )
    ]


def load_text_file(path: str) -> list[Document]:
    """Read a UTF-8 text file and return it as a single Document.

    Args:
        path: filesystem path to a ``.txt`` file.

    Returns:
        ``[Document]`` with filename in metadata.

    Raises:
        IngestionError: if the file does not exist or cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if not p.is_file():
        raise IngestionError(f"Not a regular file: {path}")
    try:
        content = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Failed to read {path}: {exc}") from exc

    return [
        Document(
            content=content,
            source=str(p),
            source_type=SourceType.TEXT,
            metadata={"filename": p.name},
        )
    ]
