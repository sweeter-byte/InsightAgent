"""Markdown file ingestion loader.

Reads a ``.md`` file and preserves the original Markdown structure (headings,
lists, code blocks, etc.) — no stripping or reformatting is performed.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType


def load_markdown(path: str) -> list[Document]:
    """Read a UTF-8 Markdown file and return it as a single Document.

    Args:
        path: filesystem path to a ``.md`` file.

    Returns:
        ``[Document]`` with ``source_type=MARKDOWN`` and filename in metadata.

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
            source_type=SourceType.MARKDOWN,
            metadata={"filename": p.name},
        )
    ]
