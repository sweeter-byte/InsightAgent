"""Ingestion context helpers.

Small utilities shared across loaders — currently just source-type inference
from file extensions. No class hierarchies, no DI.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.models import SourceType

_EXT_MAP: dict[str, SourceType] = {
    ".txt": SourceType.TEXT,
    ".text": SourceType.TEXT,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
    ".png": SourceType.IMAGE,
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
    ".bmp": SourceType.IMAGE,
    ".gif": SourceType.IMAGE,
}


def detect_source_type(path: str) -> SourceType:
    """Infer the SourceType from a file path's extension.

    Args:
        path: filesystem path string.

    Returns:
        The mapped SourceType.

    Raises:
        ValueError: if the extension is unrecognized.
    """
    suffix = Path(path).suffix.lower()
    st = _EXT_MAP.get(suffix)
    if st is None:
        raise ValueError(f"Unrecognized file extension: {suffix!r} (from {path!r})")
    return st
