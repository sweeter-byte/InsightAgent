"""Ingestion context helpers.

Small utilities shared across loaders — currently just source-type inference
from file extensions. No class hierarchies, no DI.

Inference is deliberately *extension-only*: no magic-number sniffing, no
libmagic dependency. Callers that need more robust detection should look at the
file themselves before handing a path to a loader.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
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


def infer_file_type(path: str) -> SourceType:
    """Infer the SourceType from a file path's extension.

    Args:
        path: filesystem path string.

    Returns:
        The mapped SourceType.

    Raises:
        IngestionError: if the extension is unrecognized.
    """
    suffix = Path(path).suffix.lower()
    st = _EXT_MAP.get(suffix)
    if st is None:
        supported = ", ".join(sorted(_EXT_MAP))
        raise IngestionError(
            f"Unrecognized file extension: {suffix!r} (from {path!r}); "
            f"supported extensions: {supported}"
        )
    return st
