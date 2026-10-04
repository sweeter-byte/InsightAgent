"""PDF ingestion loader using PyMuPDF.

Opens a PDF, iterates pages, and produces one Document per non-empty page.
Page numbers in metadata are 1-indexed (human-readable).

No OCR, no scanned-PDF handling — that is out of Chapter 3 scope.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType


def load_pdf(path: str) -> list[Document]:
    """Read a PDF file and return one Document per non-empty page.

    Args:
        path: filesystem path to a ``.pdf`` file.

    Returns:
        ``list[Document]`` with ``source_type=PDF``.

    Raises:
        IngestionError: if the file does not exist, cannot be opened, or fails
            during page access or text extraction.
    """
    import pymupdf  # noqa: PLC0415 — deferred to avoid import cost for non-PDF users

    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if not p.is_file():
        raise IngestionError(f"Not a regular file: {path}")

    try:
        doc = pymupdf.open(str(p))
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF {path}: {exc}") from exc

    try:
        documents: list[Document] = []
        page_count = doc.page_count
        for page_idx in range(page_count):
            page = doc[page_idx]
            text: str = page.get_text("text", sort=True)
            if not text.strip():
                continue
            documents.append(
                Document(
                    content=text,
                    source=str(p),
                    source_type=SourceType.PDF,
                    metadata={
                        "filename": p.name,
                        "page": page_idx + 1,  # 1-indexed human page number
                        "page_count": page_count,
                    },
                )
            )
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF {path}: {exc}") from exc
    finally:
        doc.close()

    return documents
