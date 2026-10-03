"""Image ingestion loader — stub for Chapter 3.

Real vision OCR/captioning will be implemented in Chapter 4.
This module provides the interface so downstream code can import it
without breaking when the implementation lands.
"""

from __future__ import annotations

from insight_agent.ingestion.models import Document


def load_image(path: str) -> list[Document]:
    """Load an image file (placeholder — raises NotImplementedError).

    Args:
        path: filesystem path to an image file.

    Raises:
        NotImplementedError: always, until Chapter 4 vision support is added.
    """
    raise NotImplementedError(
        "Image loading with vision extraction is scheduled for Chapter 4."
    )
