"""Vision extraction interface — stub for Chapter 3.

Chapter 4 will implement actual image-to-text extraction (OCR, captioning,
or VLM-based description). This module defines the call signature so the
image_loader can wire into it without forward-reference issues.
"""

from __future__ import annotations


def extract_text_from_image(path: str) -> str:
    """Extract text from an image file (placeholder).

    Args:
        path: filesystem path to an image.

    Returns:
        Extracted text string.

    Raises:
        NotImplementedError: always, until Chapter 4.
    """
    raise NotImplementedError(
        "Vision extraction is scheduled for Chapter 4."
    )
