"""Image ingestion loader.

Turns a single image file into one :class:`Document` whose ``content`` is the
vision-language model's textual description of the image (see
``insight_agent.ingestion.vision.describe_image``).

Design notes:
  * The VLM description is a *derived representation* of the image — the
    original file is only read, never moved, rewritten, or deleted. The
    Document's ``source`` keeps the original image path so downstream code can
    always trace back to the real file.
  * Exactly one Document is produced per image. No chunking, no embeddings,
    no OCR pipeline, no PDF-page images (all out of Chapter 3 scope).
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType
from insight_agent.ingestion.vision import (
    OpenAICompatibleVisionClient,
    describe_image,
    describe_image_with_client,
    guess_mime_type,
)


def load_image(
    path: str,
    *,
    vision_client: OpenAICompatibleVisionClient | None = None,
) -> list[Document]:
    """Load an image and return its visual description as a single Document.

    Args:
        path: filesystem path to a PNG, JPG, JPEG, or WEBP image file.

    Returns:
        ``[Document]`` with ``source_type=IMAGE``. ``content`` holds the VLM
        description, ``source`` the original image path, and ``metadata`` at
        least ``{"filename": ..., "mime_type": ...}``.

    Raises:
        IngestionError: if the file is missing, not a regular file, has an
            unsupported format, or the vision model call fails.
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if not p.is_file():
        raise IngestionError(f"Not a regular file: {path}")

    mime_type = guess_mime_type(path)
    # describe_image owns the actual VLM call and raises IngestionError on
    # failure; an empty description still yields a structurally valid Document.
    description = (
        describe_image(str(p))
        if vision_client is None
        else describe_image_with_client(str(p), vision_client)
    )

    return [
        Document(
            content=description,
            source=str(p),
            source_type=SourceType.IMAGE,
            metadata={"filename": p.name, "mime_type": mime_type},
        )
    ]
