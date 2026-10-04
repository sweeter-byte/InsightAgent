"""Vision-language extraction: image → textual semantic description.

This module turns a single image file into a *faithful* text description using
an OpenAI-compatible vision-language model (VLM). The description is what makes
an image searchable / analyzable by the downstream text-only pipeline.

Scope (Chapter 3):
  * One image in, one text description out (``describe_image``).
  * No OCR pipeline, no PDF-page image handling, no evidence generation,
    no embeddings or vector indexing.

The visual description is a *derived representation* of the original image.
Nothing here modifies, replaces, or deletes the source file — the original
image stays on disk and remains reachable through the ``source`` path that the
caller records on the resulting Document.

Configuration reuses the project's single env-var mechanism (see
``insight_agent.llm._require_env``). Vision is configured independently of the
text model on purpose — they may come from different providers later:
  * ``VISION_API_KEY``
  * ``VISION_BASE_URL``
  * ``VISION_MODEL``
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from openai import OpenAI

from insight_agent.ingestion.errors import IngestionError
from insight_agent.llm import _require_env

# Explicit MIME map for the formats this stage targets. Kept minimal on purpose
# — we do NOT pull in a heavyweight imaging library just to sniff rare formats.
_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_VISION_USER_TEXT = "Describe this image faithfully, following the guidelines above."

# The prompt deliberately avoids a vague "describe this image". It pins down the
# behaviors the research pipeline depends on: fidelity, chart-reading, and text
# preservation, with an explicit anti-hallucination rule.
VISION_SYSTEM_PROMPT = (
    "You are a meticulous research assistant that converts images into faithful "
    "textual descriptions so they can be searched and reasoned about as text.\n"
    "\n"
    "Guidelines:\n"
    "1. Describe ONLY what is actually present in the image. Never invent, "
    "guess, or extrapolate details that are not visible.\n"
    "2. Preserve all research-relevant information: entities, relationships, "
    "labels, quantities, units, and reported results.\n"
    "3. For charts and plots, state the axes (labels, units, and ranges), the "
    "overall trend, any comparisons between series, and the key numeric values "
    "or data points.\n"
    "4. For diagrams and schematics, describe the components and how they "
    "connect or interact.\n"
    "5. Transcribe important visible text verbatim: titles, axis labels, "
    "legends, annotations, formulas, and table cell contents.\n"
    "6. Be specific and structured; prefer clear, concrete statements over "
    "vague generalities."
)


def guess_mime_type(path: str) -> str:
    """Infer an image MIME type from its file extension.

    Only the explicitly supported PNG, JPG, JPEG, and WEBP extensions are
    accepted.

    Args:
        path: filesystem path to an image.

    Returns:
        A MIME type string such as ``"image/png"``.

    Raises:
        IngestionError: if no MIME type can be determined for the path.
    """
    suffix = Path(path).suffix.lower()
    mime = _MIME_BY_EXT.get(suffix)
    if mime is None:
        raise IngestionError(f"Unsupported or unrecognized image format: {path!r}")
    return mime


def _encode_data_url(path: Path, mime: str) -> str:
    """Read image bytes and return a base64 ``data:`` URL.

    The VLM is fed the image inline as a Data URL, so no file upload or local
    server is involved and the original file is only ever read, never altered.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"Failed to read image {path}: {exc}") from exc
    if not raw:
        raise IngestionError(f"Image file is empty: {path}")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_text(response: Any) -> str:
    """Pull the assistant message content out of an SDK response.

    Tolerates malformed or empty responses by returning ``""`` rather than
    raising — an empty VLM answer must not crash the ingestion structure.
    """
    try:
        content = response.choices[0].message.content
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


def describe_image(path: str) -> str:
    """Describe an image's visual semantics via a vision-language model.

    Args:
        path: filesystem path to an image file.

    Returns:
        The model's textual description. May be an empty string if the model
        returns no content, but the call itself will not crash on that.

    Raises:
        IngestionError: if the file is missing, unreadable, has an unsupported
            format, is empty, lacks required vision configuration, or the
            vision model call fails.
    """
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {path}")
    if not p.is_file():
        raise IngestionError(f"Not a regular file: {path}")

    mime = guess_mime_type(path)
    data_url = _encode_data_url(p, mime)

    try:
        api_key = _require_env("VISION_API_KEY")
        base_url = _require_env("VISION_BASE_URL")
        model = _require_env("VISION_MODEL")
    except RuntimeError as exc:
        raise IngestionError(f"Vision configuration error: {exc}") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _VISION_USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as exc:  # openai.SDKError and network/time
        raise IngestionError(f"Vision model call failed for {path}: {exc}") from exc

    return _extract_text(response)
