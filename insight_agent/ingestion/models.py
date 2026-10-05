"""Unified Document model for the ingestion layer.

Every Loader in this package produces ``list[Document]`` regardless of the
input format. The Document is a thin, immutable-ish dataclass that carries
raw text content plus provenance metadata for downstream consumers (chunking,
embedding, retrieval — all out of Chapter 3 scope).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Identifies the origin format of a Document."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"
    URL = "url"
    IMAGE = "image"


def normalize_source_types(source_types: set[str] | None) -> set[str] | None:
    """Validate an optional set of ``SourceType`` string values."""
    if source_types is None:
        return None
    if not isinstance(source_types, set) or not source_types:
        raise ValueError("source_types must be a non-empty set of strings")
    if any(not isinstance(value, str) for value in source_types):
        raise ValueError("source_types must be a non-empty set of strings")
    allowed = {source_type.value for source_type in SourceType}
    unknown = source_types - allowed
    if unknown:
        raise ValueError(
            "source_types contains unknown value(s): "
            + ", ".join(sorted(repr(value) for value in unknown))
        )
    return set(source_types)


@dataclass(slots=True)
class Document:
    """A single unit of ingested content.

    Attributes:
        content:     the extracted text body.
        source:      a human-readable identifier for where this came from
                     (file path, URL, inline label, etc.).
        source_type: the format category of the input.
        metadata:    loader-specific key/value data (page numbers, titles, …).
    """

    content: str
    source: str
    source_type: SourceType
    metadata: dict[str, Any] = field(default_factory=dict)
