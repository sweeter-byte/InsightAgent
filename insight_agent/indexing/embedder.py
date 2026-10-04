"""Embedding protocol and a lazily loaded Sentence Transformers adapter."""

from __future__ import annotations

import os
from typing import Any, Protocol


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class Embedder(Protocol):
    """Minimal embedding dependency consumed by the indexing pipeline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts in input order."""
        ...


def _load_sentence_transformer_class() -> Any:
    """Import the optional heavyweight implementation only when first used."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


class SentenceTransformerEmbedder:
    """Batch encoder backed by a lazily instantiated SentenceTransformer."""

    def __init__(self, model_name: str | None = None) -> None:
        configured = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        self.model_name = model_name or configured or DEFAULT_EMBEDDING_MODEL
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return normalized embeddings as ordinary nested Python lists."""
        if not texts:
            return []

        model = self._get_model()
        encoded = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        return [[float(value) for value in vector] for vector in values]

    def _get_model(self) -> Any:
        if self._model is None:
            sentence_transformer = _load_sentence_transformer_class()
            self._model = sentence_transformer(self.model_name)
        return self._model
