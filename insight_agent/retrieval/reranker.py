"""Cross-encoder reranking behind a small retrieval-owned interface."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from numbers import Real
from typing import Any, Protocol

from insight_agent.retrieval.models import RankedResult


class Reranker(Protocol):
    """Score a fused candidate pool against one query."""

    def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        ...


def _load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Lazily load and batch a real sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        model_factory: Callable[[str], Any] = _load_cross_encoder,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        """Score Query–Chunk pairs, sort descending, and return final top-k."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not candidates:
            return []

        pairs = [(query, candidate.result.content) for candidate in candidates]
        raw_scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if not isinstance(values, (list, tuple)) or len(values) != len(candidates):
            raise RuntimeError(
                "cross-encoder must return one scalar score per candidate"
            )

        scores: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(
                    "cross-encoder must return one scalar score per candidate"
                )
            scores.append(float(value))

        scored = [
            replace(candidate, rerank_score=scores[index])
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(
            key=lambda candidate: (
                -(candidate.rerank_score if candidate.rerank_score is not None else 0.0)
            )
        )
        return scored[:top_k]

    def _get_model(self) -> Any:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(self.model_name)
        return self._model
