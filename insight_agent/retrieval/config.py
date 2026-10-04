"""Environment-backed configuration for hybrid retrieval."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_DENSE_K = 20
DEFAULT_SPARSE_K = 20
DEFAULT_RERANK_K = 20
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    """Validated stage depths and model selection for hybrid retrieval."""

    dense_k: int
    sparse_k: int
    rerank_k: int
    final_top_k: int
    rrf_k: int
    reranker_model: str

    def __post_init__(self) -> None:
        for name in ("dense_k", "sparse_k", "rerank_k", "rrf_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.final_top_k, bool)
            or not isinstance(self.final_top_k, int)
            or not 1 <= self.final_top_k <= 8
        ):
            raise ValueError("final_top_k must be an integer between 1 and 8")
        if (
            not isinstance(self.reranker_model, str)
            or not self.reranker_model.strip()
        ):
            raise ValueError("reranker_model must not be empty")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> HybridRetrievalConfig:
        """Build configuration from the process environment."""
        values = os.environ if environ is None else environ
        reranker_model = values.get("RERANKER_MODEL", "").strip()
        if not reranker_model:
            raise RuntimeError(
                "Missing required environment variable 'RERANKER_MODEL'. "
                "Set it to a cross-encoder model identifier in your shell or "
                "`.env` file (see `.env.example`)."
            )

        final_top_k = _positive_int(
            values,
            "HYBRID_FINAL_TOP_K",
            DEFAULT_FINAL_TOP_K,
        )
        if final_top_k > 8:
            raise RuntimeError(
                "HYBRID_FINAL_TOP_K must be an integer between 1 and 8; "
                f"received {final_top_k!r}."
            )

        return cls(
            dense_k=_positive_int(values, "HYBRID_DENSE_K", DEFAULT_DENSE_K),
            sparse_k=_positive_int(values, "HYBRID_SPARSE_K", DEFAULT_SPARSE_K),
            rerank_k=_positive_int(values, "HYBRID_RERANK_K", DEFAULT_RERANK_K),
            final_top_k=final_top_k,
            rrf_k=_positive_int(values, "HYBRID_RRF_K", DEFAULT_RRF_K),
            reranker_model=reranker_model,
        )


def _positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be a positive integer; received {raw_value!r}."
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"{name} must be a positive integer; received {raw_value!r}."
        )
    return value
