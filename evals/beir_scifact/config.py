"""Central configuration for the SciFact retrieval benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from insight_agent.indexing.embedder import DEFAULT_EMBEDDING_MODEL


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True, slots=True)
class BenchmarkPaths:
    root: Path = Path(".data/benchmarks/beir_scifact")
    collection_name: str = "beir_scifact_chunks"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")

    @property
    def archive_path(self) -> Path:
        return self.root / "downloads" / "scifact.zip"

    @property
    def dataset_parent(self) -> Path:
        return self.root / "datasets"

    @property
    def dataset_path(self) -> Path:
        return self.dataset_parent / "scifact"

    @property
    def qdrant_path(self) -> Path:
        return self.root / "qdrant"

    @property
    def index_manifest_path(self) -> Path:
        return self.root / "index-manifest.json"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_batch_size: int = 32
    document_batch_size: int = 64
    dense_k: int = 50
    sparse_k: int = 50
    rerank_k: int = 100
    document_top_k: int = 10
    rrf_k: int = 60
    reranker_batch_size: int = 16

    def __post_init__(self) -> None:
        if not self.embedding_model.strip() or not self.reranker_model.strip():
            raise ValueError("model names must not be empty")
        positive = (
            "chunk_size",
            "embedding_batch_size",
            "document_batch_size",
            "dense_k",
            "sparse_k",
            "rerank_k",
            "document_top_k",
            "rrf_k",
            "reranker_batch_size",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    def index_identity(self) -> dict[str, object]:
        values = asdict(self)
        return {
            name: values[name]
            for name in ("embedding_model", "chunk_size", "chunk_overlap")
        }
