"""Centralized startup configuration for InsightAgent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from insight_agent.indexing import DEFAULT_EMBEDDING_MODEL, QdrantConfig
from insight_agent.ingestion.vision import VisionModelConfig
from insight_agent.llm import LLMConfig
from insight_agent.research.persistence import DEFAULT_CHECKPOINT_PATH
from insight_agent.retrieval.config import HybridRetrievalConfig
from insight_agent.runtime.policies import RuntimeConfig
from insight_agent.web_search.config import WebSearchConfig


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Validated location of the application checkpoint database."""

    path: Path = DEFAULT_CHECKPOINT_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("CHECKPOINT_PATH must not be empty")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable startup settings for one application instance."""

    llm: LLMConfig
    vision: VisionModelConfig | None
    retrieval: HybridRetrievalConfig
    embedding_model: str
    qdrant: QdrantConfig
    web_search: WebSearchConfig | None
    checkpoint: CheckpointConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if (
            not isinstance(self.embedding_model, str)
            or not self.embedding_model.strip()
        ):
            raise ValueError("embedding_model must not be empty")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> AppConfig:
        """Resolve all startup settings from one environment mapping."""
        if environ is None:
            load_dotenv()
            values = os.environ
        else:
            values = environ

        embedding_model = values.get(
            "EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ).strip()
        checkpoint_path = values.get(
            "CHECKPOINT_PATH",
            str(DEFAULT_CHECKPOINT_PATH),
        ).strip()

        return cls(
            llm=LLMConfig.from_env(values),
            vision=VisionModelConfig.from_env(values, required=False),
            retrieval=HybridRetrievalConfig.from_env(values),
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            qdrant=QdrantConfig.from_env(values),
            web_search=WebSearchConfig.from_env(values, required=False),
            checkpoint=CheckpointConfig(
                path=Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
            ),
            runtime=RuntimeConfig.from_env(values),
        )
