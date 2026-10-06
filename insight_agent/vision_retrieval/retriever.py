"""Coordination from indexed image descriptions to original-image analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from insight_agent.ingestion import SourceType
from insight_agent.planning.models import ResearchTask
from insight_agent.retrieval import RetrievalResult
from insight_agent.vision_retrieval.errors import VisionRetrievalError
from insight_agent.vision_retrieval.models import (
    VisionAnalysis,
    VisionFailure,
    VisionRetrievalResult,
)


class ImageCandidateRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked candidates from the shared Hybrid Retriever."""
        ...


class TaskVisionAnalyzer(Protocol):
    def analyze(
        self,
        *,
        image_path: str,
        question: str,
        objective: str,
        constraints: list[str],
    ) -> str:
        """Return task-focused visible information from one image."""
        ...


class VisionRetriever:
    """Retrieve image descriptions, restore sources, and analyze each image."""

    def __init__(
        self,
        hybrid_retriever: ImageCandidateRetriever,
        analyzer: TaskVisionAnalyzer,
        *,
        candidate_k: int = 8,
        vision_top_k: int = 3,
    ) -> None:
        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or not 1 <= candidate_k <= 8
        ):
            raise ValueError("candidate_k must be an integer between 1 and 8")
        if (
            isinstance(vision_top_k, bool)
            or not isinstance(vision_top_k, int)
            or not 1 <= vision_top_k <= candidate_k
        ):
            raise ValueError(
                "vision_top_k must be a positive integer not greater than candidate_k"
            )
        self.hybrid_retriever = hybrid_retriever
        self.analyzer = analyzer
        self.candidate_k = candidate_k
        self.vision_top_k = vision_top_k

    def retrieve(
        self,
        *,
        task: ResearchTask,
        query: str | None = None,
        objective: str,
        constraints: list[str],
    ) -> VisionRetrievalResult:
        if not isinstance(task, ResearchTask):
            raise VisionRetrievalError("task must be a ResearchTask")
        effective_query = task.question if query is None else query
        if not isinstance(effective_query, str) or not effective_query.strip():
            raise VisionRetrievalError("query must be a non-empty string")
        try:
            candidates = self.hybrid_retriever.retrieve(
                effective_query,
                top_k=self.candidate_k,
                source_types={SourceType.IMAGE.value},
            )
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            raise VisionRetrievalError(
                f"Image candidate retrieval failed for task {task.id}: {reason}"
            ) from exc

        if not candidates:
            return VisionRetrievalResult(
                task_id=task.id,
                query=effective_query,
                analyses=[],
                failures=[],
                no_candidates=True,
            )

        unique_candidates = _deduplicate_by_source(candidates)
        analyses: list[VisionAnalysis] = []
        failures: list[VisionFailure] = []
        for candidate in unique_candidates[: self.vision_top_k]:
            source = candidate.source
            path = Path(source)
            if not path.is_file():
                failures.append(
                    VisionFailure(
                        source=source,
                        reason="Original image is not accessible as a regular file",
                    )
                )
                continue
            try:
                content = self.analyzer.analyze(
                    image_path=source,
                    question=effective_query,
                    objective=objective,
                    constraints=list(constraints),
                )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Vision analyzer returned empty content")
            except Exception as exc:
                failures.append(
                    VisionFailure(
                        source=source,
                        reason=str(exc).strip() or type(exc).__name__,
                    )
                )
                continue
            analyses.append(
                VisionAnalysis(
                    source=source,
                    content=content.strip(),
                    metadata=dict(candidate.metadata),
                )
            )

        return VisionRetrievalResult(
            task_id=task.id,
            query=effective_query,
            analyses=analyses,
            failures=failures,
            no_candidates=False,
        )


def _deduplicate_by_source(
    candidates: list[RetrievalResult],
) -> list[RetrievalResult]:
    unique: list[RetrievalResult] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        if candidate.source_type is not SourceType.IMAGE:
            raise VisionRetrievalError(
                "Image-filtered retrieval returned a non-image candidate"
            )
        if candidate.source in seen_sources:
            continue
        seen_sources.add(candidate.source)
        unique.append(candidate)
    return unique
