"""Pure adapters from existing retrieval results to unified Evidence."""

from __future__ import annotations

from insight_agent.evidence.ids import make_evidence_id
from insight_agent.evidence.models import Evidence
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource
from insight_agent.vision_retrieval import VisionRetrievalResult
from insight_agent.web_search import WebRetrievalResult


class EvidenceCollector:
    """Convert successful retrieval material and deduplicate exact identities."""

    @staticmethod
    def collect_local(
        task_id: str,
        results: list[RetrievalResult],
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        for result in results:
            metadata = {
                **result.metadata,
                "retrieval_score": result.score,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "source_type": result.source_type.value,
                "chunk_index": result.chunk_index,
                "start_char": result.start_char,
                "end_char": result.end_char,
            }
            evidence.append(
                Evidence(
                    id=make_evidence_id(
                        task_id,
                        RetrievalSource.LOCAL,
                        result.chunk_id,
                    ),
                    task_id=task_id,
                    retrieval_source=RetrievalSource.LOCAL,
                    origin_id=result.chunk_id,
                    content=result.content,
                    source=result.source,
                    metadata=metadata,
                )
            )
        return evidence

    @staticmethod
    def collect_web(
        task_id: str,
        result: WebRetrievalResult,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        hit_by_rank = {hit.rank: hit for hit in result.hits}
        for document in result.documents:
            if not document.content.strip():
                continue
            final_url = document.metadata.get("final_url")
            origin_id = (
                final_url.strip()
                if isinstance(final_url, str) and final_url.strip()
                else document.source
            )
            metadata = dict(document.metadata)
            hit = hit_by_rank.get(metadata.get("search_rank"))
            if hit is not None:
                metadata.setdefault("original_url", hit.url)
            evidence.append(
                Evidence(
                    id=make_evidence_id(
                        task_id,
                        RetrievalSource.WEB,
                        origin_id,
                    ),
                    task_id=task_id,
                    retrieval_source=RetrievalSource.WEB,
                    origin_id=origin_id,
                    content=document.content,
                    source=document.source,
                    metadata=metadata,
                )
            )
        return evidence

    @staticmethod
    def collect_vision(
        task_id: str,
        result: VisionRetrievalResult,
    ) -> list[Evidence]:
        return [
            Evidence(
                id=make_evidence_id(
                    task_id,
                    RetrievalSource.VISION,
                    analysis.source,
                ),
                task_id=task_id,
                retrieval_source=RetrievalSource.VISION,
                origin_id=analysis.source,
                content=analysis.content,
                source=analysis.source,
                metadata=dict(analysis.metadata),
            )
            for analysis in result.analyses
        ]

    @staticmethod
    def merge(existing: list[Evidence], incoming: list[Evidence]) -> list[Evidence]:
        """Append new identities in order while keeping the first exact match."""
        merged = list(existing)
        seen_ids = {item.id for item in existing}
        for item in incoming:
            if item.id in seen_ids:
                continue
            merged.append(item)
            seen_ids.add(item.id)
        return merged
