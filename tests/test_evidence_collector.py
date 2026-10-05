"""Tests for deterministic, provenance-preserving Evidence collection."""

from __future__ import annotations

from insight_agent.evidence import Evidence, EvidenceCollector, make_evidence_id
from insight_agent.ingestion import Document, SourceType
from insight_agent.retrieval import RetrievalResult
from insight_agent.routing import RetrievalSource
from insight_agent.vision_retrieval import (
    VisionAnalysis,
    VisionFailure,
    VisionRetrievalResult,
)
from insight_agent.web_search import (
    WebFetchFailure,
    WebRetrievalResult,
    WebSearchHit,
)


def _local_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-7",
        score=0.8125,
        content="Local chunk content",
        document_id="document-3",
        source="notes/research.md",
        source_type=SourceType.MARKDOWN,
        chunk_index=2,
        start_char=120,
        end_char=260,
        metadata={"section": "Evidence", "retrieval_score": "stale"},
    )


def _web_result(*, documents: list[Document] | None = None) -> WebRetrievalResult:
    return WebRetrievalResult(
        query="current evidence collection",
        hits=[
            WebSearchHit(
                rank=1,
                title="Candidate title",
                url="https://example.com/original",
                snippet="Candidate snippet",
            )
        ],
        documents=(
            [
                Document(
                    content="Fetched page body",
                    source="https://example.com/final",
                    source_type=SourceType.URL,
                    metadata={
                        "title": "Fetched title",
                        "final_url": "https://example.com/final",
                        "content_type": "text/html",
                        "search_query": "current evidence collection",
                        "search_rank": 1,
                        "search_title": "Candidate title",
                    },
                )
            ]
            if documents is None
            else documents
        ),
        failures=[
            WebFetchFailure(
                url="https://example.com/broken",
                reason="HTTP 503",
            )
        ],
    )


def _vision_result() -> VisionRetrievalResult:
    return VisionRetrievalResult(
        task_id="T1",
        query="Inspect the workflow",
        analyses=[
            VisionAnalysis(
                source="images/workflow.png",
                content="Three visible branches converge.",
                metadata={"mime_type": "image/png"},
            )
        ],
        failures=[
            VisionFailure(source="images/missing.png", reason="not accessible")
        ],
    )


def test_local_result_becomes_one_evidence_with_chunk_provenance() -> None:
    result = _local_result()

    evidence = EvidenceCollector().collect_local("T1", [result])

    assert len(evidence) == 1
    item = evidence[0]
    assert item.task_id == "T1"
    assert item.retrieval_source is RetrievalSource.LOCAL
    assert item.origin_id == "chunk-7"
    assert item.content == "Local chunk content"
    assert item.source == "notes/research.md"
    assert item.metadata == {
        "section": "Evidence",
        "retrieval_score": 0.8125,
        "chunk_id": "chunk-7",
        "document_id": "document-3",
        "source_type": "markdown",
        "chunk_index": 2,
        "start_char": 120,
        "end_char": 260,
    }
    assert result.metadata == {"section": "Evidence", "retrieval_score": "stale"}


def test_web_collects_documents_only_and_preserves_url_metadata() -> None:
    result = _web_result()

    evidence = EvidenceCollector().collect_web("T2", result)

    assert len(evidence) == 1
    item = evidence[0]
    assert item.task_id == "T2"
    assert item.retrieval_source is RetrievalSource.WEB
    assert item.origin_id == "https://example.com/final"
    assert item.source == "https://example.com/final"
    assert item.content == "Fetched page body"
    assert item.metadata == {
        **result.documents[0].metadata,
        "original_url": "https://example.com/original",
    }
    assert item.metadata is not result.documents[0].metadata
    assert "original_url" not in result.documents[0].metadata
    assert all(item.origin_id != hit.url for hit in result.hits)
    assert all(item.origin_id != failure.url for failure in result.failures)


def test_web_falls_back_to_document_source_when_final_url_is_absent() -> None:
    document = Document(
        content="Fetched body",
        source="https://example.com/source",
        source_type=SourceType.URL,
        metadata={"title": "Title"},
    )

    evidence = EvidenceCollector().collect_web(
        "T1",
        _web_result(documents=[document]),
    )

    assert evidence[0].origin_id == "https://example.com/source"


def test_web_skips_documents_without_body_content() -> None:
    empty_document = Document(
        content="   ",
        source="https://example.com/empty",
        source_type=SourceType.URL,
        metadata={"final_url": "https://example.com/empty", "search_rank": 1},
    )

    evidence = EvidenceCollector().collect_web(
        "T1",
        _web_result(documents=[empty_document]),
    )

    assert evidence == []


def test_vision_analysis_keeps_original_image_as_source() -> None:
    result = _vision_result()

    evidence = EvidenceCollector().collect_vision("T1", result)

    assert len(evidence) == 1
    item = evidence[0]
    assert item.task_id == "T1"
    assert item.retrieval_source is RetrievalSource.VISION
    assert item.origin_id == "images/workflow.png"
    assert item.source == "images/workflow.png"
    assert item.content == "Three visible branches converge."
    assert item.metadata == {"mime_type": "image/png"}


def test_evidence_id_is_stable_and_isolated_by_task_and_source() -> None:
    first = make_evidence_id("T1", RetrievalSource.LOCAL, "shared-origin")

    assert first == make_evidence_id(
        "T1", RetrievalSource.LOCAL, "shared-origin"
    )
    assert first != make_evidence_id(
        "T2", RetrievalSource.LOCAL, "shared-origin"
    )
    assert first != make_evidence_id(
        "T1", RetrievalSource.WEB, "shared-origin"
    )


def test_merge_deduplicates_exact_ids_but_keeps_distinct_sources() -> None:
    duplicate = EvidenceCollector().collect_local("T1", [_local_result()])[0]
    same_text_from_web = Evidence(
        id=make_evidence_id("T1", RetrievalSource.WEB, duplicate.origin_id),
        task_id="T1",
        retrieval_source=RetrievalSource.WEB,
        origin_id=duplicate.origin_id,
        content=duplicate.content,
        source=duplicate.source,
        metadata={},
    )

    merged = EvidenceCollector.merge(
        [duplicate],
        [duplicate, same_text_from_web],
    )

    assert merged == [duplicate, same_text_from_web]


def test_empty_source_results_produce_no_evidence() -> None:
    collector = EvidenceCollector()

    assert collector.collect_local("T1", []) == []
    assert collector.collect_web("T1", _web_result(documents=[])) == []
    assert collector.collect_vision(
        "T1",
        VisionRetrievalResult(
            task_id="T1",
            query="Inspect",
            analyses=[],
            failures=[],
            no_candidates=True,
        ),
    ) == []
