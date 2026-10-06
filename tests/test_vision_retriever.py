"""Control-flow tests for task-conditioned Vision Retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from insight_agent.ingestion import SourceType
from insight_agent.planning import ResearchTask
from insight_agent.retrieval import RetrievalResult
from insight_agent.vision_retrieval import (
    VisionAnalysis,
    VisionFailure,
    VisionRetrievalResult,
    VisionRetriever,
)


def _result(
    chunk_id: str,
    source: str,
    *,
    score: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        content=f"Indexed description {chunk_id}",
        document_id=f"doc-{chunk_id}",
        source=source,
        source_type=SourceType.IMAGE,
        chunk_index=0,
        start_char=0,
        end_char=10,
        metadata=dict(metadata or {}),
    )


class FakeHybridRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {"query": query, "top_k": top_k, "source_types": source_types}
        )
        return self.results[:top_k]


class FakeAnalyzer:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = set(failures or ())
        self.calls: list[dict[str, Any]] = []

    def analyze(
        self,
        *,
        image_path: str,
        question: str,
        objective: str,
        constraints: list[str],
    ) -> str:
        self.calls.append(
            {
                "image_path": image_path,
                "question": question,
                "objective": objective,
                "constraints": list(constraints),
            }
        )
        if image_path in self.failures:
            raise RuntimeError("VLM unavailable")
        return f"Analysis of {Path(image_path).name}"


def _task() -> ResearchTask:
    return ResearchTask(id="T2", question="Do the branches converge?")


def test_retriever_filters_deduplicates_sources_and_forwards_context(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    hybrid = FakeHybridRetriever(
        [
            _result("a-best", str(first), metadata={"rank": "best"}),
            _result("a-other", str(first), metadata={"rank": "other"}),
            _result("b", str(second), metadata={"rank": "second"}),
        ]
    )
    analyzer = FakeAnalyzer()
    retriever = VisionRetriever(
        hybrid_retriever=hybrid,
        analyzer=analyzer,
        candidate_k=8,
        vision_top_k=3,
    )

    result = retriever.retrieve(
        task=_task(),
        query="Which storage limitations are visible?",
        objective="Inspect topology",
        constraints=["Use visible arrows only"],
    )

    assert hybrid.calls == [
        {
            "query": "Which storage limitations are visible?",
            "top_k": 8,
            "source_types": {"image"},
        }
    ]
    assert [call["image_path"] for call in analyzer.calls] == [str(first), str(second)]
    assert analyzer.calls[0]["question"] == "Which storage limitations are visible?"
    assert analyzer.calls[0]["objective"] == "Inspect topology"
    assert analyzer.calls[0]["constraints"] == ["Use visible arrows only"]
    assert result.task_id == "T2"
    assert result.query == "Which storage limitations are visible?"
    assert [analysis.source for analysis in result.analyses] == [str(first), str(second)]
    assert result.analyses[0].metadata == {"rank": "best"}
    assert result.failures == []
    assert result.no_candidates is False


def test_no_candidates_is_distinct_from_analysis_failure() -> None:
    result = VisionRetriever(
        hybrid_retriever=FakeHybridRetriever([]),
        analyzer=FakeAnalyzer(),
    ).retrieve(task=_task(), objective="Inspect", constraints=[])

    assert result.no_candidates is True
    assert result.analyses == []
    assert result.failures == []


def test_retriever_defaults_to_task_question_when_query_is_omitted() -> None:
    hybrid = FakeHybridRetriever([])

    result = VisionRetriever(
        hybrid_retriever=hybrid,
        analyzer=FakeAnalyzer(),
    ).retrieve(task=_task(), objective="Inspect", constraints=[])

    assert hybrid.calls[0]["query"] == _task().question
    assert result.query == _task().question


@pytest.mark.parametrize(
    ("analyses", "failures"),
    [
        ([VisionAnalysis(source="diagram.png", content="visible")], []),
        ([], [VisionFailure(source="diagram.png", reason="unreadable")]),
    ],
)
def test_no_candidates_rejects_per_image_outcomes(
    analyses: list[VisionAnalysis],
    failures: list[VisionFailure],
) -> None:
    with pytest.raises(ValueError, match="no_candidates=True"):
        VisionRetrievalResult(
            task_id="T1",
            query="Inspect the diagram",
            analyses=analyses,
            failures=failures,
            no_candidates=True,
        )


def test_candidate_result_requires_an_analysis_or_failure() -> None:
    with pytest.raises(ValueError, match="at least one analysis or failure"):
        VisionRetrievalResult(
            task_id="T1",
            query="Inspect the diagram",
            analyses=[],
            failures=[],
            no_candidates=False,
        )


def test_missing_original_is_structured_failure_not_success(tmp_path: Path) -> None:
    missing = tmp_path / "deleted.png"
    analyzer = FakeAnalyzer()
    result = VisionRetriever(
        hybrid_retriever=FakeHybridRetriever([_result("missing", str(missing))]),
        analyzer=analyzer,
    ).retrieve(task=_task(), objective="Inspect", constraints=[])

    assert result.no_candidates is False
    assert result.analyses == []
    assert len(result.failures) == 1
    assert result.failures[0].source == str(missing)
    assert "not accessible" in result.failures[0].reason
    assert analyzer.calls == []


def test_one_image_failure_does_not_discard_other_analyses(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("a.png", "b.png", "c.png")]
    for path in paths:
        path.write_bytes(path.name.encode())
    analyzer = FakeAnalyzer(failures={str(paths[1])})
    result = VisionRetriever(
        hybrid_retriever=FakeHybridRetriever(
            [_result(path.stem, str(path)) for path in paths]
        ),
        analyzer=analyzer,
    ).retrieve(task=_task(), objective="Inspect", constraints=[])

    assert [analysis.source for analysis in result.analyses] == [
        str(paths[0]),
        str(paths[2]),
    ]
    assert len(result.failures) == 1
    assert result.failures[0].source == str(paths[1])
    assert "VLM unavailable" in result.failures[0].reason
    assert result.no_candidates is False


def test_empty_analyzer_response_becomes_structured_failure(tmp_path: Path) -> None:
    image = tmp_path / "empty.png"
    image.write_bytes(b"image")

    class EmptyAnalyzer(FakeAnalyzer):
        def analyze(self, **kwargs: Any) -> str:
            super().analyze(**kwargs)
            return "  "

    result = VisionRetriever(
        hybrid_retriever=FakeHybridRetriever([_result("empty", str(image))]),
        analyzer=EmptyAnalyzer(),
    ).retrieve(task=_task(), objective="Inspect", constraints=[])

    assert result.analyses == []
    assert len(result.failures) == 1
    assert "empty content" in result.failures[0].reason
