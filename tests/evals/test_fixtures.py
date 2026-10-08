from __future__ import annotations

import socket
from pathlib import Path

import pytest

from evals import EvalCase
from evals.fixtures import (
    RecordedFixtureError,
    RecordedSearchProvider,
    RecordedUrlLoader,
    resolve_vision_fixture_paths,
)
from insight_agent.ingestion import Document, SourceType
from insight_agent.web_search import WebSearchHit


def _hit(url: str = "https://fixture.invalid/page") -> WebSearchHit:
    return WebSearchHit(
        rank=1,
        title="Recorded title",
        url=url,
        snippet="Recorded snippet",
        score=0.9,
    )


def _document(url: str = "https://fixture.invalid/page") -> Document:
    return Document(
        content="Recorded page body.",
        source=url,
        source_type=SourceType.URL,
        metadata={"fixture": "web-v1"},
    )


def test_recorded_web_components_use_only_fixed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    provider = RecordedSearchProvider({"fixed query": [_hit()]})
    loader = RecordedUrlLoader({_hit().url: [_document()]})

    hits = provider.search("fixed query", limit=1)
    documents = loader(hits[0].url, timeout=1.0)

    assert hits == [_hit()]
    assert documents == [_document()]
    assert documents is not loader.documents_by_url[_hit().url]


def test_recorded_provider_rejects_unknown_query_and_invalid_limit() -> None:
    provider = RecordedSearchProvider({"fixed query": [_hit()]})

    with pytest.raises(RecordedFixtureError, match="not recorded"):
        provider.search("different query", limit=1)
    with pytest.raises(RecordedFixtureError, match="positive integer"):
        provider.search("fixed query", limit=0)


def test_recorded_url_loader_rejects_unknown_url() -> None:
    loader = RecordedUrlLoader({_hit().url: [_document()]})

    with pytest.raises(RecordedFixtureError, match="not recorded"):
        loader("https://fixture.invalid/other", timeout=1.0)


def test_vision_fixture_paths_resolve_existing_files_beneath_root(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "diagram.png"
    image.parent.mkdir()
    image.write_bytes(b"fixed-image")
    case = EvalCase(
        case_id="vision-1",
        query="Inspect diagram",
        metadata={"vision_fixture_paths": ["images/diagram.png"]},
    )

    resolved = resolve_vision_fixture_paths(case, fixture_root=tmp_path)

    assert resolved == (image.resolve(),)


@pytest.mark.parametrize(
    "paths, message",
    [
        (["https://example.com/image.png"], "local relative paths"),
        (["../outside.png"], "outside fixture root"),
        (["missing.png"], "does not exist"),
    ],
)
def test_vision_fixture_paths_reject_remote_escape_and_missing_files(
    tmp_path: Path,
    paths: list[str],
    message: str,
) -> None:
    case = EvalCase(
        case_id="vision-1",
        query="Inspect diagram",
        metadata={"vision_fixture_paths": paths},
    )

    with pytest.raises(RecordedFixtureError, match=message):
        resolve_vision_fixture_paths(case, fixture_root=tmp_path)


def test_vision_fixture_paths_require_an_explicit_string_list(
    tmp_path: Path,
) -> None:
    case = EvalCase(
        case_id="vision-1",
        query="Inspect diagram",
        metadata={"vision_fixture_paths": "diagram.png"},
    )

    with pytest.raises(RecordedFixtureError, match="array"):
        resolve_vision_fixture_paths(case, fixture_root=tmp_path)
