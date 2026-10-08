"""Network-free recorded inputs for Offline Evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

from evals.models import EvalCase
from insight_agent.ingestion import Document
from insight_agent.web_search import WebSearchHit


class RecordedFixtureError(ValueError):
    """A recorded fixture was missing, unsafe, or structurally invalid."""


class RecordedSearchProvider:
    """Return only previously recorded search hits; never contact a provider."""

    def __init__(
        self,
        hits_by_query: Mapping[str, Sequence[WebSearchHit]],
    ) -> None:
        self.hits_by_query: dict[str, tuple[WebSearchHit, ...]] = {}
        for query, hits in hits_by_query.items():
            if not isinstance(query, str) or not query.strip():
                raise RecordedFixtureError(
                    "recorded search queries must be non-empty strings"
                )
            if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
                raise RecordedFixtureError("recorded search hits must be arrays")
            if any(not isinstance(hit, WebSearchHit) for hit in hits):
                raise RecordedFixtureError(
                    "recorded search hits must contain WebSearchHit values"
                )
            self.hits_by_query[query] = tuple(hits)

    def search(self, query: str, *, limit: int) -> list[WebSearchHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise RecordedFixtureError("search limit must be a positive integer")
        if query not in self.hits_by_query:
            raise RecordedFixtureError(f"search query is not recorded: {query!r}")
        return list(self.hits_by_query[query][:limit])


class RecordedUrlLoader:
    """Production-compatible URL Loader backed only by fixed Documents."""

    def __init__(
        self,
        documents_by_url: Mapping[str, Sequence[Document]],
    ) -> None:
        self.documents_by_url: dict[str, list[Document]] = {}
        for url, documents in documents_by_url.items():
            if not isinstance(url, str) or not url.strip():
                raise RecordedFixtureError(
                    "recorded page URLs must be non-empty strings"
                )
            if not isinstance(documents, Sequence) or isinstance(
                documents, (str, bytes)
            ):
                raise RecordedFixtureError("recorded page documents must be arrays")
            if any(not isinstance(document, Document) for document in documents):
                raise RecordedFixtureError(
                    "recorded page documents must contain Document values"
                )
            self.documents_by_url[url] = list(documents)

    def __call__(self, url: str, *, timeout: float) -> list[Document]:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise RecordedFixtureError("URL timeout must be a positive number")
        if url not in self.documents_by_url:
            raise RecordedFixtureError(f"page URL is not recorded: {url!r}")
        return list(self.documents_by_url[url])


def resolve_vision_fixture_paths(
    case: EvalCase,
    *,
    fixture_root: str | Path,
) -> tuple[Path, ...]:
    """Resolve fixed local images without downloading or escaping their root."""
    if not isinstance(case, EvalCase):
        raise TypeError("case must be an EvalCase")
    raw_paths = case.metadata.get("vision_fixture_paths", [])
    if not isinstance(raw_paths, list):
        raise RecordedFixtureError("vision_fixture_paths must be an array")

    root = Path(fixture_root).resolve()
    resolved: list[Path] = []
    for index, raw_path in enumerate(raw_paths, start=1):
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RecordedFixtureError(
                f"vision fixture path {index} must be a non-empty string"
            )
        parsed = urlparse(raw_path)
        if parsed.scheme or parsed.netloc or Path(raw_path).is_absolute():
            raise RecordedFixtureError(
                "vision_fixture_paths must contain local relative paths"
            )
        path = (root / raw_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RecordedFixtureError(
                f"vision fixture path is outside fixture root: {raw_path!r}"
            ) from exc
        if not path.is_file():
            raise RecordedFixtureError(
                f"vision fixture path does not exist: {raw_path!r}"
            )
        resolved.append(path)
    return tuple(resolved)
