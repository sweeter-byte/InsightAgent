"""Query iteration kept deliberately separate from qrels and reporting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class QueryRunner(Protocol):
    def run_query(self, query_id: str, query: str) -> Mapping[str, Any]: ...


def run_queries(
    queries: Mapping[str, str],
    runner: QueryRunner,
) -> dict[str, Mapping[str, Any]]:
    """Run each original query exactly once; qrels are not part of this API."""
    return {
        query_id: runner.run_query(query_id, query)
        for query_id, query in queries.items()
    }
