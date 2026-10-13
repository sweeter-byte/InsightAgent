from __future__ import annotations

from evals.beir_scifact.runner import run_queries


class FakeExperimentRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run_query(self, query_id: str, query: str):
        self.calls.append((query_id, query))
        return {"A": f"result-{query_id}"}


def test_run_queries_uses_each_original_query_once_and_never_accepts_qrels() -> None:
    runner = FakeExperimentRunner()

    results = run_queries({"q2": "second", "q1": "first"}, runner)

    assert runner.calls == [("q2", "second"), ("q1", "first")]
    assert results == {
        "q2": {"A": "result-q2"},
        "q1": {"A": "result-q1"},
    }
