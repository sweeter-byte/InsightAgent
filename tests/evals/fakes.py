from __future__ import annotations

from typing import Any

from evals.judge import JudgeRequest
from evals.models import JudgeResult
from insight_agent.evidence import Evidence
from insight_agent.routing import RetrievalSource


class ScriptedJudge:
    """Return or raise scripted values while retaining every Judge request."""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.requests: list[JudgeRequest] = []

    def evaluate(self, request: JudgeRequest) -> JudgeResult:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected Judge call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


def fixed_evidence(
    evidence_id: str = "E1",
    *,
    task_id: str = "T1",
    content: str = "Chunk overlap preserves context across boundaries.",
    source: str = "notes/chunking.md",
    retrieval_source: RetrievalSource = RetrievalSource.LOCAL,
    metadata: dict[str, Any] | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id=task_id,
        retrieval_source=retrieval_source,
        origin_id=f"origin-{evidence_id}",
        content=content,
        source=source,
        metadata=dict(metadata or {}),
    )
