"""Deterministic identity helpers for task-scoped Evidence."""

from __future__ import annotations

import hashlib
import json

from insight_agent.routing import RetrievalSource


def make_evidence_id(
    task_id: str,
    retrieval_source: RetrievalSource,
    origin_id: str,
) -> str:
    """Hash the complete task/source/origin identity without ambiguity."""
    payload = json.dumps(
        [task_id, retrieval_source.value, origin_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

