"""LLM-backed source selection for one already-planned research task."""

from __future__ import annotations

import json
from typing import Any

from insight_agent.llm import LLMClient
from insight_agent.planning.models import ResearchTask
from insight_agent.routing.models import (
    RetrievalSource,
    RouteDecision,
    RoutingError,
)


RETRIEVAL_ROUTER_SYSTEM_PROMPT = """You are InsightAgent's Retrieval Router.

Choose exactly one primary source for the current research task:
- local: user-provided, locally ingested, or already indexed material.
- web: current public information, external papers, open-source projects,
  official documentation, or other internet content.
- vision: information that requires direct inspection of an original image,
  chart, screenshot, layout, or visual relationship.

Rules:
1. Choose only from the available sources supplied by the runtime.
2. Choose one primary source. Do not split or rewrite the task.
3. Do not execute retrieval, search, HTTP requests, tools, or vision analysis.
4. Do not answer the research question and do not present facts as evidence.
5. Return only one JSON object with no Markdown fences or extra text:
   {"source": "local", "reason": "a non-empty explanation"}
"""


class RetrievalRouter:
    """Select one allowed source without executing any retrieval."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def route(
        self,
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        available_sources: set[RetrievalSource],
    ) -> RouteDecision:
        """Return a strictly parsed route decision for ``task``."""
        self._validate_input(task, objective, constraints, available_sources)
        payload = {
            "objective": objective,
            "constraints": list(constraints),
            "task": {"id": task.id, "question": task.question},
            "available_sources": sorted(source.value for source in available_sources),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        response = self.llm.chat(messages, tools=None)
        raw_text = self._extract_text(response)
        if not raw_text.strip():
            raise RoutingError("retrieval router response did not contain JSON text")

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RoutingError(
                f"retrieval router returned invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(result, dict):
            raise RoutingError("retrieval router JSON must be an object")

        for field_name in ("source", "reason"):
            if field_name not in result:
                raise RoutingError(
                    f"retrieval router JSON is missing required field {field_name!r}"
                )

        raw_source = result["source"]
        if not isinstance(raw_source, str):
            raise RoutingError("retrieval router source must be a string")
        try:
            source = RetrievalSource(raw_source)
        except ValueError as exc:
            raise RoutingError(
                f"retrieval router returned unknown source {raw_source!r}"
            ) from exc
        if source not in available_sources:
            raise RoutingError(
                f"retrieval source {source.value!r} is not available for this run"
            )

        reason = result["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise RoutingError("retrieval router reason must be a non-empty string")
        return RouteDecision(task_id=task.id, source=source, reason=reason.strip())

    @staticmethod
    def _validate_input(
        task: ResearchTask,
        objective: str,
        constraints: list[str],
        available_sources: set[RetrievalSource],
    ) -> None:
        if not isinstance(task, ResearchTask):
            raise RoutingError("task must be a ResearchTask")
        if not isinstance(task.id, str) or not task.id.strip():
            raise RoutingError("task id must be a non-empty string")
        if not isinstance(task.question, str) or not task.question.strip():
            raise RoutingError("task question must be a non-empty string")
        if not isinstance(objective, str) or not objective.strip():
            raise RoutingError("research objective must be a non-empty string")
        if not isinstance(constraints, list) or any(
            not isinstance(constraint, str) or not constraint.strip()
            for constraint in constraints
        ):
            raise RoutingError("constraints must be a list of non-empty strings")
        if not isinstance(available_sources, set) or not available_sources:
            raise RoutingError("available_sources must be a non-empty set")
        if any(not isinstance(source, RetrievalSource) for source in available_sources):
            raise RoutingError(
                "available_sources must contain only RetrievalSource values"
            )

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""

