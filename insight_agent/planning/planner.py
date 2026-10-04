"""LLM-backed query analysis and research-task decomposition only."""

from __future__ import annotations

import json
from typing import Any

from insight_agent.llm import LLMClient
from insight_agent.planning.models import (
    PlanningError,
    ResearchPlan,
    ResearchTask,
    validate_research_plan,
)


PLANNER_SYSTEM_PROMPT = """You are InsightAgent's Research Planner.

Analyze the user's original research request and return one structured plan.

Rules:
1. Identify the user's real, unified research objective.
2. Extract only constraints explicitly stated by the user, including scope, time,
   comparison dimensions, and output limits. Do not add constraints the user did
   not request, such as a date range or source count.
3. Decompose the request into a small number of clear research questions,
   usually 2 to 6 tasks.
4. Give tasks stable IDs T1, T2, T3, and so on.
5. Use depends_on to list prerequisite task IDs. A task may depend only on tasks
   that appear earlier in the tasks array.
6. Each task question describes only what needs to be researched.
7. Do not specify tools, Qdrant, search_knowledge_base, Web Search, search
   engines, retrieval types, or where to search.
8. Do not state assumptions in the plan as if they were established facts.
9. Do not answer the research questions.
10. Return only one JSON object, with no Markdown fences or extra explanation.

The JSON must have exactly this shape:
{
  "objective": "non-empty overall objective",
  "constraints": ["explicit constraint"],
  "tasks": [
    {
      "id": "T1",
      "question": "a focused research question",
      "depends_on": []
    }
  ]
}
"""


class ResearchPlanner:
    """Convert one research query into a validated :class:`ResearchPlan`."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(self, query: str) -> ResearchPlan:
        """Generate, parse, and deterministically validate a research plan."""
        if not isinstance(query, str) or not query.strip():
            raise PlanningError("research query must be a non-empty string")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = self.llm.chat(messages, tools=None)
        raw_text = self._extract_text(response)
        if not raw_text.strip():
            raise PlanningError("planner response did not contain JSON text")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"planner returned invalid JSON: {exc.msg}") from exc

        plan = self._parse_plan(payload)
        validate_research_plan(plan)
        return plan

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def _parse_plan(payload: Any) -> ResearchPlan:
        if not isinstance(payload, dict):
            raise PlanningError("planner JSON must be an object")

        for field_name in ("objective", "constraints", "tasks"):
            if field_name not in payload:
                raise PlanningError(
                    f"planner JSON is missing required field {field_name!r}"
                )

        objective = payload["objective"]
        constraints = payload["constraints"]
        raw_tasks = payload["tasks"]

        if not isinstance(constraints, list):
            raise PlanningError("research plan constraints must be a list of strings")
        if not isinstance(raw_tasks, list):
            raise PlanningError("research plan tasks must be a list")

        tasks: list[ResearchTask] = []
        for index, raw_task in enumerate(raw_tasks, start=1):
            if not isinstance(raw_task, dict):
                raise PlanningError(f"research task {index} must be a JSON object")
            for field_name in ("id", "question", "depends_on"):
                if field_name not in raw_task:
                    raise PlanningError(
                        f"research task {index} is missing required field "
                        f"{field_name!r}"
                    )
            depends_on = raw_task["depends_on"]
            if not isinstance(depends_on, list):
                raise PlanningError(
                    f"research task {index} depends_on must be a list of task IDs"
                )
            tasks.append(
                ResearchTask(
                    id=raw_task["id"],
                    question=raw_task["question"],
                    depends_on=list(depends_on),
                )
            )

        return ResearchPlan(
            objective=objective,
            constraints=list(constraints),
            tasks=tasks,
        )
