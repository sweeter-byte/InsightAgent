"""Tests for structured research-plan generation and validation."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from insight_agent.planning import PlanningError, ResearchPlanner


class FakeLLM:
    """Return one SDK-shaped response while recording the planner call."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.calls.append(
            {"messages": [dict(message) for message in messages], "tools": tools}
        )
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _plan_json(
    *,
    objective: object = "Compare memory architectures",
    constraints: object = None,
    tasks: object = None,
) -> str:
    if constraints is None:
        constraints = ["Focus on architecture"]
    if tasks is None:
        tasks = [
            {
                "id": "T1",
                "question": "What are the main architecture families?",
                "depends_on": [],
            },
            {
                "id": "T2",
                "question": "How do the architecture families compare?",
                "depends_on": ["T1"],
            },
        ]
    return json.dumps(
        {"objective": objective, "constraints": constraints, "tasks": tasks}
    )


def test_valid_json_is_parsed_into_a_research_plan_without_tools() -> None:
    llm = FakeLLM(_plan_json())
    planner = ResearchPlanner(llm=llm)  # type: ignore[arg-type]

    plan = planner.plan("research Agent Memory")

    assert plan.objective == "Compare memory architectures"
    assert plan.constraints == ["Focus on architecture"]
    assert [task.id for task in plan.tasks] == ["T1", "T2"]
    assert plan.tasks[0].question == "What are the main architecture families?"
    assert plan.tasks[1].depends_on == ["T1"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["messages"][-1] == {
        "role": "user",
        "content": "research Agent Memory",
    }


def test_planner_prompt_forbids_tools_and_unsupported_assumptions() -> None:
    llm = FakeLLM(_plan_json())
    planner = ResearchPlanner(llm=llm)  # type: ignore[arg-type]

    planner.plan("research query")

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "2 to 6" in prompt
    assert "T1" in prompt
    assert "depends_on" in prompt
    assert "Do not specify tools" in prompt
    assert "Do not add constraints" in prompt
    assert "JSON" in prompt


def test_invalid_json_raises_a_clear_planning_error() -> None:
    planner = ResearchPlanner(llm=FakeLLM("{not-json"))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="invalid JSON"):
        planner.plan("research query")


@pytest.mark.parametrize(
    ("objective", "message"),
    [("", "objective"), ("   ", "objective"), (7, "objective")],
)
def test_invalid_objective_is_rejected(objective: object, message: str) -> None:
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(objective=objective)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match=message):
        planner.plan("research query")


@pytest.mark.parametrize(
    "tasks",
    [
        [],
        [
            {"id": f"T{index}", "question": f"Question {index}", "depends_on": []}
            for index in range(1, 8)
        ],
    ],
)
def test_task_count_must_be_between_one_and_six(tasks: object) -> None:
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="between 1 and 6"):
        planner.plan("research query")


def test_duplicate_task_ids_are_rejected() -> None:
    tasks = [
        {"id": "T1", "question": "First question", "depends_on": []},
        {"id": "T1", "question": "Second question", "depends_on": []},
    ]
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="duplicate task id.*T1"):
        planner.plan("research query")


def test_empty_task_question_is_rejected() -> None:
    tasks = [{"id": "T1", "question": "  ", "depends_on": []}]
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="question"):
        planner.plan("research query")


@pytest.mark.parametrize("task_id", ["1", "task-1", "T0", "T-1"])
def test_malformed_task_ids_are_rejected(task_id: str) -> None:
    tasks = [{"id": task_id, "question": "Question", "depends_on": []}]
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="invalid task id"):
        planner.plan("research query")


def test_dependency_must_reference_an_earlier_task() -> None:
    tasks = [
        {"id": "T1", "question": "First question", "depends_on": ["T2"]},
        {"id": "T2", "question": "Second question", "depends_on": []},
    ]
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="earlier task.*T2"):
        planner.plan("research query")


def test_dependency_must_reference_an_existing_task() -> None:
    tasks = [
        {"id": "T1", "question": "First question", "depends_on": []},
        {"id": "T2", "question": "Second question", "depends_on": ["T9"]},
    ]
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(tasks=tasks)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match="earlier task.*T9"):
        planner.plan("research query")


@pytest.mark.parametrize(
    ("constraints", "message"),
    [("recent papers", "constraints"), (["valid", 3], "constraints")],
)
def test_constraints_must_be_a_list_of_strings(
    constraints: object,
    message: str,
) -> None:
    planner = ResearchPlanner(llm=FakeLLM(_plan_json(constraints=constraints)))  # type: ignore[arg-type]

    with pytest.raises(PlanningError, match=message):
        planner.plan("research query")
