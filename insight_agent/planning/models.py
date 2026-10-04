"""Minimal data models for structured research planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


MAX_RESEARCH_TASKS = 6
_TASK_ID_PATTERN = re.compile(r"^T[1-9][0-9]*$")


class PlanningError(RuntimeError):
    """Raised when a research plan cannot be parsed or validated."""


@dataclass(slots=True)
class ResearchTask:
    """One research question and the earlier tasks it depends on."""

    id: str
    question: str
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchPlan:
    """A unified objective decomposed into dependency-ordered tasks."""

    objective: str
    constraints: list[str] = field(default_factory=list)
    tasks: list[ResearchTask] = field(default_factory=list)


@dataclass(slots=True)
class ResearchState:
    """Current Chapter 7 state: the original query and its structured plan."""

    query: str
    plan: ResearchPlan | None = None


def validate_research_plan(plan: ResearchPlan) -> None:
    """Validate a plan and its already-topologically-ordered task list."""
    if not isinstance(plan.objective, str) or not plan.objective.strip():
        raise PlanningError("research plan objective must be a non-empty string")

    if not isinstance(plan.constraints, list) or any(
        not isinstance(constraint, str) or not constraint.strip()
        for constraint in plan.constraints
    ):
        raise PlanningError("research plan constraints must be a list of strings")

    if not isinstance(plan.tasks, list) or not 1 <= len(plan.tasks) <= MAX_RESEARCH_TASKS:
        raise PlanningError(
            f"research plan must contain between 1 and {MAX_RESEARCH_TASKS} tasks"
        )

    seen_ids: set[str] = set()
    for task in plan.tasks:
        if not isinstance(task, ResearchTask):
            raise PlanningError("research plan tasks must be ResearchTask objects")
        if not isinstance(task.id, str) or not _TASK_ID_PATTERN.fullmatch(task.id):
            raise PlanningError(
                f"invalid task id {task.id!r}; expected T1, T2, T3, ..."
            )
        if task.id in seen_ids:
            raise PlanningError(f"duplicate task id {task.id!r}")
        if not isinstance(task.question, str) or not task.question.strip():
            raise PlanningError(f"task {task.id!r} question must not be empty")
        if not isinstance(task.depends_on, list) or any(
            not isinstance(dependency, str) for dependency in task.depends_on
        ):
            raise PlanningError(
                f"task {task.id!r} depends_on must be a list of task IDs"
            )
        if len(task.depends_on) != len(set(task.depends_on)):
            raise PlanningError(f"task {task.id!r} contains duplicate dependencies")
        for dependency in task.depends_on:
            if dependency not in seen_ids:
                raise PlanningError(
                    f"task {task.id!r} dependency must reference an earlier task; "
                    f"got {dependency!r}"
                )
        seen_ids.add(task.id)
