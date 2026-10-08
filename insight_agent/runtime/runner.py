"""Thin asynchronous adapter over the existing checkpointed coordinator."""

from __future__ import annotations

import asyncio
from typing import Protocol

from insight_agent.planning.coordinator import ResearchThreadSnapshot
from insight_agent.runtime.service import ProgressCallback


class CheckpointedResearchCoordinator(Protocol):
    def start_research(
        self,
        query: str,
        *,
        thread_id: str | None = None,
    ) -> ResearchThreadSnapshot: ...

    def resume_research(self, thread_id: str) -> ResearchThreadSnapshot: ...


class LangGraphResearchRunner:
    """Keep Runtime unaware of graph internals and ResearchState contents."""

    def __init__(self, coordinator: CheckpointedResearchCoordinator) -> None:
        self.coordinator = coordinator

    async def run(
        self,
        *,
        thread_id: str,
        query: str | None,
        resume: bool,
        on_progress: ProgressCallback,
    ) -> str:
        if resume and query is not None:
            raise ValueError("resume execution must not provide a new query")
        if not resume and (query is None or not query.strip()):
            raise ValueError("new execution requires a non-empty query")

        if resume:
            await on_progress(
                "resuming",
                {"message": "Resuming the saved research checkpoint"},
            )
            snapshot = await asyncio.to_thread(
                self.coordinator.resume_research,
                thread_id,
            )
        else:
            assert query is not None
            await on_progress(
                "planning",
                {"message": "Planning and executing the research workflow"},
            )
            snapshot = await asyncio.to_thread(
                self.coordinator.start_research,
                query,
                thread_id=thread_id,
            )

        if not snapshot.completed:
            raise RuntimeError("research workflow returned with pending nodes")
        final_output = snapshot.state.final_output
        if not isinstance(final_output, str):
            raise RuntimeError("research workflow did not produce final_output")
        await on_progress(
            "completed",
            {"message": "Research workflow produced its final output"},
        )
        return final_output
