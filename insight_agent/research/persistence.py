"""SQLite-backed checkpoint infrastructure for research workflows."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_CHECKPOINT_PATH = Path(".insight_agent/checkpoints.sqlite")


def thread_config(thread_id: str) -> RunnableConfig:
    """Build the LangGraph config for one logical research workflow."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")
    return {"configurable": {"thread_id": thread_id}}


class SQLiteCheckpointStore:
    """Own a synchronous SQLite connection and its LangGraph saver."""

    def __init__(
        self,
        path: str | Path = DEFAULT_CHECKPOINT_PATH,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
        )
        self.checkpointer = SqliteSaver(self.connection)
        self._closed = False

    def close(self) -> None:
        """Close the owned connection; repeated calls are harmless."""
        if self._closed:
            return
        self.connection.close()
        self._closed = True

    def check_ready(self) -> None:
        """Verify the already-open SQLite connection without mutating state."""
        row = self.connection.execute("SELECT 1").fetchone()
        if row != (1,):
            raise RuntimeError("checkpoint readiness query returned no result")

    def __enter__(self) -> SQLiteCheckpointStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
