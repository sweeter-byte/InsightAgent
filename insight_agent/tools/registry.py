"""A minimal tool registry: name -> callable dispatch.

The registry does exactly three things:

  * register a tool under a name;
  * look a tool up by name;
  * execute a tool by name with keyword arguments.

It intentionally avoids class hierarchies, dependency injection, or schema
validation — those concerns live in the LLM tool schema and the agent loop.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from insight_agent.tools.file_tools import READ_FILE_SCHEMA, read_file


class UnknownToolError(KeyError):
    """Raised when the agent requests a tool name that was never registered."""


class ToolRegistry:
    """A flat ``name -> callable`` mapping with a small ``execute`` helper."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register ``fn`` under ``name``. Later registrations overwrite earlier ones."""
        if not name or not callable(fn):
            raise ValueError("register() requires a non-empty name and a callable.")
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., str]:
        """Return the callable registered under ``name``.

        Raises:
            UnknownToolError: if no tool is registered under that name.
        """
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "<none>"
            raise UnknownToolError(
                f"Unknown tool {name!r}. Registered tools: {known}."
            )
        return self._tools[name]

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> str:
        """Look up ``name`` and call it with ``arguments`` as keyword args.

        The registry itself does not swallow exceptions raised by the tool —
        the agent loop is responsible for turning a raised exception into an
        error string that is written back into the message history.
        """
        fn = self.get(name)
        return fn(**dict(arguments or {}))

    def names(self) -> list[str]:
        """Return the sorted list of registered tool names (handy for tests / debug)."""
        return sorted(self._tools)


def build_default_registry() -> ToolRegistry:
    """Return a registry pre-populated with the tools this chapter provides."""
    registry = ToolRegistry()
    registry.register("read_file", read_file)
    return registry


def default_tool_schemas() -> list[dict[str, Any]]:
    """Return the OpenAI-format schemas for every tool in the default registry.

    Keeping the schema list next to the registry builder — but in a separate
    function — makes the coupling explicit: whoever adds a tool must add both
    the callable here and its schema in ``default_tool_schemas()``.
    """
    return [READ_FILE_SCHEMA]
