"""Tests for `read_file` and the default tool registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.tools.file_tools import READ_FILE_SCHEMA, read_file
from insight_agent.tools.registry import (
    ToolRegistry,
    UnknownToolError,
    build_default_registry,
    default_tool_schemas,
)


# ---------- read_file ---------------------------------------------------------


def test_read_file_returns_content(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("Method A: 81.3%\nMethod B: 87.6%\n", encoding="utf-8")

    assert read_file(str(f)) == "Method A: 81.3%\nMethod B: 87.6%\n"


def test_read_file_missing_file_returns_error_string(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"
    result = read_file(str(missing))

    assert isinstance(result, str)
    assert result.startswith("Error: ")
    assert "not found" in result


def test_read_file_directory_returns_error_string(tmp_path: Path) -> None:
    result = read_file(str(tmp_path))
    assert result.startswith("Error: ")
    assert "not a regular file" in result


def test_read_file_empty_path_returns_error_string() -> None:
    assert read_file("").startswith("Error: ")


def test_read_file_does_not_raise_on_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permission / OS error must be caught and returned as an error string."""
    f = tmp_path / "secret.txt"
    f.write_text("hidden", encoding="utf-8")

    def _boom(*_a: object, **_kw: object) -> str:
        raise PermissionError("mocked permission denied")

    monkeypatch.setattr(Path, "read_text", _boom)

    result = read_file(str(f))
    assert result.startswith("Error: ")


# ---------- Tool schema -------------------------------------------------------


def test_read_file_schema_shape() -> None:
    assert READ_FILE_SCHEMA["type"] == "function"
    fn = READ_FILE_SCHEMA["function"]
    assert fn["name"] == "read_file"
    assert "description" in fn
    params = fn["parameters"]
    assert params["type"] == "object"
    assert "path" in params["properties"]
    assert params["properties"]["path"]["type"] == "string"
    assert params["required"] == ["path"]


# ---------- Tool registry -----------------------------------------------------


def test_default_registry_exposes_read_file() -> None:
    registry = build_default_registry()
    assert "read_file" in registry.names()
    assert registry.get("read_file") is read_file


def test_default_registry_executes_read_file(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("abc", encoding="utf-8")

    registry = build_default_registry()
    assert registry.execute("read_file", {"path": str(f)}) == "abc"


def test_unknown_tool_raises_semantic_error() -> None:
    registry = build_default_registry()
    with pytest.raises(UnknownToolError, match="Unknown tool"):
        registry.get("does_not_exist")
    with pytest.raises(UnknownToolError, match="Unknown tool"):
        registry.execute("does_not_exist", {})


def test_register_rejects_bad_input() -> None:
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register("", lambda: "x")
    with pytest.raises(ValueError):
        registry.register("not_callable", "a string")  # type: ignore[arg-type]


def test_default_tool_schemas_reference_registered_tools() -> None:
    schemas = default_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == set(build_default_registry().names())
