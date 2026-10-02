"""Filesystem tools exposed to the agent.

Chapter 1 keeps this surface tiny: only `read_file` is implemented. Sandbox,
permission, size limits, binary handling and path-traversal defenses are
deliberately out of scope for this chapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: filesystem path to a text file.

    Returns:
        The file's decoded text on success, or a human-readable error string
        starting with ``"Error: "`` when the read fails. This function never
        raises low-level ``OSError`` / ``UnicodeDecodeError`` to the caller —
        the agent loop forwards the returned string straight back to the LLM,
        so a descriptive error message is more useful than a traceback.
    """
    if not isinstance(path, str) or not path:
        return "Error: `path` must be a non-empty string."

    p = Path(path)
    try:
        if not p.exists():
            return f"Error: file not found: {path}"
        if not p.is_file():
            return f"Error: not a regular file: {path}"
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return f"Error: file is not valid UTF-8 text: {path} ({exc.reason})"
    except OSError as exc:
        return f"Error: failed to read {path}: {exc.strerror or exc}"


READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read the contents of a UTF-8 text file at the given path and "
            "return them as a string. Use this whenever the user asks about "
            "a file you have not yet inspected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the text file to read.",
                },
            },
            "required": ["path"],
        },
    },
}
