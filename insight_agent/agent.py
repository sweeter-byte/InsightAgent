"""The minimal Research Agent loop.

`ResearchAgent` is a plain ``while True`` loop over the OpenAI Chat Completions
tool-calling protocol. It:

  1. sends the running message list to the LLM (with tool schemas attached);
  2. appends the assistant message (including any ``tool_calls``) back to the
     message history;
  3. if the assistant produced no tool calls, returns its text as the final
     answer;
  4. otherwise executes every tool call through the registry, appends one
     ``role="tool"`` message per call, and loops again.

The only safety rail in this chapter is ``max_steps``. Retry / timeout /
context compaction / memory are explicitly out of scope.
"""

from __future__ import annotations

import json
from typing import Any

from insight_agent.llm import LLMClient
from insight_agent.tools.registry import ToolRegistry, UnknownToolError


DEFAULT_SYSTEM_PROMPT = (
    "You are InsightAgent, a research assistant. When you need information "
    "from a file, call the `read_file` tool. When a question depends on local "
    "knowledge, use `search_knowledge_base` first and base the answer on its "
    "results. If the retrieved evidence is insufficient, say that the local "
    "knowledge is insufficient. Do not state as fact information that is not "
    "present in the tool results. Always reason step by step, and "
    "when you have enough information, reply with a plain text answer and do "
    "NOT emit any further tool calls."
)


class AgentStepsExceeded(RuntimeError):
    """Raised when the loop hits ``max_steps`` without a final answer."""


class ResearchAgent:
    """A minimal LLM-driven agent that can call registered tools."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        tool_schemas: list[dict[str, Any]] | None,
        *,
        max_steps: int = 10,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        self.llm = llm
        self.registry = registry
        # A non-empty schema list is required by the OpenAI tool-calling
        # protocol; when the registry is empty, pass ``None`` so the model
        # is invoked as a plain chat completion.
        self.tool_schemas: list[dict[str, Any]] | None = (
            list(tool_schemas) if tool_schemas else None
        )
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ public

    def run(self, query: str) -> str:
        """Run the loop for a single user query and return the final text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        return self._loop(messages)

    # ----------------------------------------------------------------- private

    def _loop(self, messages: list[dict[str, Any]]) -> str:
        for _step in range(self.max_steps):
            response = self.llm.chat(messages, tools=self.tool_schemas)
            assistant_msg = response.choices[0].message
            assistant_dict = _assistant_message_to_dict(assistant_msg)
            messages.append(assistant_dict)

            tool_calls = assistant_dict.get("tool_calls")
            if not tool_calls:
                content = assistant_dict.get("content")
                return content if isinstance(content, str) and content else ""

            for call in tool_calls:
                messages.append(self._execute_tool_call(call))

        raise AgentStepsExceeded(
            f"Agent exceeded {self.max_steps} steps without producing a final answer."
        )

    def _execute_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call and return the ``role="tool"`` message to append."""
        call_id = call.get("id", "")
        fn = call.get("function", {}) or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "") or "{}"

        # 1) parse arguments — malformed JSON is an *error message*, not a crash.
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Error: invalid arguments for tool {name!r}: {exc}",
            }

        # 2) unknown tool — surface a clear message rather than raising.
        try:
            self.registry.get(name)
        except UnknownToolError as exc:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Error: {exc}",
            }

        # 3) execute; any tool-side exception is captured as the tool result.
        try:
            result = self.registry.execute(name, arguments)
        except Exception as exc:  # noqa: BLE001 — the loop must not crash on a bad tool.
            result = f"Error: tool {name!r} failed: {exc}"

        if not isinstance(result, str):
            result = str(result)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """Convert an SDK ``ChatCompletionMessage`` into a plain dict for replay.

    We build the dict by hand (rather than calling ``model_dump``) so that the
    message we write back contains only fields the Chat Completions API
    accepts on subsequent turns.
    """
    out: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    out["content"] = content if isinstance(content, str) else None

    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": getattr(tc, "type", "function") or "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in raw_tool_calls
        ]
    return out
