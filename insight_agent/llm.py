"""Thin wrapper around the OpenAI-compatible Chat Completions client.

`LLMClient` is intentionally minimal. It:
  1. accepts an explicit ``LLMConfig`` or falls back to environment variables;
  2. builds an `openai.OpenAI` client;
  3. exposes a single `chat(...)` method that forwards arguments to the SDK
     and returns the raw SDK response.

No agent loop, no tool execution, no retry / memory / cache logic lives here.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI


logger = logging.getLogger(__name__)

_LLM_REQUEST_CONTEXT: ContextVar[tuple[str | None, str | None]] = ContextVar(
    "llm_request_context",
    default=(None, None),
)

# Single, shared registry of required environment variables and their hints.
# Both the text LLM client and the vision client read from here so the project
# keeps ONE config mechanism instead of a second, parallel one.
_ENV_VAR_HINTS: dict[str, str] = {
    "LLM_API_KEY": "API key for the OpenAI-compatible endpoint.",
    "LLM_BASE_URL": "Base URL of the OpenAI-compatible endpoint, e.g. https://api.openai.com/v1",
    "LLM_MODEL": "Model name to send to the endpoint, e.g. gpt-4o-mini.",
    "VISION_API_KEY": "API key for the OpenAI-compatible vision endpoint.",
    "VISION_BASE_URL": "Base URL of the OpenAI-compatible vision endpoint, e.g. https://api.openai.com/v1",
    "VISION_MODEL": "Vision-language model name, e.g. gpt-4o-mini.",
}


@contextmanager
def llm_request_context(
    *,
    node_name: str,
    task_id: str | None,
) -> Iterator[None]:
    """Associate LLM metrics with one workflow node for this execution context."""
    token = _LLM_REQUEST_CONTEXT.set((node_name, task_id))
    try:
        yield
    finally:
        _LLM_REQUEST_CONTEXT.reset(token)


def _require_env(name: str, environ: Mapping[str, str] | None = None) -> str:
    if environ is None:
        load_dotenv()
        values = os.environ
    else:
        values = environ
    value = values.get(name, "").strip()
    if not value:
        hint = _ENV_VAR_HINTS.get(name, "")
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"{hint} "
            f"Please set it in your shell or in a `.env` file "
            f"(see `.env.example` for reference)."
        )
    return value


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Validated configuration for one OpenAI-compatible text model."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    context_window_tokens: int | None = None
    reserved_output_tokens: int = 0
    context_safety_margin_tokens: int = 0
    request_metrics_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("api_key", "base_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLM {name} must not be empty")
        if (
            self.context_window_tokens is not None
            and (
                isinstance(self.context_window_tokens, bool)
                or not isinstance(self.context_window_tokens, int)
                or self.context_window_tokens <= 0
            )
        ):
            raise ValueError("LLM context_window_tokens must be a positive integer")
        for name in ("reserved_output_tokens", "context_safety_margin_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"LLM {name} must be a non-negative integer")
        if not isinstance(self.request_metrics_enabled, bool):
            raise ValueError("LLM request_metrics_enabled must be a boolean")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> LLMConfig:
        """Resolve the existing ``LLM_*`` variables from a mapping."""
        if environ is None:
            load_dotenv()
        values = os.environ if environ is None else environ
        resolved: dict[str, str] = {}
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
            value = values.get(name, "").strip()
            if not value:
                hint = _ENV_VAR_HINTS[name]
                raise RuntimeError(
                    f"Missing required environment variable {name!r}. "
                    f"{hint} Please set it in your shell or in a `.env` file "
                    f"(see `.env.example` for reference)."
                )
            resolved[name] = value
        return cls(
            api_key=resolved["LLM_API_KEY"],
            base_url=resolved["LLM_BASE_URL"],
            model=resolved["LLM_MODEL"],
            context_window_tokens=_optional_positive_int(
                values, "LLM_CONTEXT_WINDOW_TOKENS"
            ),
            reserved_output_tokens=_nonnegative_int(
                values, "LLM_RESERVED_OUTPUT_TOKENS"
            ),
            context_safety_margin_tokens=_nonnegative_int(
                values, "LLM_CONTEXT_SAFETY_MARGIN_TOKENS"
            ),
            request_metrics_enabled=_boolean(
                values, "LLM_REQUEST_METRICS_ENABLED", default=False
            ),
        )


def _optional_positive_int(
    values: Mapping[str, str],
    name: str,
) -> int | None:
    raw = values.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(values: Mapping[str, str], name: str) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(
    values: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    raw = values.get(name, "")
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a boolean")
    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _estimate_chat_input_tokens(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
) -> int:
    """Conservatively estimate tokens in the complete JSON request input.

    This intentionally model-agnostic heuristic counts at least one estimated
    token per Unicode character and at least one per three UTF-8 bytes. It is
    not an exact tokenizer and may overestimate, especially for plain ASCII.
    """
    payload: dict[str, Any] = {"messages": list(messages)}
    if tools is not None:
        payload["tools"] = list(tools)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    utf8_bytes = len(serialized.encode("utf-8"))
    return max(len(serialized), (utf8_bytes + 2) // 3)


class ContextBudgetExceeded(ValueError):
    """Raised before an SDK call when an estimated request exceeds its budget."""

    def __init__(
        self,
        *,
        context_window_tokens: int,
        estimated_input_tokens: int,
        reserved_output_tokens: int,
        safety_margin_tokens: int,
    ) -> None:
        self.context_window_tokens = context_window_tokens
        self.estimated_input_tokens = estimated_input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.input_budget_tokens = (
            context_window_tokens
            - reserved_output_tokens
            - safety_margin_tokens
        )
        self.estimated_total_tokens = (
            estimated_input_tokens + reserved_output_tokens + safety_margin_tokens
        )
        super().__init__(
            "LLM context budget exceeded: "
            f"estimated_input_tokens={estimated_input_tokens}, "
            f"reserved_output_tokens={reserved_output_tokens}, "
            f"safety_margin_tokens={safety_margin_tokens}, "
            f"estimated_total_tokens={self.estimated_total_tokens}, "
            f"context_window_tokens={context_window_tokens}, "
            f"input_budget_tokens={self.input_budget_tokens}"
        )


class LLMClient:
    """A thin OpenAI-compatible Chat Completions client.

    Attributes:
        model:  the configured model name.
        client: the underlying ``openai.OpenAI`` instance.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        resolved = config if config is not None else LLMConfig.from_env()
        self.model: str = resolved.model
        self._context_window_tokens = resolved.context_window_tokens
        self._reserved_output_tokens = resolved.reserved_output_tokens
        self._context_safety_margin_tokens = resolved.context_safety_margin_tokens
        self._request_metrics_enabled = resolved.request_metrics_enabled
        self.client: OpenAI = OpenAI(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
        )
        self._closed = False
        self._usage_lock = threading.Lock()
        self._usage_calls = 0
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_unmetered_calls = 0

    def close(self) -> None:
        """Release the SDK connection pool at most once."""
        if self._closed:
            return
        self._closed = True
        self.client.close()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Call the Chat Completions API and return the raw SDK response.

        Args:
            messages: OpenAI-format chat messages.
            tools:    Optional list of OpenAI-format tool schemas. When omitted,
                      the call is made without the ``tools`` kwarg so the model
                      cannot emit tool calls.
            timeout:  Optional per-request SDK deadline in seconds.

        Returns:
            The ``ChatCompletion`` object produced by the SDK. The caller is
            responsible for reading ``.choices[0].message`` etc.
        """
        self._ensure_usage_state()
        request_messages = list(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
        }
        request_tools: list[dict[str, Any]] | None = None
        if tools is not None:
            request_tools = list(tools)
            kwargs["tools"] = request_tools
        if timeout is not None:
            kwargs["timeout"] = timeout
        context_window_tokens = getattr(self, "_context_window_tokens", None)
        request_metrics_enabled = getattr(
            self, "_request_metrics_enabled", False
        )
        estimated_input_tokens: int | None = None
        if context_window_tokens is not None or request_metrics_enabled:
            estimated_input_tokens = _estimate_chat_input_tokens(
                request_messages, request_tools
            )
        input_budget_tokens: int | None = None
        budget_usage_ratio: float | None = None
        if context_window_tokens is not None:
            reserved_output_tokens = self._reserved_output_tokens
            safety_margin_tokens = self._context_safety_margin_tokens
            if request_metrics_enabled:
                input_budget_tokens = (
                    context_window_tokens
                    - reserved_output_tokens
                    - safety_margin_tokens
                )
                if input_budget_tokens > 0:
                    budget_usage_ratio = estimated_input_tokens / input_budget_tokens
            if (
                estimated_input_tokens
                + reserved_output_tokens
                + safety_margin_tokens
                > context_window_tokens
            ):
                error = ContextBudgetExceeded(
                    context_window_tokens=context_window_tokens,
                    estimated_input_tokens=estimated_input_tokens,
                    reserved_output_tokens=reserved_output_tokens,
                    safety_margin_tokens=safety_margin_tokens,
                )
                if request_metrics_enabled:
                    self._log_request_metrics(
                        estimated_input_tokens=estimated_input_tokens,
                        context_window_tokens=context_window_tokens,
                        input_budget_tokens=input_budget_tokens,
                        budget_usage_ratio=budget_usage_ratio,
                        actual_input_tokens=None,
                        actual_output_tokens=None,
                        status="context_rejected",
                        request_sent=False,
                        error_type=type(error).__name__,
                    )
                raise error
        try:
            response = self.client.chat.completions.create(**kwargs)
        except BaseException as exc:
            with self._usage_lock:
                self._usage_calls += 1
                self._usage_unmetered_calls += 1
            if request_metrics_enabled:
                self._log_request_metrics(
                    estimated_input_tokens=estimated_input_tokens,
                    context_window_tokens=context_window_tokens,
                    input_budget_tokens=input_budget_tokens,
                    budget_usage_ratio=budget_usage_ratio,
                    actual_input_tokens=None,
                    actual_output_tokens=None,
                    status="api_error",
                    request_sent=True,
                    error_type=type(exc).__name__,
                )
            raise
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        metered = (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
        )
        with self._usage_lock:
            self._usage_calls += 1
            if metered:
                self._usage_input_tokens += input_tokens
                self._usage_output_tokens += output_tokens
            else:
                self._usage_unmetered_calls += 1
        if request_metrics_enabled:
            actual_input_tokens = (
                input_tokens
                if isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                else None
            )
            actual_output_tokens = (
                output_tokens
                if isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                else None
            )
            self._log_request_metrics(
                estimated_input_tokens=estimated_input_tokens,
                context_window_tokens=context_window_tokens,
                input_budget_tokens=input_budget_tokens,
                budget_usage_ratio=budget_usage_ratio,
                actual_input_tokens=actual_input_tokens,
                actual_output_tokens=actual_output_tokens,
                status="success",
                request_sent=True,
                error_type=None,
            )
        return response

    def _log_request_metrics(
        self,
        *,
        estimated_input_tokens: int,
        context_window_tokens: int | None,
        input_budget_tokens: int | None,
        budget_usage_ratio: float | None,
        actual_input_tokens: int | None,
        actual_output_tokens: int | None,
        status: str,
        request_sent: bool,
        error_type: str | None,
    ) -> None:
        """Emit one content-free JSON event for an LLM request attempt."""
        node_name, task_id = _LLM_REQUEST_CONTEXT.get()
        event = {
            "event": "llm_request_metrics",
            "model": self.model,
            "node_name": node_name,
            "task_id": task_id,
            "estimated_input_tokens": estimated_input_tokens,
            "input_token_estimate_method": "model_agnostic_heuristic",
            "context_window_tokens": context_window_tokens,
            "input_budget_tokens": input_budget_tokens,
            "budget_usage_ratio": budget_usage_ratio,
            "actual_input_tokens": actual_input_tokens,
            "actual_output_tokens": actual_output_tokens,
            "status": status,
            "request_sent": request_sent,
            "error_type": error_type,
        }
        logger.info(
            json.dumps(event, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        )

    def usage_snapshot(self) -> dict[str, int]:
        """Return cumulative provider usage without guessing missing token data."""
        self._ensure_usage_state()
        with self._usage_lock:
            return {
                "calls": self._usage_calls,
                "input_tokens": self._usage_input_tokens,
                "output_tokens": self._usage_output_tokens,
                "unmetered_calls": self._usage_unmetered_calls,
            }

    def _ensure_usage_state(self) -> None:
        """Support lightweight test doubles constructed without __init__."""
        if hasattr(self, "_usage_lock"):
            return
        self._usage_lock = threading.Lock()
        self._usage_calls = 0
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._usage_unmetered_calls = 0
