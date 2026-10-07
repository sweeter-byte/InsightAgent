"""Minimal Intent Router for classifying user queries.

The router performs a single LLM call with a focused classification prompt and
maps the response to one of three intents: ``direct``, ``analyze``, ``research``.

Priority rule: research > analyze > direct.

No structured output, no keyword heuristics, no retry logic — just one LLM call
and a simple string-to-enum conversion with a safe fallback.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from insight_agent.llm import LLMClient


# ---------------------------------------------------------------------------
# Intent enumeration
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    DIRECT = "direct"
    ANALYZE = "analyze"
    RESEARCH = "research"


# ---------------------------------------------------------------------------
# Router system prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = (
    "You are an intent classifier. Given a user query, classify it into exactly"
    " one of: direct, analyze, research.\n"
    "\n"
    "Definitions:\n"
    "- direct: general knowledge question; no external materials or search needed.\n"
    "- analyze: the user asks to read, summarize, compare, or explain already-"
    "provided or explicitly-named materials (files, documents), without explicitly"
    " requesting research or an open-ended multi-step investigation.\n"
    "- research: the user explicitly asks to conduct research, needs proactive external"
    " search, up-to-date information, comparison of multiple external sources, or an"
    " open-ended multi-step investigation. Research may use indexed local materials or"
    " images as its evidence sources.\n"
    "\n"
    "Priority: research > analyze > direct. If the query involves BOTH provided"
    " materials AND external/search elements, classify as research.\n"
    "An explicit request to conduct research MUST be classified as research, including"
    " when the request relies only on indexed local materials or images.\n"
    "\n"
    "Rules:\n"
    "1. You MUST reply with only one word: direct, analyze, or research.\n"
    "2. Do NOT answer the user's question.\n"
    "3. Do NOT add any explanation, punctuation, or extra text."
)


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------


class IntentRouter:
    """Classifies a user query into an :class:`Intent` via a single LLM call."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def route(self, query: str) -> Intent:
        """Route *query* to an Intent.

        Returns:
            The classified Intent, falling back to ``Intent.RESEARCH`` when the
            model output is empty or does not match a valid intent string.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response: Any = self.llm.chat(messages)  # no tools passed
        raw_text = self._extract_text(response)
        return self._parse_intent(raw_text)

    # ------------------------------------------------------------------ private

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant message content string from an SDK response."""
        try:
            message = response.choices[0].message
            content = message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def _parse_intent(raw: str) -> Intent:
        """Normalize and validate *raw* into an Intent or fallback."""
        cleaned = raw.strip().lower()
        if not cleaned:
            return Intent.RESEARCH
        try:
            return Intent(cleaned)
        except ValueError:
            return Intent.RESEARCH
