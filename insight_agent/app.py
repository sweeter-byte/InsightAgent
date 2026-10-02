"""High-level application entry point for InsightAgent.

`InsightAgent` is the single façade that external callers (CLI, Web API, …)
should use. It owns exactly one responsibility: **intent dispatch**.

Flow::

    user query
        ↓
    IntentRouter.route(query) → Intent
        ↓
    ┌──────────────┬───────────────────┬────────────────────┐
    │ DIRECT       │ ANALYZE           │ RESEARCH           │
    │  one plain   │  ResearchAgent    │  ResearchAgent     │
    │  chat call,  │  .run(query)      │  .run(query)       │
    │  no tools,   │                   │                    │
    │  no loop     │                   │                    │

`analyze` and `research` currently share the same downstream handler on
purpose — the intent label is preserved so future chapters can split them
without touching this dispatch site.

This module intentionally does NOT introduce handler registries, middleware,
DAGs, workflow engines, plug-in systems, DI frameworks, or LangGraph. It is
just a small composition object.
"""

from __future__ import annotations

from typing import Any

from insight_agent.agent import ResearchAgent
from insight_agent.llm import LLMClient
from insight_agent.router import Intent, IntentRouter


DIRECT_SYSTEM_PROMPT = (
    "You are InsightAgent, a helpful research assistant. "
    "Answer the user's question directly and clearly."
)


class InsightAgent:
    """Unified entry point that routes a query to the correct execution path.

    All collaborators are injected so tests can pass fakes/stubs and observe
    which branch was taken; ``run()`` never constructs real objects itself.
    """

    def __init__(
        self,
        *,
        router: IntentRouter,
        llm: LLMClient,
        research_agent: ResearchAgent,
        direct_system_prompt: str = DIRECT_SYSTEM_PROMPT,
    ) -> None:
        self.router = router
        self.llm = llm
        self.research_agent = research_agent
        self.direct_system_prompt = direct_system_prompt

    # ------------------------------------------------------------------ public

    def run(self, query: str) -> str:
        """Classify *query* then dispatch to the matching execution path."""
        intent = self.router.route(query)

        if intent is Intent.DIRECT:
            return self._answer_direct(query)

        # ANALYZE and RESEARCH both delegate to ResearchAgent for now; the two
        # intents remain separate so later chapters can fork them without
        # rewriting this dispatch site.
        return self.research_agent.run(query)

    # ----------------------------------------------------------------- private

    def _answer_direct(self, query: str) -> str:
        """One-shot chat completion — no tools, no agent loop, no registry."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.direct_system_prompt},
            {"role": "user", "content": query},
        ]
        # ``tools`` is deliberately omitted so the model cannot emit tool calls
        # and the ResearchAgent loop is never entered.
        response: Any = self.llm.chat(messages)
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant message content string from an SDK response."""
        try:
            message = response.choices[0].message
            content = message.content
            return content if isinstance(content, str) else ""
        except (AttributeError, IndexError, TypeError):
            return ""
