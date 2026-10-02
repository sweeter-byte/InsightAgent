"""InsightAgent: a minimal Research Agent built on OpenAI-compatible Chat Completions."""

from insight_agent.agent import ResearchAgent
from insight_agent.llm import LLMClient

__all__ = ["LLMClient", "ResearchAgent"]
__version__ = "0.1.0"
