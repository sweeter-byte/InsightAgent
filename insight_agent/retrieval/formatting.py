"""Formatting of retrieval results for an Agent tool observation."""

from __future__ import annotations

import json

from insight_agent.retrieval.models import RetrievalResult


EMPTY_RESULTS_MESSAGE = "知识库中没有找到相关内容。"


def format_results(results: list[RetrievalResult]) -> str:
    """Render ranked results without separating content from provenance."""
    if not results:
        return EMPTY_RESULTS_MESSAGE

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        metadata = json.dumps(
            result.metadata,
            ensure_ascii=False,
            sort_keys=True,
        )
        blocks.append(
            "\n".join(
                [
                    f"[Result {index}]",
                    f"score: {result.score:.4f}",
                    f"chunk_id: {result.chunk_id}",
                    f"document_id: {result.document_id}",
                    f"source: {result.source}",
                    f"source_type: {result.source_type.value}",
                    f"chunk_index: {result.chunk_index}",
                    f"char_range: {result.start_char}-{result.end_char}",
                    f"metadata: {metadata}",
                    "content:",
                    result.content,
                ]
            )
        )
    return "\n\n".join(blocks)
