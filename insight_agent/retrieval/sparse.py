"""In-memory BM25 retrieval derived from Qdrant-backed Chunks."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from insight_agent.indexing import Chunk
from insight_agent.ingestion.models import normalize_source_types
from insight_agent.retrieval.models import RetrievalResult


_TOKEN_PATTERN = re.compile(
    r"(?:/[A-Za-z0-9._-]+)+/?"
    r"|[A-Za-z][A-Za-z0-9_]*"
    r"|\d+(?:\.\d+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_CHINESE_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text while keeping technical identifiers intact."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if _CHINESE_PATTERN.fullmatch(value):
            tokens.extend(
                token.strip().casefold()
                for token in jieba.lcut(value, cut_all=False)
                if token.strip()
            )
        else:
            tokens.append(value.casefold())
    return tokens


@dataclass(frozen=True, slots=True)
class _BM25Snapshot:
    chunks: tuple[Chunk, ...]
    index: Any | None


class _PositiveIDFBM25Okapi(BM25Okapi):
    """Okapi BM25 with the common non-negative log(1 + RSJ) IDF."""

    def _calc_idf(self, nd: dict[str, int]) -> None:
        idf_sum = 0.0
        for word, frequency in nd.items():
            idf = math.log1p(
                (self.corpus_size - frequency + 0.5) / (frequency + 0.5)
            )
            self.idf[word] = idf
            idf_sum += idf
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0.0


class BM25Retriever:
    """Reuse one lock-swapped BM25 snapshot across many queries."""

    def __init__(
        self,
        chunk_loader: Callable[[], list[Chunk]],
        *,
        tokenizer: Callable[[str], list[str]] = tokenize,
        bm25_factory: Callable[[list[list[str]]], Any] = _PositiveIDFBM25Okapi,
    ) -> None:
        self._chunk_loader = chunk_loader
        self._tokenizer = tokenizer
        self._bm25_factory = bm25_factory
        self._snapshot_lock = threading.Lock()
        self._snapshot = _BM25Snapshot(chunks=(), index=None)
        self.refresh()

    def refresh(self) -> None:
        """Build a complete replacement before atomically publishing it."""
        tokenized = [
            (chunk, self._tokenizer(chunk.content))
            for chunk in self._chunk_loader()
        ]
        searchable = [(chunk, tokens) for chunk, tokens in tokenized if tokens]
        chunks = tuple(chunk for chunk, _tokens in searchable)
        corpus = [tokens for _chunk, tokens in searchable]
        index = self._bm25_factory(corpus) if corpus else None
        replacement = _BM25Snapshot(chunks=chunks, index=index)
        with self._snapshot_lock:
            self._snapshot = replacement

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Rank the current snapshot with the same tokenizer used at build time."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        normalized_types = normalize_source_types(source_types)

        with self._snapshot_lock:
            snapshot = self._snapshot
        if snapshot.index is None:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []
        scores = snapshot.index.get_scores(query_tokens)
        eligible_indices = (
            range(len(snapshot.chunks))
            if normalized_types is None
            else (
                index
                for index, chunk in enumerate(snapshot.chunks)
                if chunk.source_type.value in normalized_types
            )
        )
        ranked_indices = sorted(
            eligible_indices,
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [
            self._to_result(snapshot.chunks[index], float(scores[index]))
            for index in ranked_indices
        ]

    @staticmethod
    def _to_result(chunk: Chunk, score: float) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk.id,
            score=score,
            content=chunk.content,
            document_id=chunk.document_id,
            source=chunk.source,
            source_type=chunk.source_type,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            metadata=dict(chunk.metadata),
        )
