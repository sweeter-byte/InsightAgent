"""Ingestion context helpers.

Two small utilities shared across the ingestion layer:

  * :func:`infer_file_type` — source-type inference from a file extension.
  * :func:`documents_to_context` — render ``Document`` objects into a single
    text block for temporary LLM integration (Chapter 3 smoke-testing only).

No class hierarchies, no DI.

File-type inference is deliberately *extension-only*: no magic-number
sniffing, no libmagic dependency. Callers that need more robust detection
should look at the file themselves before handing a path to a loader.
"""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType

_EXT_MAP: dict[str, SourceType] = {
    ".txt": SourceType.TEXT,
    ".text": SourceType.TEXT,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".pdf": SourceType.PDF,
    ".png": SourceType.IMAGE,
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
}


def infer_file_type(path: str) -> SourceType:
    """Infer the SourceType from a file path's extension.

    Args:
        path: filesystem path string.

    Returns:
        The mapped SourceType.

    Raises:
        IngestionError: if the extension is unrecognized.
    """
    suffix = Path(path).suffix.lower()
    st = _EXT_MAP.get(suffix)
    if st is None:
        supported = ", ".join(sorted(_EXT_MAP))
        raise IngestionError(
            f"Unrecognized file extension: {suffix!r} (from {path!r}); "
            f"supported extensions: {supported}"
        )
    return st


def documents_to_context(documents: list[Document]) -> str:
    """Render a list of Documents into one plain-text block for a prompt.

    Temporary Chapter 3 integration helper — it lets the already-built LLM /
    Agent stack *consume* ingested Documents so we can smoke-test the whole
    ingestion path end to end. Output shape::

        [Document 1]
        source: path/to/file.txt
        source_type: text
        metadata: filename=file.txt
        <verbatim content>

        [Document 2]
        ...

    .. warning::
        **This is NOT RAG.** It is a naive dump and stays deliberately dumb:

          * No chunking, no embedding, no vector index, no retrieval, no
            relevance ranking — whatever list you pass in is emitted verbatim,
            in the order given.
          * No token budget and no context compression: every character of
            every Document's content is included.

        Do **not** feed a large Document set straight into an Agent context
        through this helper — with no budget or filtering it can easily
        overflow the model window or inflate cost. Real retrieval / context
        assembly under a budget is future work outside this chapter.

    Args:
        documents: the Documents to render (may be empty).

    Returns:
        A single text block; an empty string when *documents* is empty.
    """
    if not documents:
        return ""

    blocks: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        # ``SourceType`` is a ``str`` enum; use ``.value`` for a clean label.
        source_type = getattr(doc.source_type, "value", doc.source_type)
        # Sorted keys keep the output deterministic (matters for tests/diffs).
        metadata = ", ".join(f"{k}={doc.metadata[k]}" for k in sorted(doc.metadata))
        blocks.append(
            f"[Document {idx}]\n"
            f"source: {doc.source}\n"
            f"source_type: {source_type}\n"
            f"metadata: {metadata or '{}'}\n"
            f"{doc.content}"
        )
    return "\n\n".join(blocks)
