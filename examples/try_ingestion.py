#!/usr/bin/env python3
"""Minimal end-to-end example for the Chapter 3 Multimodal Ingestion layer.

It runs a single input through the unified ingestion entry point
(:func:`insight_agent.ingestion.ingest_file`) and prints a short summary of the
:class:`Document` list that comes back, then shows how the temporary
:func:`documents_to_context` adapter renders those Documents as prompt text.

The default input is a plain ``.txt`` file, so the script runs with **no API
keys and no network access**::

    python examples/try_ingestion.py                     # -> examples/result.txt
    python examples/try_ingestion.py path/to/notes.md     # .txt / .md need no config

PDF works offline too (PyMuPDF); one Document is produced per non-empty page::

    python examples/try_ingestion.py path/to/paper.pdf

Image ingestion calls a vision-language model and therefore requires the
``VISION_API_KEY`` / ``VISION_BASE_URL`` / ``VISION_MODEL`` environment
variables (see ``.env.example``). To avoid surprising API calls or cost, this
example only touches images when you *explicitly* pass an image file::

    python examples/try_ingestion.py path/to/chart.png

Scope note: this demonstrates ingestion + a dumb context dump. It is **NOT a
RAG demo** — there is no chunking, embedding, or retrieval anywhere in the
Chapter 3 layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insight_agent.ingestion import Document, IngestionError, documents_to_context, ingest_file

#: How many leading characters of each Document's content to preview.
PREVIEW_CHARS = 200

#: Repo-relative default so the example works from any current directory.
_DEFAULT_FILE = Path(__file__).resolve().parent / "result.txt"


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Return the first *limit* characters, with a marker when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [+{len(text) - limit} more characters]"


def summarize(documents: list[Document]) -> None:
    """Print count + per-document source_type / source / metadata / preview."""
    print(f"Documents: {len(documents)}")
    for idx, doc in enumerate(documents, start=1):
        source_type = getattr(doc.source_type, "value", doc.source_type)
        print(f"\n[Document {idx}]")
        print(f"  source_type : {source_type}")
        print(f"  source      : {doc.source}")
        print(f"  metadata    : {doc.metadata}")
        print(f"  content     : {_preview(doc.content)!r}")


def render_context(documents: list[Document]) -> None:
    """Show the temporary context-adapter output (a plain text dump, not RAG)."""
    block = documents_to_context(documents)
    print(f"\n--- documents_to_context() ({len(block)} chars; NOT RAG) ---")
    print(_preview(block) if block else "(empty)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest one file and print the resulting Documents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(_DEFAULT_FILE),
        help="file to ingest (.txt/.md/.pdf offline; images need VISION_* env). "
        "Defaults to examples/result.txt.",
    )
    args = parser.parse_args(argv)

    try:
        documents = ingest_file(args.path)
    except IngestionError as exc:
        print(f"[ingestion error] {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # describe_image() raises RuntimeError when VISION_* env vars are unset;
        # surface it as a friendly hint instead of a traceback.
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    summarize(documents)
    render_context(documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
