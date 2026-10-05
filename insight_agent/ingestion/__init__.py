"""InsightAgent Multimodal Ingestion package.

Public API — this is the only import surface upper-layer code should use:
    Document, SourceType           — unified data model
    IngestionError                 — single exception type
    ingest()                       — unified entry point (explicit SourceType)
    ingest_file()                  — local file entry point (type from extension)
    ingest_text_file()             — plain-text file convenience
    infer_file_type()              — extension → SourceType
    documents_to_context()         — Documents → prompt text (NOT RAG; Ch3 adapter)
    safe_ingest()                  — minimal error boundary around ingest()
    load_text(), load_text_file(), load_markdown(), load_pdf(), load_url(),
    load_image()                   — per-format loaders (escape hatches)
    describe_image()               — image → VLM textual description

The modules under ``ingestion.loaders`` are internal implementation details;
depending on them directly (e.g. ``ingestion.loaders.pdf_loader``) couples
callers to a layout that is free to change.
"""

from insight_agent.ingestion.context import documents_to_context, infer_file_type
from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.ingest import ingest, ingest_file, ingest_text_file, safe_ingest
from insight_agent.ingestion.loaders.image_loader import load_image
from insight_agent.ingestion.loaders.markdown_loader import load_markdown
from insight_agent.ingestion.loaders.pdf_loader import load_pdf
from insight_agent.ingestion.loaders.text_loader import load_text, load_text_file
from insight_agent.ingestion.loaders.url_loader import load_url
from insight_agent.ingestion.models import Document, SourceType
from insight_agent.ingestion.vision import (
    OpenAICompatibleVisionClient,
    VisionModelConfig,
    analyze_image_with_prompt,
    describe_image,
    guess_mime_type,
)

__all__ = [
    "Document",
    "IngestionError",
    "OpenAICompatibleVisionClient",
    "SourceType",
    "VisionModelConfig",
    "analyze_image_with_prompt",
    "describe_image",
    "documents_to_context",
    "infer_file_type",
    "guess_mime_type",
    "ingest",
    "ingest_file",
    "ingest_text_file",
    "load_image",
    "load_markdown",
    "load_pdf",
    "load_text",
    "load_text_file",
    "load_url",
    "safe_ingest",
]
