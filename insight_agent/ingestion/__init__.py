"""InsightAgent Multimodal Ingestion package.

Public API:
    Document, SourceType  — unified data model
    IngestionError        — single exception type
    ingest()              — unified entry point (URL or file path)
    load_text(), load_text_file(), load_markdown(), load_pdf(), load_url()
"""

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.ingest import ingest
from insight_agent.ingestion.loaders.markdown_loader import load_markdown
from insight_agent.ingestion.loaders.pdf_loader import load_pdf
from insight_agent.ingestion.loaders.text_loader import load_text, load_text_file
from insight_agent.ingestion.loaders.url_loader import load_url
from insight_agent.ingestion.models import Document, SourceType

__all__ = [
    "Document",
    "IngestionError",
    "SourceType",
    "ingest",
    "load_markdown",
    "load_pdf",
    "load_text",
    "load_text_file",
    "load_url",
]
