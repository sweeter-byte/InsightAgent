"""Public API for evidence-grounded report generation."""

from insight_agent.reporting.citations import (
    CitationRegistry,
    assemble_report,
    build_citation,
)
from insight_agent.reporting.generator import (
    REPORT_GENERATOR_SYSTEM_PROMPT,
    ReportGenerator,
    select_report_candidates,
    validate_claims,
)
from insight_agent.reporting.models import (
    Citation,
    Claim,
    ReportGenerationError,
    ReportSection,
    StructuredReport,
)
from insight_agent.reporting.renderer import MarkdownReportRenderer

__all__ = [
    "REPORT_GENERATOR_SYSTEM_PROMPT",
    "Citation",
    "CitationRegistry",
    "Claim",
    "MarkdownReportRenderer",
    "ReportGenerationError",
    "ReportGenerator",
    "ReportSection",
    "StructuredReport",
    "assemble_report",
    "build_citation",
    "select_report_candidates",
    "validate_claims",
]
