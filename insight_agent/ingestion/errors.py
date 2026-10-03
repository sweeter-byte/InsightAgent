"""Ingestion-layer exceptions.

Single exception class to keep the error surface minimal. Loaders raise
``IngestionError`` for unrecoverable input problems (missing file, invalid
format, HTTP failure). Downstream callers can catch this one type instead of
juggling loader-specific hierarchies.
"""

from __future__ import annotations


class IngestionError(RuntimeError):
    """Raised when an ingestion loader cannot produce Documents from its input."""
