"""Errors exposed by task-conditioned Vision Retrieval."""

from __future__ import annotations


class VisionRetrievalError(RuntimeError):
    """Base error for failures that prevent Vision Retrieval execution."""


class VisionAnalysisError(VisionRetrievalError):
    """Raised when one original image cannot produce a usable analysis."""
