"""Public errors raised by the research runtime layer."""

from __future__ import annotations


class ResearchRuntimeError(RuntimeError):
    """Base class for runtime-layer failures."""


class RunNotFound(ResearchRuntimeError, LookupError):
    """Raised when a public run identifier does not exist."""


class InvalidRunState(ResearchRuntimeError):
    """Raised when an operation is invalid for the run's current state."""


class DuplicateRun(ResearchRuntimeError):
    """Raised when a run identifier already exists."""


class RuntimeUnavailable(ResearchRuntimeError):
    """Raised after the service has stopped accepting new work."""


class TransientRuntimeError(ResearchRuntimeError):
    """A narrowly classified infrastructure error that may be retried."""
