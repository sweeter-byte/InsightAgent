"""Finite domain models for report self-check results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SelfCheckStatus(str, Enum):
    """The only terminal decisions a report self-check may return."""

    PASS = "pass"
    REVISE = "revise"


class SelfCheckIssueCode(str, Enum):
    """The bounded set of report-fidelity problems checked in Chapter 14."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_MISMATCH = "citation_mismatch"
    MISSING_GAP_DISCLOSURE = "missing_gap_disclosure"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INCONSISTENT_CLAIMS = "inconsistent_claims"


class SelfCheckError(RuntimeError):
    """Raised when self-check or controlled repair violates its contract."""


@dataclass(slots=True)
class SelfCheckIssue:
    """One actionable fidelity problem in a structured report."""

    code: SelfCheckIssueCode
    reason: str
    task_id: str | None = None
    claim_id: str | None = None


@dataclass(slots=True)
class SelfCheckResult:
    """A finite pass-or-revise decision with validated issues."""

    status: SelfCheckStatus
    issues: list[SelfCheckIssue]
    summary: str
