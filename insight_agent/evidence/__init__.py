"""Public API for unified retrieval Evidence collection."""

from insight_agent.evidence.collector import EvidenceCollector
from insight_agent.evidence.grader import (
    EVIDENCE_GRADER_SYSTEM_PROMPT,
    EvidenceGrader,
)
from insight_agent.evidence.ids import make_evidence_id
from insight_agent.evidence.models import (
    Evidence,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceGradingError,
    EvidenceJudgment,
    EvidenceQuality,
    EvidenceRelevance,
)

__all__ = [
    "Evidence",
    "EvidenceAssessment",
    "EvidenceCollector",
    "EvidenceCoverage",
    "EvidenceGrader",
    "EvidenceGradingError",
    "EvidenceJudgment",
    "EvidenceQuality",
    "EvidenceRelevance",
    "EVIDENCE_GRADER_SYSTEM_PROMPT",
    "make_evidence_id",
]
