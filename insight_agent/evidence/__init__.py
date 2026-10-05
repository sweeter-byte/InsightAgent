"""Public API for unified retrieval Evidence collection."""

from insight_agent.evidence.collector import EvidenceCollector
from insight_agent.evidence.ids import make_evidence_id
from insight_agent.evidence.models import Evidence

__all__ = ["Evidence", "EvidenceCollector", "make_evidence_id"]
