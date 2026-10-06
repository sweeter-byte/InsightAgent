"""Deterministic Markdown rendering for structured research reports."""

from __future__ import annotations

from insight_agent.reporting.models import Citation, StructuredReport


class MarkdownReportRenderer:
    """Render a StructuredReport without invoking models or retrieval."""

    def render(self, report: StructuredReport) -> str:
        number_by_id = {
            citation.evidence_id: citation.number for citation in report.citations
        }
        lines = ["# 研究结果", "", "## 研究目标", "", report.objective]
        for section in report.sections:
            lines.extend(["", f"## {section.title}", ""])
            if section.claims:
                for claim in section.claims:
                    markers = "".join(
                        f"[{number_by_id[evidence_id]}]"
                        for evidence_id in claim.evidence_ids
                    )
                    lines.append(f"{claim.text} {markers}")
                    lines.append("")
                lines.pop()
            else:
                lines.append("当前没有可靠结论。")
            if section.missing_information:
                lines.extend(["", "### 证据缺口", ""])
                lines.extend(f"- {item}" for item in section.missing_information)

        lines.extend(["", "## 参考来源"])
        if report.citations:
            lines.append("")
            lines.extend(self._render_citation(item) for item in report.citations)
        else:
            lines.extend(["", "当前没有可引用的来源。"])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_citation(citation: Citation) -> str:
        if citation.source.startswith(("http://", "https://")):
            body = f"{citation.label}，{citation.source}"
        elif citation.locator == "图片":
            body = f"{citation.label}（图片）"
        elif citation.locator:
            body = f"{citation.label}，{citation.locator}"
        else:
            body = citation.label
        return f"[{citation.number}] {body}"
