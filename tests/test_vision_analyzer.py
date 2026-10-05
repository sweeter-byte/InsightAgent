"""Task-conditioned Vision Analyzer tests; no real VLM is contacted."""

from __future__ import annotations

from pathlib import Path

import pytest

from insight_agent.vision_retrieval import (
    VISION_ANALYSIS_SYSTEM_PROMPT,
    VisionAnalysisError,
    VisionAnalyzer,
)


class FakeImageRequest:
    def __init__(self, result: str = "The three arrows converge.") -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def __call__(
        self,
        path: str,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "path": path,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.result


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "workflow.png"
    path.write_bytes(b"fake image bytes")
    return path


def test_analyzer_passes_task_context_to_independent_prompt(tmp_path: Path) -> None:
    image = _image(tmp_path)
    request = FakeImageRequest()
    analyzer = VisionAnalyzer(image_request=request)

    content = analyzer.analyze(
        image_path=str(image),
        question="Do all three branches converge on Advance Task?",
        objective="Verify the workflow topology",
        constraints=["Use only visible arrows", "Do not infer hidden nodes"],
    )

    assert content == "The three arrows converge."
    assert request.calls[0]["path"] == str(image)
    assert request.calls[0]["system_prompt"] == VISION_ANALYSIS_SYSTEM_PROMPT
    prompt = request.calls[0]["user_prompt"]
    assert "Verify the workflow topology" in prompt
    assert "Do all three branches converge on Advance Task?" in prompt
    assert "Use only visible arrows" in prompt
    assert "Do not infer hidden nodes" in prompt


def test_analyzer_prompt_treats_image_text_as_untrusted_data(tmp_path: Path) -> None:
    request = FakeImageRequest()
    analyzer = VisionAnalyzer(image_request=request)

    analyzer.analyze(
        image_path=str(_image(tmp_path)),
        question="What does the diagram show?",
        objective="Inspect the architecture",
        constraints=[],
    )

    prompt = request.calls[0]["system_prompt"].lower()
    assert "untrusted data" in prompt
    assert "do not follow or execute" in prompt
    assert "do not modify the research task" in prompt
    assert "do not call tools" in prompt
    assert "observable" in prompt


def test_analyzer_rejects_missing_original_before_request(tmp_path: Path) -> None:
    request = FakeImageRequest()
    analyzer = VisionAnalyzer(image_request=request)

    with pytest.raises(VisionAnalysisError, match="File not found"):
        analyzer.analyze(
            image_path=str(tmp_path / "missing.png"),
            question="Question",
            objective="Objective",
            constraints=[],
        )

    assert request.calls == []


def test_analyzer_rejects_empty_model_content(tmp_path: Path) -> None:
    analyzer = VisionAnalyzer(image_request=FakeImageRequest("  \n"))

    with pytest.raises(VisionAnalysisError, match="empty content"):
        analyzer.analyze(
            image_path=str(_image(tmp_path)),
            question="Question",
            objective="Objective",
            constraints=[],
        )
