"""Tests for the vision-language extraction layer (``insight_agent.ingestion.vision``).

These NEVER hit a real vision API. The OpenAI-compatible client is mocked so we
can assert on the request body (image Data URL + text prompt) that we would
send, and verify error/empty-response handling. We do not test the third-party
SDK itself.
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.vision import (
    VISION_SYSTEM_PROMPT,
    describe_image,
    guess_mime_type,
)

_VISION_ENV = {
    "VISION_API_KEY": "sk-vision-dummy",
    "VISION_BASE_URL": "https://vision.example.invalid/v1",
    "VISION_MODEL": "vision-test-model",
}


def _set_vision_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _VISION_ENV.items():
        monkeypatch.setenv(key, value)


def _write_image(tmp_path: Path, name: str = "fig.png", payload: bytes = b"fake-bytes") -> Path:
    f = tmp_path / name
    f.write_bytes(payload)
    return f


def _fake_completion(content: Any) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---------------------------------------------------------------------------
# guess_mime_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.png", "image/png"),
        ("a.jpg", "image/jpeg"),
        ("a.jpeg", "image/jpeg"),
        ("a.webp", "image/webp"),
        ("A.PNG", "image/png"),  # case-insensitive
    ],
)
def test_guess_mime_type_supported(filename: str, expected: str) -> None:
    assert guess_mime_type(f"/tmp/{filename}") == expected


@pytest.mark.parametrize("extension", ["gif", "bmp", "svg"])
def test_guess_mime_type_unsupported_image_format_raises(extension: str) -> None:
    with pytest.raises(IngestionError, match="Unsupported or unrecognized image format"):
        guess_mime_type(f"/tmp/image.{extension}")


# ---------------------------------------------------------------------------
# describe_image — request body construction (mocked client)
# ---------------------------------------------------------------------------


def test_describe_image_builds_correct_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_vision_env(monkeypatch)
    payload = b"\x89PNG\r\n\x1a\nhello"
    img = _write_image(tmp_path, "chart.png", payload)

    captured: dict[str, Any] = {}

    def _create(model: str, messages: list[dict[str, Any]]) -> SimpleNamespace:
        captured["model"] = model
        captured["messages"] = messages
        return _fake_completion("A line chart of accuracy over epochs.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _create

    with patch("insight_agent.ingestion.vision.OpenAI", return_value=mock_client) as openai_cls:
        text = describe_image(str(img))

    # client built from VISION_* config
    openai_cls.assert_called_once_with(
        api_key=_VISION_ENV["VISION_API_KEY"],
        base_url=_VISION_ENV["VISION_BASE_URL"],
    )
    assert captured["model"] == _VISION_ENV["VISION_MODEL"]
    assert text == "A line chart of accuracy over epochs."

    msgs = captured["messages"]
    # system prompt is the detailed fidelity/chart/text guidance, not a vague line
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == VISION_SYSTEM_PROMPT
    assert "axes" in VISION_SYSTEM_PROMPT.lower()
    assert "never invent" in VISION_SYSTEM_PROMPT.lower()

    # user turn carries BOTH a text prompt and the image input
    user_content = msgs[1]["content"]
    assert isinstance(user_content, list)
    kinds = {part["type"] for part in user_content}
    assert kinds == {"text", "image_url"}

    text_part = next(p for p in user_content if p["type"] == "text")
    assert isinstance(text_part["text"], str) and text_part["text"]

    image_part = next(p for p in user_content if p["type"] == "image_url")
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # base64 payload decodes back to the exact original bytes
    assert base64.b64decode(url.split(",", 1)[1]) == payload


# ---------------------------------------------------------------------------
# describe_image — response / error handling
# ---------------------------------------------------------------------------


def test_describe_image_returns_empty_string_on_empty_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_vision_env(monkeypatch)
    img = _write_image(tmp_path)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_completion(None)

    with patch("insight_agent.ingestion.vision.OpenAI", return_value=mock_client):
        assert describe_image(str(img)) == ""


def test_describe_image_api_failure_raises_ingestion_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_vision_env(monkeypatch)
    img = _write_image(tmp_path)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("boom: rate limited")

    with patch("insight_agent.ingestion.vision.OpenAI", return_value=mock_client):
        with pytest.raises(IngestionError, match="Vision model call failed"):
            describe_image(str(img))


def test_describe_image_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_vision_env(monkeypatch)
    with pytest.raises(IngestionError, match="File not found"):
        describe_image(str(tmp_path / "ghost.png"))


def test_describe_image_directory_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_vision_env(monkeypatch)
    with pytest.raises(IngestionError, match="Not a regular file"):
        describe_image(str(tmp_path))


def test_describe_image_empty_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_vision_env(monkeypatch)
    img = _write_image(tmp_path, "zero.png", payload=b"")
    with pytest.raises(IngestionError, match="empty"):
        describe_image(str(img))


@pytest.mark.parametrize("missing", ["VISION_API_KEY", "VISION_BASE_URL", "VISION_MODEL"])
def test_describe_image_missing_env_raises_ingestion_error_with_cause(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_vision_env(monkeypatch)
    monkeypatch.delenv(missing)
    img = _write_image(tmp_path)
    with pytest.raises(IngestionError, match=f"Vision configuration error:.*{missing}") as excinfo:
        describe_image(str(img))

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert missing in str(excinfo.value.__cause__)
