"""Tests for the small embedding boundary and lazy model adapter."""

from __future__ import annotations

from typing import Any

from insight_agent.indexing import SentenceTransformerEmbedder


class _ArrayResult:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


class _FakeModel:
    def __init__(self, result: list[list[float]]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> _ArrayResult:
        self.calls.append({"texts": list(texts), **kwargs})
        return _ArrayResult(self.result)


def test_constructor_does_not_load_model(monkeypatch) -> None:
    loads: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            loads.append(model_name)

    monkeypatch.setattr(
        "insight_agent.indexing.embedder._load_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )

    SentenceTransformerEmbedder(model_name="test/model")

    assert loads == []


def test_embed_documents_loads_once_and_encodes_one_batch(monkeypatch) -> None:
    model = _FakeModel([[1, 2], [3.5, 4]])
    loads: list[str] = []

    class FakeSentenceTransformer:
        def __new__(cls, model_name: str) -> _FakeModel:
            loads.append(model_name)
            return model

    monkeypatch.setattr(
        "insight_agent.indexing.embedder._load_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )
    embedder = SentenceTransformerEmbedder(model_name="test/model")

    first = embedder.embed_documents(["alpha", "beta"])
    second = embedder.embed_documents(["gamma"])

    assert loads == ["test/model"]
    assert first == [[1.0, 2.0], [3.5, 4.0]]
    assert second == [[1.0, 2.0], [3.5, 4.0]]
    assert model.calls == [
        {
            "texts": ["alpha", "beta"],
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
        {
            "texts": ["gamma"],
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        },
    ]


def test_default_model_comes_from_environment(monkeypatch) -> None:
    created_with: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            created_with.append(model_name)

        def encode(self, texts: list[str], **kwargs: Any) -> _ArrayResult:
            return _ArrayResult([[1.0] for _ in texts])

    monkeypatch.setenv("EMBEDDING_MODEL", "configured/model")
    monkeypatch.setattr(
        "insight_agent.indexing.embedder._load_sentence_transformer_class",
        lambda: FakeSentenceTransformer,
    )

    SentenceTransformerEmbedder().embed_documents(["text"])

    assert created_with == ["configured/model"]


def test_empty_batch_returns_without_loading_model(monkeypatch) -> None:
    def fail_if_loaded() -> type[object]:
        raise AssertionError("model should not load for an empty batch")

    monkeypatch.setattr(
        "insight_agent.indexing.embedder._load_sentence_transformer_class",
        fail_if_loaded,
    )

    assert SentenceTransformerEmbedder().embed_documents([]) == []

