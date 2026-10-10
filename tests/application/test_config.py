from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from insight_agent.application import AppConfig, CheckpointConfig, MaterialConfig
from insight_agent.indexing import DEFAULT_EMBEDDING_MODEL


def _required_env() -> dict[str, str]:
    return {
        "LLM_API_KEY": "llm-secret",
        "LLM_BASE_URL": "https://llm.example/v1",
        "LLM_MODEL": "text-model",
        "RERANKER_MODEL": "example/reranker",
    }


def test_from_env_populates_complete_explicit_mapping() -> None:
    config = AppConfig.from_env(
        {
            "LLM_API_KEY": " llm-secret ",
            "LLM_BASE_URL": " https://llm.example/v1 ",
            "LLM_MODEL": " text-model ",
            "VISION_API_KEY": " vision-secret ",
            "VISION_BASE_URL": " https://vision.example/v1 ",
            "VISION_MODEL": " vision-model ",
            "HYBRID_DENSE_K": "31",
            "HYBRID_SPARSE_K": "29",
            "HYBRID_RERANK_K": "17",
            "HYBRID_FINAL_TOP_K": "7",
            "HYBRID_RRF_K": "41",
            "RERANKER_MODEL": " example/reranker ",
            "EMBEDDING_MODEL": " example/embedder ",
            "QDRANT_PATH": " custom/qdrant ",
            "QDRANT_COLLECTION": " custom_documents ",
            "TAVILY_API_KEY": " tavily-secret ",
            "WEB_SEARCH_TIMEOUT": "12.5",
            "WEB_SEARCH_LIMIT": "8",
            "WEB_FETCH_LIMIT": "4",
            "CHECKPOINT_PATH": " custom/checkpoints.sqlite ",
            "RUNTIME_REDIS_URL": " redis://runtime.example/8 ",
            "RUNTIME_MAX_CONCURRENCY": "4",
            "RUNTIME_RUN_TIMEOUT_SECONDS": "12.5",
            "RUNTIME_INFRA_RETRY_ATTEMPTS": "5",
            "RUNTIME_INFRA_RETRY_BACKOFF_SECONDS": "0.75",
            "RUNTIME_EVENT_TTL_SECONDS": "123",
            "MATERIAL_UPLOAD_DIR": " custom/materials ",
            "MATERIAL_MAX_BYTES": "42",
        }
    )

    assert config.llm.api_key == "llm-secret"
    assert config.llm.base_url == "https://llm.example/v1"
    assert config.llm.model == "text-model"
    assert config.vision is not None
    assert config.vision.api_key == "vision-secret"
    assert config.vision.base_url == "https://vision.example/v1"
    assert config.vision.model == "vision-model"
    assert config.retrieval.dense_k == 31
    assert config.retrieval.sparse_k == 29
    assert config.retrieval.rerank_k == 17
    assert config.retrieval.final_top_k == 7
    assert config.retrieval.rrf_k == 41
    assert config.retrieval.reranker_model == "example/reranker"
    assert config.embedding_model == "example/embedder"
    assert config.qdrant.path == Path("custom/qdrant")
    assert config.qdrant.collection_name == "custom_documents"
    assert config.web_search is not None
    assert config.web_search.api_key == "tavily-secret"
    assert config.web_search.timeout == 12.5
    assert config.web_search.search_limit == 8
    assert config.web_search.fetch_limit == 4
    assert config.checkpoint.path == Path("custom/checkpoints.sqlite")
    assert config.runtime.redis_url == "redis://runtime.example/8"
    assert config.runtime.policy.max_concurrency == 4
    assert config.runtime.policy.run_timeout_seconds == 12.5
    assert config.runtime.policy.infra_retry_attempts == 5
    assert config.runtime.policy.infra_retry_backoff_seconds == 0.75
    assert config.runtime.policy.event_ttl_seconds == 123
    assert config.materials == MaterialConfig(
        upload_dir=Path("custom/materials"), max_bytes=42
    )


def test_optional_vision_and_web_search_are_absent() -> None:
    config = AppConfig.from_env(_required_env())

    assert config.vision is None
    assert config.web_search is None


def test_from_env_has_stable_component_defaults() -> None:
    values = _required_env()
    values.update(
        {
            "EMBEDDING_MODEL": "   ",
            "CHECKPOINT_PATH": "   ",
        }
    )

    config = AppConfig.from_env(values)

    assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-m3"
    assert config.embedding_model == "BAAI/bge-m3"
    assert config.retrieval.dense_k == 20
    assert config.retrieval.sparse_k == 20
    assert config.retrieval.rerank_k == 20
    assert config.retrieval.final_top_k == 5
    assert config.retrieval.rrf_k == 60
    assert config.retrieval.reranker_model == "example/reranker"
    assert config.qdrant.path == Path(".data/qdrant")
    assert config.qdrant.collection_name == "insight_documents"
    assert config.checkpoint.path == Path(".insight_agent/checkpoints.sqlite")
    assert config.runtime.redis_url == "redis://localhost:6379/0"
    assert config.runtime.policy.max_concurrency == 2
    assert config.runtime.policy.run_timeout_seconds == 900.0
    assert config.runtime.policy.infra_retry_attempts == 3
    assert config.runtime.policy.infra_retry_backoff_seconds == 0.2
    assert config.runtime.policy.event_ttl_seconds == 86_400
    assert config.materials == MaterialConfig(
        upload_dir=Path(".data/materials"), max_bytes=20 * 1024 * 1024
    )


def test_repr_does_not_expose_provider_secrets() -> None:
    values = _required_env()
    values.update(
        {
            "LLM_API_KEY": "llm-secret-that-must-not-leak",
            "VISION_API_KEY": "vision-secret-that-must-not-leak",
            "VISION_BASE_URL": "https://vision.example/v1",
            "VISION_MODEL": "vision-model",
            "TAVILY_API_KEY": "tavily-secret-that-must-not-leak",
        }
    )

    rendered = repr(AppConfig.from_env(values))

    assert "llm-secret-that-must-not-leak" not in rendered
    assert "vision-secret-that-must-not-leak" not in rendered
    assert "tavily-secret-that-must-not-leak" not in rendered


def test_from_env_reuses_hybrid_validation() -> None:
    values = _required_env()
    values["HYBRID_FINAL_TOP_K"] = "9"

    with pytest.raises(RuntimeError, match="HYBRID_FINAL_TOP_K"):
        AppConfig.from_env(values)


def test_from_env_reuses_partial_vision_validation() -> None:
    values = _required_env()
    values["VISION_API_KEY"] = "vision-secret"

    with pytest.raises(RuntimeError, match="VISION_BASE_URL"):
        AppConfig.from_env(values)


def test_explicit_mapping_is_isolated_from_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "ignored-process-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://ignored.example/v1")
    monkeypatch.setenv("LLM_MODEL", "ignored-model")
    monkeypatch.setenv("RERANKER_MODEL", "ignored/reranker")
    monkeypatch.setenv("EMBEDDING_MODEL", "ignored/embedder")
    monkeypatch.setenv("VISION_API_KEY", "ignored-vision-secret")
    monkeypatch.setenv("VISION_BASE_URL", "https://ignored-vision.example/v1")
    monkeypatch.setenv("VISION_MODEL", "ignored-vision-model")
    monkeypatch.setenv("TAVILY_API_KEY", "ignored-tavily-secret")
    monkeypatch.setattr(
        "insight_agent.application.config.load_dotenv",
        lambda: pytest.fail("load_dotenv must not run for explicit mappings"),
    )

    config = AppConfig.from_env(_required_env())

    assert config.llm.model == "text-model"
    assert config.retrieval.reranker_model == "example/reranker"
    assert config.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert config.vision is None
    assert config.web_search is None


@pytest.mark.parametrize("embedding_model", ["", "   ", None])
def test_direct_app_config_rejects_invalid_embedding_model(
    embedding_model: object,
) -> None:
    config = AppConfig.from_env(_required_env())

    with pytest.raises(ValueError, match="embedding_model"):
        replace(config, embedding_model=embedding_model)


@pytest.mark.parametrize("path", ["", "   ", None])
def test_direct_checkpoint_config_rejects_empty_path(path: object) -> None:
    with pytest.raises(ValueError, match="CHECKPOINT_PATH"):
        CheckpointConfig(path=path)


def test_application_configs_are_frozen_and_slotted() -> None:
    checkpoint = CheckpointConfig()
    config = AppConfig.from_env(_required_env())

    with pytest.raises(FrozenInstanceError):
        checkpoint.path = Path("other.sqlite")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.embedding_model = "other/model"  # type: ignore[misc]
    assert not hasattr(checkpoint, "__dict__")
    assert not hasattr(config, "__dict__")


@pytest.mark.parametrize("path", ["", "   ", None])
def test_material_config_rejects_empty_upload_dir(path: object) -> None:
    with pytest.raises(ValueError, match="MATERIAL_UPLOAD_DIR"):
        MaterialConfig(upload_dir=path)  # type: ignore[arg-type]


def test_material_config_mapping_rejects_empty_upload_dir() -> None:
    with pytest.raises(ValueError, match="MATERIAL_UPLOAD_DIR"):
        AppConfig.from_env({**_required_env(), "MATERIAL_UPLOAD_DIR": "   "})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "invalid"])
def test_material_config_rejects_invalid_max_bytes(value: object) -> None:
    with pytest.raises(ValueError, match="MATERIAL_MAX_BYTES"):
        MaterialConfig(max_bytes=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_material_config_mapping_rejects_invalid_max_bytes(value: str) -> None:
    with pytest.raises(ValueError, match="MATERIAL_MAX_BYTES"):
        AppConfig.from_env({**_required_env(), "MATERIAL_MAX_BYTES": value})
