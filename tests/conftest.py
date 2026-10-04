"""Shared deterministic environment for the network-free test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _configured_test_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let default registry tests build without loading a real model."""
    monkeypatch.setenv("RERANKER_MODEL", "tests/fake-reranker")
