from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from evals.beir_scifact.dataset import (
    SCIFACT_ARCHIVE_MD5,
    SCIFACT_CORPUS_COUNT,
    SCIFACT_QUERY_COUNT,
    SciFactDataset,
    dataset_to_documents,
    load_scifact,
    safe_extract_zip,
    verify_extracted_files,
    verify_archive,
)


def _dataset() -> SciFactDataset:
    return SciFactDataset(
        corpus={
            "doc-1": {"title": "Alpha", "text": "First finding."},
            "doc-2": {"title": "Beta", "text": "Second finding."},
        },
        queries={"query-1": "Does alpha hold?"},
        qrels={"query-1": {"doc-1": 2}},
    )


def test_official_scifact_identity_is_pinned() -> None:
    assert SCIFACT_ARCHIVE_MD5 == "5f7d1de60b170fc8027bb7898e2efca1"
    assert SCIFACT_CORPUS_COUNT == 5183
    assert SCIFACT_QUERY_COUNT == 300


def test_load_scifact_preserves_official_ids(tmp_path: Path) -> None:
    expected = _dataset()

    class FakeLoader:
        def __init__(self, data_folder: str) -> None:
            assert data_folder == str(tmp_path)

        def load(self, split: str):
            assert split == "test"
            return expected.corpus, expected.queries, expected.qrels

    loaded = load_scifact(
        tmp_path,
        loader_factory=FakeLoader,
        expected_corpus_count=2,
        expected_query_count=1,
    )

    assert loaded == expected
    assert list(loaded.corpus) == ["doc-1", "doc-2"]
    assert list(loaded.queries) == ["query-1"]


def test_load_scifact_rejects_wrong_official_counts(tmp_path: Path) -> None:
    expected = _dataset()

    class FakeLoader:
        def __init__(self, data_folder: str) -> None:
            pass

        def load(self, split: str):
            return expected.corpus, expected.queries, expected.qrels

    with pytest.raises(ValueError, match="corpus count.*expected=5183.*actual=2"):
        load_scifact(tmp_path, loader_factory=FakeLoader)


def test_load_scifact_rejects_qrel_for_unknown_document(tmp_path: Path) -> None:
    expected = _dataset()
    qrels = {"query-1": {"missing": 1}}

    class FakeLoader:
        def __init__(self, data_folder: str) -> None:
            pass

        def load(self, split: str):
            return expected.corpus, expected.queries, qrels

    with pytest.raises(ValueError, match="unknown corpus document.*missing"):
        load_scifact(
            tmp_path,
            loader_factory=FakeLoader,
            expected_corpus_count=2,
            expected_query_count=1,
        )


def test_documents_keep_beir_document_id_and_join_title() -> None:
    documents = dataset_to_documents(_dataset())

    assert [document.metadata["beir_doc_id"] for document in documents] == [
        "doc-1",
        "doc-2",
    ]
    assert documents[0].content == "Alpha\n\nFirst finding."
    assert documents[0].source == "beir://scifact/doc-1"


def test_verify_archive_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(b"not the official archive")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(archive)


def test_verify_archive_accepts_explicit_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    archive.write_bytes(b"fixture")
    checksum = hashlib.md5(b"fixture").hexdigest()  # noqa: S324 - dataset checksum

    assert verify_archive(archive, expected_md5=checksum) == checksum


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_zip_extracts_regular_members(tmp_path: Path) -> None:
    archive = tmp_path / "ok.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("scifact/corpus.jsonl", "{}\n")
    archive.write_bytes(buffer.getvalue())

    root = safe_extract_zip(archive, tmp_path / "out")

    assert root == tmp_path / "out" / "scifact"
    assert (root / "corpus.jsonl").read_text() == "{}\n"


def test_verify_extracted_files_detects_modification(tmp_path: Path) -> None:
    archive = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("scifact/corpus.jsonl", "corpus\n")
        handle.writestr("scifact/queries.jsonl", "queries\n")
        handle.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\n")
    root = safe_extract_zip(archive, tmp_path / "out")
    assert set(verify_extracted_files(archive, root)) == {
        "corpus.jsonl",
        "queries.jsonl",
        "qrels/test.tsv",
    }

    (root / "queries.jsonl").write_text("modified\n")
    with pytest.raises(ValueError, match="differs from verified archive"):
        verify_extracted_files(archive, root)
