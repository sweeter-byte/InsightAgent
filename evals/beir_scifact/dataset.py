"""Official SciFact download, validation, and InsightAgent adaptation."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from insight_agent.ingestion import Document, SourceType


SCIFACT_DATASET_NAME = "scifact"
SCIFACT_DATASET_VERSION = "beir-scifact-test-md5-5f7d1de6"
SCIFACT_ARCHIVE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
    "datasets/scifact.zip"
)
SCIFACT_ARCHIVE_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SCIFACT_CORPUS_COUNT = 5183
SCIFACT_QUERY_COUNT = 300


Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class SciFactDataset:
    """Unmodified identifiers and relevance labels from the BEIR test split."""

    corpus: Corpus
    queries: Queries
    qrels: Qrels


class _Loader(Protocol):
    def load(self, split: str) -> tuple[Corpus, Queries, Qrels]: ...


def _beir_loader(data_folder: str) -> _Loader:
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise RuntimeError(
            'BEIR benchmark dependencies are missing; run: pip install -e ".[benchmark]"'
        ) from exc
    return GenericDataLoader(data_folder=data_folder)


def load_scifact(
    data_path: str | Path,
    *,
    loader_factory: Callable[[str], _Loader] = _beir_loader,
    expected_corpus_count: int = SCIFACT_CORPUS_COUNT,
    expected_query_count: int = SCIFACT_QUERY_COUNT,
) -> SciFactDataset:
    """Load and validate the official test split without rewriting its IDs."""
    root = Path(data_path)
    corpus_raw, queries_raw, qrels_raw = loader_factory(str(root)).load(split="test")
    corpus = {
        str(doc_id): {"title": str(value.get("title", "")), "text": str(value["text"])}
        for doc_id, value in corpus_raw.items()
    }
    queries = {str(query_id): str(text) for query_id, text in queries_raw.items()}
    qrels = {
        str(query_id): {str(doc_id): int(relevance) for doc_id, relevance in values.items()}
        for query_id, values in qrels_raw.items()
    }
    dataset = SciFactDataset(corpus=corpus, queries=queries, qrels=qrels)
    validate_scifact(
        dataset,
        expected_corpus_count=expected_corpus_count,
        expected_query_count=expected_query_count,
    )
    return dataset


def validate_scifact(
    dataset: SciFactDataset,
    *,
    expected_corpus_count: int = SCIFACT_CORPUS_COUNT,
    expected_query_count: int = SCIFACT_QUERY_COUNT,
) -> None:
    """Reject incomplete or referentially invalid benchmark data."""
    if len(dataset.corpus) != expected_corpus_count:
        raise ValueError(
            "SciFact corpus count mismatch: "
            f"expected={expected_corpus_count}, actual={len(dataset.corpus)}"
        )
    if len(dataset.queries) != expected_query_count:
        raise ValueError(
            "SciFact query count mismatch: "
            f"expected={expected_query_count}, actual={len(dataset.queries)}"
        )
    for doc_id, record in dataset.corpus.items():
        if not doc_id.strip() or not isinstance(record, Mapping):
            raise ValueError("SciFact corpus contains an invalid document ID or record")
        if not str(record.get("text", "")).strip():
            raise ValueError(f"SciFact document {doc_id!r} has empty text")
    for query_id, query in dataset.queries.items():
        if not query_id.strip() or not query.strip():
            raise ValueError("SciFact queries contain an empty ID or text")
        if query_id not in dataset.qrels:
            raise ValueError(f"SciFact query {query_id!r} has no test qrels")
    unknown_queries = set(dataset.qrels) - set(dataset.queries)
    if unknown_queries:
        raise ValueError(
            f"SciFact qrels reference unknown query {sorted(unknown_queries)[0]!r}"
        )
    for query_id, relevance_by_doc in dataset.qrels.items():
        if not any(relevance > 0 for relevance in relevance_by_doc.values()):
            raise ValueError(f"SciFact query {query_id!r} has no positive qrel")
        unknown_docs = set(relevance_by_doc) - set(dataset.corpus)
        if unknown_docs:
            raise ValueError(
                "SciFact qrels reference unknown corpus document "
                f"{sorted(unknown_docs)[0]!r}"
            )


def dataset_to_documents(dataset: SciFactDataset) -> list[Document]:
    """Convert corpus records while preserving the official document key."""
    documents: list[Document] = []
    for doc_id, record in dataset.corpus.items():
        title = record.get("title", "").strip()
        text = record["text"].strip()
        content = f"{title}\n\n{text}" if title else text
        documents.append(
            Document(
                content=content,
                source=f"beir://scifact/{doc_id}",
                source_type=SourceType.TEXT,
                metadata={"beir_doc_id": doc_id, "title": title},
            )
        )
    return documents


def verify_archive(
    archive_path: str | Path,
    *,
    expected_md5: str = SCIFACT_ARCHIVE_MD5,
) -> str:
    """Return the archive MD5 or fail if it differs from the pinned checksum."""
    digest = hashlib.md5()  # noqa: S324 - BEIR publishes an MD5 identity
    with Path(archive_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_md5:
        raise ValueError(
            f"SciFact archive checksum mismatch: expected={expected_md5}, actual={actual}"
        )
    return actual


def safe_extract_zip(archive_path: str | Path, output_dir: str | Path) -> Path:
    """Extract an archive after rejecting absolute and parent-traversal names."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member: {member.filename!r}")
        archive.extractall(destination)
    dataset_root = destination / SCIFACT_DATASET_NAME
    if not dataset_root.is_dir():
        raise ValueError("SciFact archive does not contain the expected scifact directory")
    return dataset_root


def verify_extracted_files(
    archive_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, str]:
    """Match evaluation-bearing extracted files byte-for-byte to the archive."""
    root = Path(dataset_root)
    members = {
        "corpus.jsonl": "scifact/corpus.jsonl",
        "queries.jsonl": "scifact/queries.jsonl",
        "qrels/test.tsv": "scifact/qrels/test.tsv",
    }
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for relative, member in members.items():
            extracted = root / relative
            if not extracted.is_file():
                raise ValueError(f"SciFact extracted file is missing: {relative}")
            try:
                archived_bytes = archive.read(member)
            except KeyError as exc:
                raise ValueError(f"SciFact archive is missing required member: {member}") from exc
            archive_hash = hashlib.sha256(archived_bytes).hexdigest()
            extracted_hash = hashlib.sha256(extracted.read_bytes()).hexdigest()
            if extracted_hash != archive_hash:
                raise ValueError(
                    f"SciFact extracted file differs from verified archive: {relative}"
                )
            hashes[relative] = archive_hash
    return hashes


def download_archive(
    destination: str | Path,
    *,
    url: str = SCIFACT_ARCHIVE_URL,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Download atomically; checksum validation is deliberately a separate step."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with opener(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
