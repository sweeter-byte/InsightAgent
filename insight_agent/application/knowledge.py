"""Safe material storage and explicit knowledge-preparation orchestration."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from insight_agent.indexing import (
    Embedder,
    TextChunker,
    VectorStore,
    index_documents,
    make_document_id,
)
from insight_agent.ingestion import Document


_DOCUMENT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".pdf"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


class MaterialError(RuntimeError):
    """Base class for public material-import failures."""


class MaterialValidationError(MaterialError):
    """The submitted upload cannot be accepted safely."""


class MaterialTooLargeError(MaterialValidationError):
    """The upload exceeds the configured byte limit."""


class ImageIngestionUnavailableError(MaterialValidationError):
    """Image upload was requested without complete Vision configuration."""


class MaterialProcessingError(MaterialError):
    """A stored material failed in one preparation stage."""

    def __init__(
        self,
        stage: str,
        *,
        material_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.material_id = material_id
        super().__init__(f"material processing failed during {stage}")


class AsyncUpload(Protocol):
    filename: str | None

    async def read(self, size: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class StoredMaterial:
    material_id: str
    path: Path
    deduplicated: bool


class MaterialStatus(str, Enum):
    INDEXED = "indexed"


class MaterialImportResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    material_id: str
    document_ids: list[str]
    status: MaterialStatus
    chunk_count: int
    deduplicated: bool


class Refreshable(Protocol):
    def refresh(self) -> None: ...


class MaterialStorage:
    """Stream uploads into one controlled, content-addressed directory."""

    def __init__(
        self,
        upload_dir: str | Path,
        *,
        max_bytes: int,
        image_ingestion_enabled: bool = False,
        chunk_bytes: int = 64 * 1024,
    ) -> None:
        if not isinstance(upload_dir, (str, Path)) or not str(upload_dir).strip():
            raise ValueError("upload_dir must not be empty")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(chunk_bytes, bool)
            or not isinstance(chunk_bytes, int)
            or chunk_bytes <= 0
        ):
            raise ValueError("chunk_bytes must be a positive integer")
        self.upload_dir = Path(upload_dir)
        self.max_bytes = max_bytes
        self.image_ingestion_enabled = image_ingestion_enabled
        self.chunk_bytes = chunk_bytes

    async def store(self, upload: AsyncUpload) -> StoredMaterial:
        suffix = self._validated_suffix(upload.filename)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=".upload-", suffix=".tmp", dir=self.upload_dir
        )
        temp_path = Path(raw_temp_path)
        digest = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    block = await upload.read(self.chunk_bytes)
                    if not block:
                        break
                    total += len(block)
                    if total > self.max_bytes:
                        raise MaterialTooLargeError(
                            f"upload exceeds the {self.max_bytes}-byte limit"
                        )
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if total == 0:
                raise MaterialValidationError("uploaded file must not be empty")

            material_id = digest.hexdigest()
            destination = self.upload_dir / f"{material_id}{suffix}"
            try:
                os.link(temp_path, destination)
            except FileExistsError:
                deduplicated = True
            else:
                deduplicated = False
            return StoredMaterial(material_id, destination, deduplicated)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _validated_suffix(self, filename: str | None) -> str:
        if not isinstance(filename, str) or not filename.strip():
            raise MaterialValidationError("uploaded file must have a filename")
        suffix = Path(filename).suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            if not self.image_ingestion_enabled:
                raise ImageIngestionUnavailableError(
                    "Vision configuration is unavailable for image ingestion"
                )
            return suffix
        if suffix not in _DOCUMENT_SUFFIXES:
            allowed = ", ".join(sorted(_DOCUMENT_SUFFIXES | _IMAGE_SUFFIXES))
            raise MaterialValidationError(
                f"unsupported material type; allowed extensions: {allowed}"
            )
        return suffix


class KnowledgeService:
    """Prepare one controlled local file for shared local retrieval."""

    def __init__(
        self,
        *,
        ingestor: Callable[[str], list[Document]],
        chunker: TextChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        refresher: Refreshable,
    ) -> None:
        self.ingestor = ingestor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.refresher = refresher
        self._import_lock = threading.Lock()

    def import_material(self, material: StoredMaterial) -> MaterialImportResult:
        with self._import_lock:
            try:
                documents = self.ingestor(str(material.path))
            except Exception as exc:
                # The broad boundary is deliberate: callers receive one safe
                # application error while logs retain ``__cause__``.
                raise MaterialProcessingError(
                    "ingestion", material_id=material.material_id
                ) from exc

            document_ids = list(dict.fromkeys(make_document_id(doc) for doc in documents))
            try:
                chunk_count = index_documents(
                    documents,
                    self.chunker,
                    self.embedder,
                    self.vector_store,
                )
            except Exception as exc:
                raise MaterialProcessingError(
                    "indexing", material_id=material.material_id
                ) from exc

            try:
                self.refresher.refresh()
            except Exception as exc:
                raise MaterialProcessingError(
                    "refresh", material_id=material.material_id
                ) from exc

            return MaterialImportResult(
                material_id=material.material_id,
                document_ids=document_ids,
                status=MaterialStatus.INDEXED,
                chunk_count=chunk_count,
                deduplicated=material.deduplicated,
            )
