"""Small deterministic smoke test for the complete local Vector RAG data path."""

from __future__ import annotations

from pathlib import Path

from insight_agent.ingestion import ingest_file
from insight_agent.indexing import QdrantVectorStore, TextChunker, index_documents
from insight_agent.retrieval import VectorRetriever


class HeliosEmbedder:
    """Tiny deterministic embedding space; no model or network is involved."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "Beta" in text or "最高" in text:
                vectors.append([1.0, 0.0])
            elif "Gamma" in text:
                vectors.append([0.6, 0.8])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_ingest_index_and_retrieve_helios_fact(tmp_path: Path) -> None:
    facts = {
        "alpha.txt": "Project Helios 的 Alpha 方法得分为 31.7。",
        "beta.txt": "Project Helios 的 Beta 方法得分为 48.9。",
        "gamma.txt": "Project Helios 的 Gamma 方法得分为 42.3。",
    }
    documents = []
    for name, content in facts.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        documents.extend(ingest_file(str(path)))

    embedder = HeliosEmbedder()
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="helios",
    )
    try:
        indexed = index_documents(
            documents,
            TextChunker(chunk_size=100, chunk_overlap=10),
            embedder,
            store,
        )
        results = VectorRetriever(embedder, store).retrieve(
            "Project Helios 中得分最高的方法是什么？",
            top_k=1,
        )

        assert indexed == 3
        assert len(results) == 1
        assert "Beta" in results[0].content
        assert "48.9" in results[0].content
        assert results[0].source.endswith("beta.txt")
    finally:
        store.close()
