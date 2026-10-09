"""Public API for InsightAgent's document indexing layer."""

from insight_agent.indexing.chunker import TextChunker
from insight_agent.indexing.embedder import Embedder, SentenceTransformerEmbedder
from insight_agent.indexing.models import Chunk, make_chunk_id, make_document_id
from insight_agent.indexing.pipeline import index_documents
from insight_agent.indexing.vector_store import (
    QdrantConfig,
    QdrantVectorStore,
    VectorStore,
)

__all__ = [
    "Chunk",
    "Embedder",
    "QdrantConfig",
    "QdrantVectorStore",
    "SentenceTransformerEmbedder",
    "TextChunker",
    "VectorStore",
    "index_documents",
    "make_chunk_id",
    "make_document_id",
]
