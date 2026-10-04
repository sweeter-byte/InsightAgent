"""Index one local file, then inspect Vector RAG retrieval results.

This is deliberately an end-to-end demo. Production query handling should use
an already-populated Qdrant collection and must not re-index on every question.
"""

from __future__ import annotations

import argparse

from insight_agent.ingestion import ingest_file
from insight_agent.indexing import (
    QdrantVectorStore,
    SentenceTransformerEmbedder,
    TextChunker,
    index_documents,
)
from insight_agent.retrieval import VectorRetriever, format_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to a local txt, md, or PDF file")
    parser.add_argument("query", help="Question to retrieve relevant chunks for")
    parser.add_argument(
        "--top-k",
        type=int,
        choices=range(1, 9),
        default=5,
        metavar="1..8",
        help="Number of chunks to retrieve (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = ingest_file(args.source)
    embedder = SentenceTransformerEmbedder()
    store = QdrantVectorStore()
    try:
        indexed = index_documents(documents, TextChunker(), embedder, store)
        print(f"Indexed {indexed} chunk(s).")

        retriever = VectorRetriever(embedder=embedder, vector_store=store)
        results = retriever.retrieve(args.query, top_k=args.top_k)
        print(format_results(results))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
