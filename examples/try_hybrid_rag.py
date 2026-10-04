"""Index local files and inspect every Hybrid Retrieval ranking stage.

This manual integration example may download the configured embedding and
cross-encoder models. Production query handling should reuse an existing
Qdrant collection instead of re-indexing for every query.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from insight_agent.ingestion import ingest_file
from insight_agent.indexing import (
    QdrantVectorStore,
    SentenceTransformerEmbedder,
    TextChunker,
    index_documents,
)
from insight_agent.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetrievalConfig,
    HybridRetriever,
    RetrievalTrace,
    VectorRetriever,
    format_results,
)


DEFAULT_QUERIES = [
    "DEEPSEEK_API_KEY 在哪里被使用？",
    "为什么文本切分时需要保留一部分上下文？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Local txt, md, or PDF files")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to run; repeat the option for multiple queries",
    )
    return parser.parse_args()


def _print_trace(trace: RetrievalTrace) -> None:
    print(f"\nQuery: {trace.query}")
    for name in ("dense", "sparse", "fused", "reranked", "final"):
        entries = getattr(trace, name)
        ranking = ", ".join(
            f"{entry.chunk_id}={entry.score:.6f}"
            if entry.score is not None
            else f"{entry.chunk_id}=None"
            for entry in entries
        )
        print(f"{name}: {ranking or '<empty>'}")


def main() -> int:
    load_dotenv()
    args = parse_args()
    config = HybridRetrievalConfig.from_env()
    documents = [
        document
        for source in args.sources
        for document in ingest_file(source)
    ]
    embedder = SentenceTransformerEmbedder()
    store = QdrantVectorStore()
    try:
        indexed = index_documents(documents, TextChunker(), embedder, store)
        print(f"Indexed {indexed} chunk(s).")

        traces: list[RetrievalTrace] = []
        retriever = HybridRetriever(
            VectorRetriever(embedder, store),
            BM25Retriever(store.load_chunks),
            CrossEncoderReranker(config.reranker_model),
            config=config,
            trace_callback=traces.append,
        )
        for query in args.queries or DEFAULT_QUERIES:
            results = retriever.retrieve(query)
            _print_trace(traces[-1])
            print(format_results(results))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
