# InsightAgent

A minimal, hand-rolled **Research Agent** built directly on the OpenAI-compatible
Chat Completions API. Chapter 1 verifies the smallest thing that deserves to be
called an agent loop — LLM decides → runtime executes a tool → result is
written back into messages → LLM decides again. Chapter 2 wraps that loop in an
`InsightAgent` façade fronted by an **Intent Router** that picks the execution
path for every query. Chapter 3 stands up an independent **Multimodal
Ingestion** layer that turns Text / Markdown / PDF / URL / Image inputs into a
unified `list[Document]`.

No LangChain, no LangGraph, no OpenAI Agents SDK, no other agent framework.

## Requirements

- Python **3.11+**
- An OpenAI-compatible chat-completions endpoint (OpenAI itself, or any
  gateway that speaks the same `/v1/chat/completions` protocol with tool
  calling, e.g. vLLM, DeepSeek, Qwen, Moonshot, …).

## Install

The project uses a dedicated conda environment named `insight-agent`:

```bash
conda activate insight-agent
pip install -e ".[dev]"
```

## Configure `.env`

Copy the template and fill in the required variables. In addition to the chat
endpoint, Hybrid Retrieval requires an explicit cross-encoder model:

```bash
cp .env.example .env
```

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

`LLMClient` reads the first three at construction time. The knowledge-search
builder reads `RERANKER_MODEL`; either component raises a clear `RuntimeError`
when its required configuration is missing.

## Run the CLI

```bash
python -m insight_agent
```

This opens an interactive REPL:

```text
>> 什么是 Agent Loop？
```

You never pick a mode by hand — the Intent Router classifies each query and
decides whether it is answered directly or handed to the ResearchAgent. For a
general-knowledge question the router chooses `direct` and the answer comes
from a single LLM call; for a file-reading request like the Chapter 1 example
below, it chooses `analyze`/`research` and the `ResearchAgent` calls
`read_file`, receives the file contents as a tool message, asks the LLM again,
and prints the final answer. Type `exit` or press Ctrl-D to quit.

A one-shot form is also supported:

```bash
python -m insight_agent "读取 README.md，并概括项目当前实现范围"
```

## Run the tests

```bash
pytest
```

The default suite uses a scripted `FakeLLMClient` and **never** contacts a
real endpoint, so it produces no API cost.

## Manual smoke test against a real endpoint

Not part of `pytest`. Once `.env` is configured, run the CLI with the example
query above and confirm the assistant calls `read_file` and returns the right
answer. This is the fastest way to sanity-check that your `LLM_BASE_URL` and
`LLM_MODEL` actually support tool calling.

## What Chapter 1 implements

- `LLMClient`: reads env config, wraps `openai.OpenAI`, exposes a single
  `chat(messages, tools=None)` that returns the raw SDK response.
- `read_file(path) -> str`: reads a UTF-8 text file, returns an error string
  instead of raising on missing / unreadable files.
- `READ_FILE_SCHEMA`: the OpenAI tool-calling schema for `read_file`.
- `ToolRegistry`: `name -> callable` dispatch, plus `build_default_registry()`
  and `default_tool_schemas()`.
- `ResearchAgent`: the actual agent loop, `max_steps` guard, message-history
  handling for tool calls / tool results, and safe error handling for unknown
  tools, malformed JSON arguments, and tool-side exceptions.
- `python -m insight_agent` CLI (REPL + one-shot).

## What Chapter 2 adds

Chapter 2 introduces an **Intent Router** and a single application façade,
`InsightAgent`, that the CLI now builds via `_build_app()` (composing
`LLMClient` + `ToolRegistry` + `ResearchAgent` + `IntentRouter` + `InsightAgent`
over one shared `LLMClient`). The control flow is:

```text
User
  ↓
Intent Router
  ├─ direct   → single LLM call   (no tools, no agent loop)
  ├─ analyze  → ResearchAgent
  └─ research → ResearchAgent
```

Details:

- `Intent`: a three-value enum (`direct` / `analyze` / `research`), priority
  `research > analyze > direct`.
- `IntentRouter`: one LLM call with a classification prompt, string→enum parse,
  and a safe fallback to `research` on empty/invalid output. No structured
  output, no keyword heuristics, no confidence score, no hybrid routing.
- `InsightAgent`: intent dispatch only. `direct` answers with a plain chat
  completion (never enters the agent loop, so it cannot raise
  `AgentStepsExceeded`); `analyze` and `research` delegate to `ResearchAgent`.
- `analyze` and `research` **currently share the same `ResearchAgent` on
  purpose** — this is the Chapter 2 design, not a bug. The intent label is kept
  so later chapters can fork them into a Document / Multimodal Analysis Pipeline
  and a Research Workflow respectively.
- The CLI (`python -m insight_agent`) keeps both modes unchanged; the router
  decides the path automatically and the intent is not printed to normal output.

## What Chapter 2 does NOT implement

Still deliberately out of scope until later chapters:

- Document / Multimodal Analysis Pipeline and Research Workflow as separate
  handlers (the intents exist but share one handler)
- PDF parsing and vision were added in **Chapter 3** (Multimodal Ingestion); see
  "What Chapter 3 adds" below. RAG, embeddings, and vector databases remain out
  of scope through Chapter 3.
- Web search, Research Planner, Workflow orchestration, LangGraph
- Memory, MCP
- Router confidence scores, hybrid routing, intent-evaluation benchmarks

## What Chapter 3 adds

Chapter 3 introduces an **independent Multimodal Ingestion layer**
(`insight_agent/ingestion/`). It is a data-processing layer, *not* part of the
agent's decision loop. Three concept layers stay cleanly separated:

```text
Intent Router        → decides which high-level path a request takes
Multimodal Ingestion → decides how material becomes a unified Document
Agent Loop / Tools   → decides which external capability to call next
```

Five input formats normalize into one shape — `list[Document]`:

| Input | Entry | Documents produced |
|---------|---------------------------------------|--------------------------------------------------|
| Text | `ingest("text", inline_text)` | 1 — inline text as content |
| Markdown | `ingest_file("x.md")` | 1 — raw markdown preserved |
| PDF | `ingest_file("x.pdf")` | 1 per non-empty page (`metadata.page`/`page_count`) |
| URL | `ingest("url", "https://…")` | 1 — cleaned visible text (`metadata.title`/`final_url`/`content_type`) |
| Image (PNG / JPG / JPEG / WEBP) | `ingest_file("x.png")` | 1 — VLM description as content; original path kept in `source` |

- `Document`: a plain dataclass with `content` / `source` / `source_type` /
  `metadata` (see `ingestion.models`).
- Public API — import from `insight_agent.ingestion`, never from
  `ingestion.loaders.*`: `ingest`, `ingest_file`, `ingest_text_file`,
  `infer_file_type`, `safe_ingest`, plus per-format `load_*` escape hatches.
- Local files are dispatched **by the program** from their extension
  (`infer_file_type`); the LLM never picks a loader and loaders are **not**
  registered as Agent tools.
- Single error type `IngestionError` (a `RuntimeError`); `safe_ingest` wraps any
  lower-level or third-party exception into it while preserving `__cause__`.

Vision config (images only) reuses the project's single env-var mechanism:
`VISION_API_KEY`, `VISION_BASE_URL`, `VISION_MODEL` (see `.env.example`). Text /
Markdown / PDF ingestion needs no API; URL ingestion needs network but no key.

Basic file ingestion:

```python
from insight_agent.ingestion import documents_to_context, ingest_file

documents = ingest_file("path/to/paper.pdf")

for document in documents:
    print("source_type:", document.source_type.value)
    print("source:", document.source)
    print("metadata:", document.metadata)
    print("content preview:", document.content[:200])

context = documents_to_context(documents)
print(context[:500])
```

Text, Markdown, and PDF files work without API configuration. Image ingestion,
for example `ingest_file("path/to/chart.png")`, requires `VISION_API_KEY`,
`VISION_BASE_URL`, and `VISION_MODEL`.

### Temporary context adapter (**not** RAG)

`documents_to_context(documents)` (in `ingestion.context`) renders a
`list[Document]` into a single `[Document N] / source / source_type / metadata /
<content>` text block, so the current LLM/Agent stack can consume ingested
material in small integration tests. It is **not RAG** and does nothing beyond
string formatting: no chunking, embedding, vector store, retrieval, or ranking;
no token budget and no context compression (content is dumped verbatim). Do not
push large Document sets into an Agent context through it.

## What Chapter 3 does NOT implement

Still deliberately out of scope until later chapters:

- Chunking / text splitting
- Embedding and any vector database
- Retrieval / a RAG pipeline, token budgets, or context compression
- OCR and scanned-PDF handling
- SSRF protection, JS rendering, or login/JS-heavy extraction in the URL loader
- Registering loaders as LLM tools, or letting the model choose a loader
- Wiring ingestion into the CLI / `InsightAgent` façade (integration is future work)

## What Chapter 1 does NOT implement

Deliberately out of scope until later chapters:

- RAG, embeddings, vector databases
- Web search / browsing
- Vision / multimodal input
- Planner, multi-agent orchestration
- Long-term memory, context compaction
- MCP, hooks, permission system, workspace sandbox
- Path-traversal protection, file size limits, binary files
- Retry / timeout / fallback / checkpoint
- FastAPI or any server component
- LangGraph, LangChain, OpenAI Agents SDK

## What Chapter 5 adds

Chapter 5 turns the persisted Chapter 4 index into a minimal local Vector RAG
capability while keeping indexing and querying separate:

```text
query → embed_documents([query]) → Qdrant top-k search
      → RetrievalResult → formatted observation → search_knowledge_base
      → existing ResearchAgent loop
```

- `insight_agent.retrieval.VectorRetriever` reuses the same configured
  `SentenceTransformerEmbedder` and `QdrantVectorStore` as indexing.
- `RetrievalResult` is a query-time model with `score`; the index-time `Chunk`
  model is unchanged.
- Qdrant searches request payloads but not stored vectors. Invalid/missing
  provenance payloads fail explicitly rather than producing partial results.
- `search_knowledge_base(query, top_k=5)` is registered in the original flat
  `ToolRegistry` (`1 <= top_k <= 8`) and returns source-aware observations.
- The Intent Router remains unaware of embeddings and retrieval. It still only
  selects `direct` / `analyze` / `research`; the ResearchAgent chooses the tool.

The normal query path searches the existing collection and never ingests or
indexes documents. For a standalone ingestion → indexing → retrieval demo:

```bash
conda run --no-capture-output -n insight-agent \
  python examples/try_vector_rag.py notes/rag.md "为什么分块需要 overlap？" --top-k 5
```

The first sentence-transformer use may download the configured model. Chapter 5
does not add sparse/hybrid retrieval, reranking, query rewriting, citations,
web fallback, LangChain, or LangGraph.

## What Chapter 6 adds

Chapter 6 upgrades the same `search_knowledge_base` Tool to Hybrid Retrieval:

```text
query
├── dense retrieval ─┐
└── BM25 retrieval ──┴→ RRF → candidate cutoff → cross-encoder → final top-k
```

- Qdrant remains the source of truth. At first Tool use, its Chunk payloads are
  loaded once to build an in-memory BM25 snapshot; queries reuse that snapshot.
- `HybridRetriever.refresh()` explicitly rebuilds BM25 from Qdrant after
  indexing. It builds the replacement first and swaps it under a short lock, so
  a failed refresh leaves the old snapshot usable.
- BM25 uses the same `jieba`-aware tokenizer for corpus and query text while
  preserving technical identifiers such as `DEEPSEEK_API_KEY` and file paths.
  Its positive `log(1 + RSJ)` IDF keeps unique terms useful in very small
  corpora; Chunks with no searchable tokens remain available to Dense retrieval.
- RRF uses ranks only and deduplicates by stable `chunk_id`; raw dense and BM25
  scores are never added together.
- Only the fused `HYBRID_RERANK_K` candidates reach the configured cross-encoder.
  `RERANKER_MODEL` is required and has no code fallback. A recommended bilingual
  choice is `BAAI/bge-reranker-v2-m3`.
- Stage depths are configured with `HYBRID_DENSE_K`, `HYBRID_SPARSE_K`,
  `HYBRID_RERANK_K`, `HYBRID_FINAL_TOP_K`, and `HYBRID_RRF_K`. Internal recall
  depths may exceed eight; only the final Agent-facing Tool argument remains
  bounded to `1 <= top_k <= 8`. An explicit Tool argument overrides
  `HYBRID_FINAL_TOP_K`.
- DEBUG logging and an optional trace callback expose dense, sparse, fused,
  reranked, and final rankings without adding internal scores to the Agent
  observation.

The Tool name/schema, Registry, Intent Router, and Agent Loop remain unchanged.
Query rewriting, citations, web fallback, automatic index-change detection,
LangChain, and LangGraph remain out of scope.

To manually inspect the dense, sparse, fused, reranked, and final rankings for
the exact-identifier and semantic example queries:

```bash
conda run --no-capture-output -n insight-agent \
  python examples/try_hybrid_rag.py notes/config.md notes/chunking.md
```

## What Chapter 9 adds

Chapter 9 turns the Chapter 8 `web_entry` destination into a minimal public-Web
retrieval path:

```text
Research Task -> Retrieval Router -> WebRetriever
              -> Tavily Search -> candidate URLs
              -> existing URL Loader -> Document -> ResearchState.web_results
```

Tavily is used only to discover candidate pages. The provider explicitly does
not request Tavily answers, images, or raw page content; selected URLs are
fetched and cleaned by the existing Chapter 3 URL Loader. Search context is
added under `search_query`, `search_rank`, `search_title`, `search_snippet`, and
optional `search_score` without replacing loader metadata such as `title`,
`final_url`, or `content_type`.

Web Search is optional. Configure it in `.env` to let the runtime advertise
`web` to the Retrieval Router:

```text
TAVILY_API_KEY=tvly-...
WEB_SEARCH_TIMEOUT=30
WEB_SEARCH_LIMIT=5
WEB_FETCH_LIMIT=3
```

Without `TAVILY_API_KEY`, local routing remains available but WEB is omitted
from `available_sources`. Search-provider failure is an
explicit error; failure to fetch one candidate page is retained in
`WebRetrievalResult.failures` while later candidates continue. Results are raw
retrieval material, not Evidence, Citations, confidence, or a final report.

Chapter 9 does not add query rewriting, multi-query or iterative search,
Tavily answer/extract/crawl/research APIs, browser automation, source authority
scoring, Evidence/Citation generation, report generation, or Vision Retrieval.

## What Chapter 10 adds

Chapter 10 turns the existing `vision_entry` into task-conditioned visual
retrieval:

```text
Research Task -> Hybrid Retrieval (source_type=image)
              -> deduplicate original image paths
              -> reload accessible originals
              -> task-conditioned VLM analysis
              -> ResearchState.vision_results[task_id]
```

The existing image-ingestion path is unchanged: `describe_image(path)` still
creates the general-purpose description stored in the text index. Vision
Retrieval searches those descriptions, restores `RetrievalResult.source`, and
uses a separate prompt containing the plan objective, current task, and user
constraints. Image text is treated as untrusted data and never as instructions.

`HybridRetriever.retrieve()` now accepts an optional keyword-only
`source_types` set. Dense retrieval converts it to a Qdrant payload filter, and
BM25 restricts eligible snapshot chunks before ranking and top-k truncation.
Omitting the argument preserves the original local-retrieval behavior.

Vision is advertised to the Retrieval Router only when all existing
`VISION_API_KEY`, `VISION_BASE_URL`, and `VISION_MODEL` settings are present and
a one-time startup probe finds an indexed image with a readable supported
original file. The local knowledge tool and Vision Retriever then share the
same Hybrid Retriever. No per-route knowledge-base scan is performed.

Each task stores a `VisionRetrievalResult` with successful analyses, structured
per-image failures, and a separate no-candidate state. One missing image or VLM
failure does not discard successful analyses from other candidates. These
results are retrieval material, not Evidence or Citations.

A small end-to-end fixture is available at
[`examples/vision_workflow.png`](examples/vision_workflow.png) (with the
editable SVG beside it). A representative task is: “Determine whether the
Local, Web, and Vision paths all converge on the same Advance Task node.”

Chapter 10 does not add CLIP, a second vector store, OCR, crops, bounding boxes,
a multi-turn Vision Agent, Evidence grading, citations, or report generation.

## What Chapter 16 adds

Chapter 16 wraps the existing checkpointed Research Workflow in a separate,
single-process runtime:

```text
POST /v1/research/runs -> Redis Run Registry -> queued
                       -> asyncio Semaphore -> running
                       -> existing ResearchCoordinator / SQLite Checkpoint
                       -> Redis Stream events -> SSE
```

The runtime keeps `request_id`, `run_id`, and LangGraph `thread_id` separate.
Redis stores only runtime records and client-facing events; plans, evidence,
reports, and the canonical `ResearchState` remain in the Chapter 15 SQLite
checkpointer. A timed-out or interrupted run can be resumed explicitly with its
original `run_id -> thread_id -> checkpoint` chain. Startup marks stale
`running` records as `interrupted` but never resumes them automatically.

Start a local Redis, configure the existing model/retrieval settings, then run:

```bash
conda run --no-capture-output -n insight-agent \
  uvicorn insight_agent.runtime.app:app
```

Create and observe a run:

```bash
curl -X POST http://127.0.0.1:8000/v1/research/runs \
  -H 'Content-Type: application/json' \
  -d '{"query":"Compare the main Agent Memory designs"}'

curl http://127.0.0.1:8000/v1/research/runs/<run_id>
curl -N http://127.0.0.1:8000/v1/research/runs/<run_id>/events
curl -X POST http://127.0.0.1:8000/v1/research/runs/<run_id>/resume
```

Runtime policy is configured with `RUNTIME_REDIS_URL`,
`RUNTIME_MAX_CONCURRENCY`, `RUNTIME_RUN_TIMEOUT_SECONDS`,
`RUNTIME_EVENT_TTL_SECONDS`, `RUNTIME_INFRA_RETRY_ATTEMPTS`, and
`RUNTIME_INFRA_RETRY_BACKOFF_SECONDS`. This chapter intentionally does not add
authentication, distributed workers, automatic resume, WebSockets, or
exactly-once execution.
