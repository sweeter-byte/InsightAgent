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

Copy the template and fill in the three required variables:

```bash
cp .env.example .env
```

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

`LLMClient` reads these at construction time and raises a clear `RuntimeError`
if any is missing.

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
python -m insight_agent "读取 examples/result.txt,准确率最高的方法是什么?"
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
| Image | `ingest_file("x.png")` | 1 — VLM description as content; original path kept in `source` |

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

Try it (defaults to an offline `.txt` input):

```bash
python examples/try_ingestion.py              # ingests examples/result.txt
python examples/try_ingestion.py paper.pdf    # one Document per non-empty page
python examples/try_ingestion.py chart.png    # requires VISION_* to be configured
```

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
