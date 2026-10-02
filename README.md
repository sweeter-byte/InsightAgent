# InsightAgent

A minimal, hand-rolled **Research Agent** built directly on the OpenAI-compatible
Chat Completions API. Chapter 1 verifies the smallest thing that deserves to be
called an agent loop — LLM decides → runtime executes a tool → result is
written back into messages → LLM decides again. Chapter 2 wraps that loop in an
`InsightAgent` façade fronted by an **Intent Router** that picks the execution
path for every query.

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
- RAG, embeddings, vector databases, PDF parsing, vision
- Web search, Research Planner, Workflow orchestration, LangGraph
- Memory, MCP
- Router confidence scores, hybrid routing, intent-evaluation benchmarks

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
