# InsightAgent

A minimal, hand-rolled **Research Agent** built directly on the OpenAI-compatible
Chat Completions API. Chapter 1 verifies the smallest thing that deserves to be
called an agent loop — LLM decides → runtime executes a tool → result is
written back into messages → LLM decides again.

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
>> 请读取 examples/result.txt,并告诉我准确率最高的方法。
```

The agent will call `read_file`, receive the file contents as a tool message,
ask the LLM again, and print the final answer. Type `exit` or press Ctrl-D to
quit.

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

## What Chapter 1 does NOT implement

Deliberately out of scope until later chapters:

- RAG, embeddings, vector databases
- Web search / browsing
- Vision / multimodal input
- Planner, intent router, multi-agent orchestration
- Long-term memory, context compaction
- MCP, hooks, permission system, workspace sandbox
- Path-traversal protection, file size limits, binary files
- Retry / timeout / fallback / checkpoint
- FastAPI or any server component
- LangGraph, LangChain, OpenAI Agents SDK
