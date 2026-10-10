# InsightAgent

InsightAgent is a backend research agent with one HTTP query entry point,
controlled local-material ingestion, hybrid retrieval, and asynchronous research
runs. The research workflow uses **LangGraph** with SQLite checkpoints; Redis
stores public run status and SSE events. A separate top-level `evals` package
provides deterministic offline Evaluation, baseline registration, and regression
comparison.

The current delivery is backend-only. It supports three intent paths:

- `direct`: one normal answer, completed in the request; no Research Run.
- `analyze`: synchronous analysis using tools and the shared local knowledge
  index; no Research Run.
- `research`: immediate `202 Accepted` with a `run_id`, followed by status/SSE
  observation and a final Research Report.

## Architecture

```text
POST /v1/materials -> controlled storage -> ingestion -> chunk/embed -> Qdrant
                                                       -> refresh hybrid retrieval

POST /v1/query -> IntentRouter
                 |- direct  -> one-shot LLM answer
                 |- analyze -> ResearchAgent + local tools
                 `- research -> ResearchRuntimeService -> LangGraph workflow
                                 |- Redis status + SSE events
                                 `- SQLite checkpoint + resume

python -m evals -> versioned Dataset + recorded Fixture -> Evaluation Result
                  -> explicit Baseline -> Regression report
```

The composition root is `insight_agent.application.build_application`. The
FastAPI lifespan in `insight_agent.runtime.app` creates shared resources once,
reuses them for every request, and closes them in reverse ownership order.

## Install

Python 3.11 or newer is required. This repository uses Conda; do not create a
project `.venv`.

```bash
conda env list
conda create -n insight-agent python=3.11
conda run --no-capture-output -n insight-agent python -m pip install -e ".[dev]"
```

If the named environment already exists, keep it and update packages only when
needed.

## Configure

```bash
cp .env.example .env
```

At minimum, set:

```dotenv
LLM_API_KEY=replace-me
LLM_BASE_URL=https://your-openai-compatible-host/v1
LLM_MODEL=your-tool-capable-model
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

`LLM_BASE_URL` must expose an OpenAI-compatible Chat Completions API. The model
must support the tool-calling behavior used by Analyze and the structured
decisions used by Research. `VISION_*` and `TAVILY_API_KEY` are optional.

Embedding and reranker models may download on first use. Pre-provision their
caches for offline deployment. Never commit `.env`.

## Start Redis and InsightAgent

Use two terminals:

```bash
redis-server --port 6379
```

```bash
conda run --no-capture-output -n insight-agent \
  python -m uvicorn insight_agent.runtime.app:app \
  --host 127.0.0.1 --port 8000
```

FastAPI lifespan opens Redis, local Qdrant, model clients, hybrid retrieval, and
the SQLite checkpoint store once. Startup fails when a required resource cannot
be initialized.

## Health and readiness

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

`GET /health` is process liveness. `GET /ready` checks the already-owned Redis,
Qdrant, checkpoint, and configuration resources. Web and Vision are optional;
a required component failure returns HTTP 503.

## Import demo material

The endpoint accepts `.md`, `.markdown`, `.txt`, and `.pdf`. Image types also
require complete `VISION_*` configuration. Uploads are streamed into
`MATERIAL_UPLOAD_DIR`, bounded by `MATERIAL_MAX_BYTES`, content-addressed, and
indexed before the response is returned.

```bash
curl --fail \
  -F 'file=@examples/chapter18_demo_material.md;type=text/markdown' \
  http://127.0.0.1:8000/v1/materials
```

```json
{
  "material_id": "<sha256>",
  "document_ids": ["<document-id>"],
  "status": "indexed",
  "chunk_count": 1,
  "deduplicated": false
}
```

New content returns HTTP 201. Repeated content returns HTTP 200 with
`deduplicated: true`. The API never accepts a server-local path or URL.

## Unified Query

### Direct

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"Direct answer only: what is 2 + 2?"}' \
  http://127.0.0.1:8000/v1/query
```

Direct and Analyze return HTTP 200 with the same stable fields:

```json
{
  "request_id": "<request-id>",
  "intent": "direct",
  "status": "completed",
  "answer": "4",
  "run_id": null
}
```

### Analyze imported knowledge

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"Analyze the imported Chapter 18 demo notes and report the exact acceptance marker."}' \
  http://127.0.0.1:8000/v1/query
```

The expected answer is grounded in the shared index and includes
`IA-CH18-READY`. The current Query contract has no material-level filter.

### Research through the unified entry point

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"Research and compare three approaches to agent memory, including trade-offs and sources."}' \
  http://127.0.0.1:8000/v1/query
```

Research immediately returns HTTP 202:

```json
{
  "request_id": "<request-id>",
  "intent": "research",
  "status": "queued",
  "answer": null,
  "run_id": "<run-id>"
}
```

Intent classification is model-driven. Use the explicit Runtime endpoint below
when an integration must deterministically request Research.

## Research Run, status, and SSE

Create a run:

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"Compare direct answering, document analysis, and multi-step research."}' \
  http://127.0.0.1:8000/v1/research/runs
```

The HTTP 202 response contains `run_id`, `thread_id`, `status`, timestamps,
`error`, and `final_output`. Inspect status and the final report:

```bash
curl --fail http://127.0.0.1:8000/v1/research/runs/<run_id>
```

Statuses are `queued`, `running`, `completed`, `failed`, `timed_out`, and
`interrupted`. `final_output` contains the Research Report after completion.

Stream progress until a terminal event:

```bash
curl -N --fail \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/events
```

Reconnect exclusively after a previous SSE id:

```bash
curl -N --fail -H 'Last-Event-ID: <event-id>' \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/events
```

Events include `run.queued`, `run.started`, `workflow.progress`,
`run.completed`, `run.failed`, `run.timed_out`, and `run.interrupted`.

## Resume interrupted work

Resume is valid only for `interrupted` or `timed_out` runs. It preserves the
original `run_id` and LangGraph `thread_id` and continues from the SQLite
checkpoint without submitting the initial query again.

```bash
curl --fail -X POST \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/resume
```

To reproduce a restart safely, submit a long Research Run, stop the service
while it is `running`, and start it with the same Redis database and
`CHECKPOINT_PATH`. Startup reconciles a leftover `running` record to
`interrupted`; call Resume afterward. Other states return HTTP 409 rather than
starting duplicate work.

## Fixed five-scenario demo

After the real service is ready:

```bash
conda run --no-capture-output -n insight-agent \
  bash examples/chapter18_demo.sh
```

The script covers Health/Readiness, material import, Direct, Analyze, explicit
Research, status, SSE, and offline Evaluation. Resume is opt-in because it needs
a real interrupted or timed-out run:

```bash
INTERRUPTED_RUN_ID=<run-id> \
conda run --no-capture-output -n insight-agent \
  bash examples/chapter18_demo.sh
```

Provider-backed steps are a manual Smoke Test. An offline pass does not prove
that a configured external model is reachable.

## Offline Evaluation

The supported entry point is the top-level `evals` package. This recorded
fixture needs no Redis, FastAPI, or external API:

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --dataset eval_data/research_eval_smoke_v1.jsonl \
  --fixture evals/fixtures/smoke_v1.json \
  --mode offline \
  --output /tmp/insight-agent-eval/runs \
  --run-id chapter18-candidate
```

The CLI labels the result `MOCK TEST`; it does not measure real-model quality.
Baseline registration is explicit:

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --register-baseline /tmp/insight-agent-eval/runs/chapter18-candidate \
  --baseline-output /tmp/insight-agent-eval/baseline.json \
  --baseline-kind mock_only \
  --confirm-baseline
```

Compare the candidate:

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --baseline /tmp/insight-agent-eval/baseline.json \
  --candidate /tmp/insight-agent-eval/runs/chapter18-candidate \
  --policy evals/policies/default_v1.json \
  --report-output /tmp/insight-agent-eval/reports
```

Comparison exits with 0 pass, 1 regression, 2 incompatible, or 3 inconclusive.

## Tests

```bash
conda run --no-capture-output -n insight-agent python -m pytest -q
```

Tests use fake LLM/provider components, in-memory runtime stores, temporary
files, and temporary SQLite checkpoints. External provider Smoke Tests remain
manual and are not reported as successful when credentials or models are
unavailable.

## Final HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | HTTP liveness |
| `GET` | `/ready` | Required and optional dependency readiness |
| `POST` | `/v1/materials` | Multipart material import and indexing |
| `POST` | `/v1/query` | Unified Direct / Analyze / Research entry |
| `POST` | `/v1/research/runs` | Explicit asynchronous Research creation |
| `GET` | `/v1/research/runs/{run_id}` | Run status and final report |
| `GET` | `/v1/research/runs/{run_id}/events` | SSE progress/replay |
| `POST` | `/v1/research/runs/{run_id}/resume` | Resume interrupted/timed-out work |

Every HTTP response includes `X-Request-ID`.

## Engineering limits

- The Research Runtime is single-process; Redis persists public state, but this
  release has no distributed worker queue.
- SQLite checkpoint storage is local. Multiple replicas need a shared
  checkpoint strategy before horizontal scaling.
- The knowledge index is shared; `/v1/query` has no per-material filter.
- Material preparation is synchronous to the client, although blocking
  ingestion/indexing is moved off the event loop.
- Authentication, authorization, rate limiting, and production TLS termination
  are deployment responsibilities outside this repository.
