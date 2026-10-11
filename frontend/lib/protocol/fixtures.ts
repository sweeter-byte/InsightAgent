/**
 * Fixed examples mirrored from Chapter 18 acceptance/runtime tests.
 * They are protocol fixtures, not demo-domain state.
 */
export const queryRequestFixture = {
  query: "research: compare approaches",
} as const;

export const directQueryResponseFixture = {
  request_id: "request-direct-1",
  intent: "direct",
  status: "completed",
  answer: "direct answer",
  run_id: null,
} as const;

export const analyzeQueryResponseFixture = {
  request_id: "request-analyze-1",
  intent: "analyze",
  status: "completed",
  answer: "analyze answer",
  run_id: null,
} as const;

export const researchQueryResponseFixture = {
  request_id: "request-research-1",
  intent: "research",
  status: "queued",
  answer: null,
  run_id: "demo-run",
} as const;

export const materialImportResultFixture = {
  material_id: "material-1",
  document_ids: ["document-1"],
  status: "indexed",
  chunk_count: 2,
  deduplicated: false,
} as const;

export const queuedRunFixture = {
  run_id: "demo-run",
  thread_id: "demo-thread",
  status: "queued",
  created_at: "2026-01-18T09:00:00Z",
  updated_at: "2026-01-18T09:00:00Z",
  error: null,
  final_output: null,
} as const;

export const completedRunFixture = {
  ...queuedRunFixture,
  status: "completed",
  updated_at: "2026-01-18T09:03:12Z",
  final_output: "# Research Report\n\nOffline acceptance report.",
} as const;

export const runtimeEventFixture = {
  event_id: "3-0",
  run_id: "demo-run",
  type: "workflow.progress",
  created_at: "2026-01-18T09:00:05Z",
  data: { stage: "retrieval", task_id: "T1" },
} as const;

export const healthResponseFixture = { status: "ok" } as const;

export const readinessResponseFixture = {
  status: "ready",
  required: {
    redis: { status: "ready", detail: null },
    qdrant: { status: "ready", detail: null },
    checkpoint: { status: "ready", detail: null },
    config: { status: "ready", detail: null },
  },
  optional: {
    web: { status: "available", detail: null },
    vision: { status: "unconfigured", detail: null },
  },
} as const;

export const notReadyResponseFixture = {
  status: "not_ready",
  required: {
    redis: { status: "unavailable", detail: "redis unavailable" },
    qdrant: { status: "ready", detail: null },
    checkpoint: { status: "ready", detail: null },
    config: { status: "ready", detail: null },
  },
  optional: {
    web: { status: "unconfigured", detail: null },
    vision: { status: "unconfigured", detail: null },
  },
} as const;

/** Exact SSE frame shape produced by runtime/api.py::_encode_sse. */
export const sseTextFixture = [
  "id: 1-0\nevent: run.queued\ndata: {}\n\n",
  "id: 2-0\nevent: run.started\ndata: {\"resume\":false}\n\n",
  "id: 3-0\nevent: workflow.progress\ndata: {\"stage\":\"retrieval\",\"task_id\":\"T1\"}\n\n",
  "id: 4-0\nevent: run.completed\ndata: {\"final_output\":\"# Research Report\\n\\nOffline acceptance report.\"}\n\n",
].join("");

export const sseEventFixtures = [
  { id: "1-0", event: "run.queued", data: {} },
  { id: "2-0", event: "run.started", data: { resume: false } },
  {
    id: "3-0",
    event: "workflow.progress",
    data: { stage: "retrieval", task_id: "T1" },
  },
  {
    id: "4-0",
    event: "run.completed",
    data: { final_output: "# Research Report\n\nOffline acceptance report." },
  },
] as const;
