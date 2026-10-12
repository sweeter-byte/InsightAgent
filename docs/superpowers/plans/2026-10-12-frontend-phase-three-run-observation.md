# Frontend Phase Three Run Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add authoritative Research Run reads, standards-compliant SSE observation, bounded reconnect/reconciliation, same-tab remount recovery, and truthful live Run presentation to the existing Next.js workbench.

**Architecture:** Keep transport independent from React: one incremental SSE parser and fetch-stream client feed a pure reducer through an abortable observation coordinator. A small hook owns one observation generation, while `InsightWorkbench` owns only the selected Run and sessionStorage persistence; presentation components consume typed state and never open connections.

**Tech Stack:** Next.js 16, React 19, TypeScript 5.9, Zod 4, native Fetch/ReadableStream/TextDecoder, Vitest 5, Testing Library, MSW 3, Node HTTP for controlled proxy-stream verification.

---

## File Structure

Create these focused files:

- `frontend/lib/research-events/sse-parser.ts` — incremental SSE line/frame parser only.
- `frontend/lib/research-events/client.ts` — one validated `fetch` stream connection and typed SSE transport errors.
- `frontend/components/workbench/run-state.ts` — Run/connection state types, terminal helpers, and pure reducer.
- `frontend/components/workbench/run-observer.ts` — GET/SSE/reconcile/retry orchestration independent of React.
- `frontend/components/workbench/use-research-run.ts` — one-effect React lifecycle wrapper.
- `frontend/components/workbench/run-selection.ts` — safe versioned sessionStorage read/write/clear helpers.
- `frontend/tests/sse-parser.test.ts` — chunk, CRLF, multiline data, comments, and incomplete-frame coverage.
- `frontend/tests/research-events-client.test.ts` — request headers, validation, HTTP/content-type/network/abort coverage.
- `frontend/tests/run-reducer.test.ts` — status transitions, terminal semantics, and deduplication.
- `frontend/tests/run-observer.test.ts` — Last-Event-ID, reconciliation, retry, cursor fallback, terminal stop, and stale cancellation.
- `frontend/tests/run-selection.test.ts` — persistence and unavailable-storage behavior.
- `frontend/tests/run-observation.test.tsx` — hook/workbench submission, remount, switch, abort, and honest UI behavior.
- `frontend/scripts/verify-sse-proxy.mjs` — controlled non-MSW Next.js rewrite streaming verification.

Modify these existing files:

- `frontend/lib/api-client.ts` — add validated Run GET.
- `frontend/lib/protocol/fixtures.ts` — add status/event fixtures used across tests.
- `frontend/components/workbench/workbench-types.ts` — replace submission-only Run facts with active selection types.
- `frontend/components/workbench/insight-workbench.tsx` — select/persist/observe one Run and seed explicit Run records.
- `frontend/components/workbench/research-workspace.tsx` — render live execution/progress/final-output facts.
- `frontend/components/workbench/run-inspector.tsx` — render live IDs, status, connection state, errors, and event timeline.
- `frontend/app/globals.css` — minimal styles for live progress, connection label, and plain-text report preview.
- `frontend/tests/api-client.test.ts` — Run GET contract and failures.
- `frontend/tests/query-submission.test.tsx` — smart Research metadata hydration and explicit Run observation expectations.
- `frontend/tests/run-state.test.tsx` — preserve explicit Demo behavior while adding live terminal presentations.
- `frontend/tests/mocks/handlers.ts` — default Run GET and finite SSE handlers.
- `frontend/package.json` — add the controlled proxy verification command.
- `frontend/README.md` — document the SSE proxy check and current Run persistence boundary.

No Python backend file is modified.

### Task 1: Add the Authoritative Run GET API

**Files:**
- Modify: `frontend/tests/api-client.test.ts`
- Modify: `frontend/lib/api-client.ts`
- Modify: `frontend/lib/protocol/fixtures.ts`

- [ ] **Step 1: Write failing Run GET tests**

Add tests that require the exact same-origin path, Zod validation, 404 typing, and Abort propagation:

```ts
it("gets and validates a known Research Run", async () => {
  let requestedUrl = "";
  mockServer.use(
    http.get(`${appBase}/v1/research/runs/demo-run`, ({ request }) => {
      requestedUrl = request.url;
      return HttpResponse.json(queuedRunFixture);
    }),
  );

  await expect(api.getResearchRun("demo-run")).resolves.toEqual(
    queuedRunFixture,
  );
  expect(requestedUrl).toBe(`${appBase}/v1/research/runs/demo-run`);
});

it("distinguishes missing and malformed Run records", async () => {
  mockServer.use(
    http.get(`${appBase}/v1/research/runs/missing`, () =>
      HttpResponse.json({ detail: "unknown run: missing" }, { status: 404 }),
    ),
    http.get(`${appBase}/v1/research/runs/broken`, () =>
      HttpResponse.json({ run_id: "broken" }),
    ),
  );

  await expect(api.getResearchRun("missing")).rejects.toMatchObject({
    name: "ApiHttpError",
    status: 404,
  });
  await expect(api.getResearchRun("broken")).rejects.toBeInstanceOf(
    ApiProtocolError,
  );
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `cd frontend && npm test -- tests/api-client.test.ts`

Expected: FAIL because `api.getResearchRun` does not exist.

- [ ] **Step 3: Implement the minimal API helper**

Add URL encoding at the route boundary and reuse the existing request pipeline:

```ts
researchRun: (runId: string) =>
  `/v1/research/runs/${encodeURIComponent(runId)}`,

getResearchRun(runId: string, signal?: AbortSignal): Promise<RunRecord> {
  return requestJson({
    path: apiRoutes.researchRun(runId),
    schema: runRecordSchema,
    init: { signal, cache: "no-store" },
  });
},
```

Extend fixtures with immutable `running`, `failed`, `timed_out`, and `interrupted` RunRecords by spreading `queuedRunFixture` and changing only real backend fields.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `cd frontend && npm test -- tests/api-client.test.ts`

Expected: all API-client tests PASS.

- [ ] **Step 5: Commit the Run GET slice**

```bash
git add frontend/lib/api-client.ts frontend/lib/protocol/fixtures.ts frontend/tests/api-client.test.ts
git commit -m "feat(frontend): fetch authoritative research runs"
```

### Task 2: Build a Standards-Aware Incremental SSE Parser

**Files:**
- Create: `frontend/tests/sse-parser.test.ts`
- Create: `frontend/lib/research-events/sse-parser.ts`

- [ ] **Step 1: Write failing chunk-boundary and line-semantics tests**

Define the intended parser API and cover arbitrary chunk boundaries, CRLF, comments, blank lines, and multiline data:

```ts
import { describe, expect, it, vi } from "vitest";
import { createSseParser } from "@/lib/research-events/sse-parser";

describe("incremental SSE parser", () => {
  it("waits across chunks before dispatching a complete frame", () => {
    const onEvent = vi.fn();
    const parser = createSseParser(onEvent);
    parser.feed("id: 1-0\nevent: run.st");
    parser.feed("arted\ndata: {\"resume\":false}");
    expect(onEvent).not.toHaveBeenCalled();
    parser.feed("\n\n");
    expect(onEvent).toHaveBeenCalledWith({
      id: "1-0",
      event: "run.started",
      data: '{"resume":false}',
    });
  });

  it("handles CRLF, comments, empty lines, and joins data lines", () => {
    const events: unknown[] = [];
    const parser = createSseParser((event) => events.push(event));
    parser.feed(
      ": keepalive\r\nid: 2-0\r\nevent: workflow.progress\r\n" +
        "data: {\"stage\":\"planning\",\r\ndata: \"message\":\"real\"}\r\n\r\n",
    );
    expect(events).toEqual([{
      id: "2-0",
      event: "workflow.progress",
      data: '{"stage":"planning",\n"message":"real"}',
    }]);
  });

  it("does not dispatch an unterminated final frame", () => {
    const onEvent = vi.fn();
    const parser = createSseParser(onEvent);
    parser.feed("id: 3-0\nevent: run.completed\ndata: {}");
    parser.finish();
    expect(onEvent).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run the parser test and verify RED**

Run: `cd frontend && npm test -- tests/sse-parser.test.ts`

Expected: FAIL because the parser module does not exist.

- [ ] **Step 3: Implement the minimal stateful line parser**

Use a buffer and per-frame fields; do not split the whole response on `"\n\n"`:

```ts
export type ParsedSseFrame = {
  id: string;
  event: string;
  data: string;
};

export function createSseParser(
  onEvent: (event: ParsedSseFrame) => void,
) {
  let buffer = "";
  let id = "";
  let event = "";
  let data: string[] = [];

  const dispatch = () => {
    if (data.length > 0) {
      onEvent({ id, event: event || "message", data: data.join("\n") });
    }
    id = "";
    event = "";
    data = [];
  };

  const consumeLine = (rawLine: string) => {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") return dispatch();
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "id" && !value.includes("\0")) id = value;
    if (field === "event") event = value;
    if (field === "data") data.push(value);
  };

  return {
    feed(chunk: string) {
      buffer += chunk;
      let newline = buffer.indexOf("\n");
      while (newline !== -1) {
        consumeLine(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf("\n");
      }
    },
    finish() {
      buffer = "";
    },
  };
}
```

- [ ] **Step 4: Run parser tests and verify GREEN**

Run: `cd frontend && npm test -- tests/sse-parser.test.ts`

Expected: all parser tests PASS.

- [ ] **Step 5: Commit the parser slice**

```bash
git add frontend/lib/research-events/sse-parser.ts frontend/tests/sse-parser.test.ts
git commit -m "feat(frontend): parse incremental SSE frames"
```

### Task 3: Add the Validated Fetch-Stream SSE Client

**Files:**
- Create: `frontend/tests/research-events-client.test.ts`
- Create: `frontend/lib/research-events/client.ts`

- [ ] **Step 1: Write failing stream-client tests**

Use injected `fetchImpl` and a real `ReadableStream` so these tests exercise decoding without MSW buffering:

```ts
const responseFromChunks = (chunks: string[], contentType = "text/event-stream") =>
  new Response(new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
      controller.close();
    },
  }), { headers: { "Content-Type": contentType } });

it("validates split events and sends Last-Event-ID", async () => {
  const events: SseEvent[] = [];
  const fetchImpl = vi.fn().mockResolvedValue(responseFromChunks([
    "id: 8-0\nevent: workflow.progress\ndata: {\"stage\":",
    "\"planning\"}\n\n",
  ]));

  await streamResearchEvents({
    runId: "run one",
    lastEventId: "7-0",
    signal: new AbortController().signal,
    onEvent: (event) => events.push(event),
    fetchImpl,
  });

  expect(fetchImpl).toHaveBeenCalledWith(
    "http://localhost:3000/v1/research/runs/run%20one/events",
    expect.objectContaining({
      headers: expect.objectContaining({
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
        "Last-Event-ID": "7-0",
      }),
    }),
  );
  expect(events).toEqual([{
    id: "8-0",
    event: "workflow.progress",
    data: { stage: "planning" },
  }]);
});

it.each([
  [new Response("missing", { status: 404 }), "SseHttpError"],
  [responseFromChunks(["{}"], "application/json"), "SseProtocolError"],
  [responseFromChunks(["id: 1\nevent: run.started\ndata: []\n\n"]), "SseProtocolError"],
])("classifies invalid stream responses", async (response, name) => {
  await expect(streamResearchEvents({
    runId: "run",
    signal: new AbortController().signal,
    onEvent: vi.fn(),
    fetchImpl: vi.fn().mockResolvedValue(response),
  })).rejects.toMatchObject({ name });
});
```

Also assert fetch rejection becomes `SseNetworkError`, while an already-aborted signal preserves `AbortError`.

- [ ] **Step 2: Run the client test and verify RED**

Run: `cd frontend && npm test -- tests/research-events-client.test.ts`

Expected: FAIL because the stream client module does not exist.

- [ ] **Step 3: Implement the minimal stream client**

Create typed errors and validate only the SSE envelope, not `RuntimeEvent`:

```ts
export class SseHttpError extends Error {
  readonly name = "SseHttpError";
  constructor(readonly status: number) {
    super(`SSE request failed with HTTP ${status}`);
  }
}

export class SseNetworkError extends Error {
  readonly name = "SseNetworkError";
  constructor() { super("Research event stream disconnected"); }
}

export class SseProtocolError extends Error {
  readonly name = "SseProtocolError";
  constructor() { super("Research event stream returned invalid data"); }
}

type StreamOptions = {
  runId: string;
  lastEventId?: string;
  signal: AbortSignal;
  onEvent: (event: SseEvent) => void;
  fetchImpl?: typeof fetch;
};

function browserUrl(path: string): string {
  return typeof window === "undefined"
    ? path
    : new URL(path, window.location.origin).toString();
}

export async function streamResearchEvents(options: StreamOptions): Promise<void> {
  const headers: Record<string, string> = {
    Accept: "text/event-stream",
    "Cache-Control": "no-cache",
  };
  if (options.lastEventId) headers["Last-Event-ID"] = options.lastEventId;
  let response: Response;
  try {
    response = await (options.fetchImpl ?? fetch)(
      browserUrl(apiRoutes.researchEvents(options.runId)),
      { headers, cache: "no-store", signal: options.signal },
    );
  } catch (error) {
    if (options.signal.aborted || (error instanceof Error && error.name === "AbortError")) throw error;
    throw new SseNetworkError();
  }
  if (!response.ok) throw new SseHttpError(response.status);
  if (!response.headers.get("content-type")?.toLowerCase().startsWith("text/event-stream")) {
    throw new SseProtocolError();
  }
  if (!response.body) throw new SseProtocolError();

  const decoder = new TextDecoder();
  const parser = createSseParser((frame) => {
    try {
      options.onEvent(sseEventSchema.parse({
        id: frame.id,
        event: frame.event,
        data: JSON.parse(frame.data),
      }));
    } catch {
      throw new SseProtocolError();
    }
  });
  const reader = response.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value, { stream: true }));
  }
  parser.feed(decoder.decode());
  parser.finish();
}
```

Use a local `browserUrl` helper equivalent to the API client's same-origin behavior and ensure both route builders encode Run IDs exactly once.

- [ ] **Step 4: Run parser and client tests and verify GREEN**

Run: `cd frontend && npm test -- tests/sse-parser.test.ts tests/research-events-client.test.ts`

Expected: both suites PASS with no unhandled stream errors.

- [ ] **Step 5: Commit the transport slice**

```bash
git add frontend/lib/research-events/client.ts frontend/tests/research-events-client.test.ts frontend/lib/api-client.ts
git commit -m "feat(frontend): stream validated research events"
```

### Task 4: Implement the Pure Run Reducer

**Files:**
- Create: `frontend/tests/run-reducer.test.ts`
- Create: `frontend/components/workbench/run-state.ts`

- [ ] **Step 1: Write failing reducer tests**

Exercise real backend states and keep progress separate from terminal execution:

```ts
const startedEvent = {
  id: "2-0",
  event: "run.started",
  data: { resume: false },
} as const;
const completedEvent = {
  id: "4-0",
  event: "run.completed",
  data: { final_output: "candidate" },
} as const;

it("moves queued to running to completed from real events", () => {
  let state = selectRun(initialRunState, "demo-run");
  state = runReducer(state, { type: "record", runId: "demo-run", record: queuedRunFixture });
  state = runReducer(state, { type: "event", runId: "demo-run", event: startedEvent });
  expect(state.observedStatus).toBe("running");
  state = runReducer(state, { type: "event", runId: "demo-run", event: completedEvent });
  expect(state.observedStatus).toBe("completed");
});

it("does not treat workflow.progress completed as a terminal Run", () => {
  const state = runReducer(selectRun(initialRunState, "demo-run"), {
    type: "event",
    runId: "demo-run",
    event: { id: "3-0", event: "workflow.progress", data: { stage: "completed" } },
  });
  expect(state.observedStatus).toBeNull();
  expect(state.latestProgress).toEqual({ stage: "completed" });
  expect(isTerminalRunStatus(state.observedStatus)).toBe(false);
});

it.each([
  ["run.failed", "failed"],
  ["run.timed_out", "timed_out"],
  ["run.interrupted", "interrupted"],
])("maps %s without inventing data", (event, status) => {
  const state = runReducer(selectRun(initialRunState, "demo-run"), {
    type: "event",
    runId: "demo-run",
    event: { id: "9-0", event, data: {} },
  });
  expect(state.observedStatus).toBe(status);
});

it("deduplicates IDs and ignores events for an old Run", () => {
  const selected = selectRun(initialRunState, "new-run");
  const old = runReducer(selected, { type: "event", runId: "old-run", event: startedEvent });
  const once = runReducer(old, { type: "event", runId: "new-run", event: startedEvent });
  const twice = runReducer(once, { type: "event", runId: "new-run", event: startedEvent });
  expect(twice.events).toHaveLength(1);
  expect(twice.lastEventId).toBe(startedEvent.id);
});
```

- [ ] **Step 2: Run the reducer test and verify RED**

Run: `cd frontend && npm test -- tests/run-reducer.test.ts`

Expected: FAIL because `run-state.ts` does not exist.

- [ ] **Step 3: Implement state, terminal helpers, and reducer**

Define `RunConnectionStatus`, `ObservedRunEvent`, and `ResearchRunState`. `select` creates a fresh state; every other action first checks `state.runId === action.runId`. The event branch must clone the ID set and map only known lifecycle events:

```ts
const EVENT_STATUS: Partial<Record<string, RunStatus>> = {
  "run.queued": "queued",
  "run.started": "running",
  "run.completed": "completed",
  "run.failed": "failed",
  "run.timed_out": "timed_out",
  "run.interrupted": "interrupted",
};

export const terminalRunStatuses = new Set<RunStatus>([
  "completed", "failed", "timed_out", "interrupted",
]);

export function isTerminalRunStatus(status: RunStatus | null | undefined) {
  return status ? terminalRunStatuses.has(status) : false;
}
```

The `record` action always replaces authoritative metadata and synchronizes `observedStatus` to `record.status`. The `event` action stores the original validated envelope, advances `lastEventId`, sets progress only for `workflow.progress`, and never reads a timestamp or Run ID from event data.

- [ ] **Step 4: Run reducer tests and verify GREEN**

Run: `cd frontend && npm test -- tests/run-reducer.test.ts`

Expected: all reducer tests PASS.

- [ ] **Step 5: Commit the state slice**

```bash
git add frontend/components/workbench/run-state.ts frontend/tests/run-reducer.test.ts
git commit -m "feat(frontend): model research run observation state"
```

### Task 5: Implement Reconciliation and Bounded Reconnection

**Files:**
- Create: `frontend/tests/run-observer.test.ts`
- Create: `frontend/components/workbench/run-observer.ts`

- [ ] **Step 1: Write failing observer tests for normal and terminal flows**

Drive the pure coordinator with injected functions and capture dispatched reducer actions:

```ts
it("gets the Run, streams it, then confirms a terminal event with GET", async () => {
  const getRun = vi.fn()
    .mockResolvedValueOnce(queuedRunFixture)
    .mockResolvedValueOnce(completedRunFixture);
  const stream = vi.fn(async ({ onEvent }) => {
    onEvent({ id: "1-0", event: "run.started", data: { resume: false } });
    onEvent({ id: "2-0", event: "run.completed", data: { final_output: "untrusted until GET" } });
  });
  const actions: RunAction[] = [];

  await observeResearchRun({
    runId: "demo-run",
    signal: new AbortController().signal,
    dispatch: (action) => actions.push(action),
    getRun,
    streamEvents: stream,
    wait: vi.fn(),
  });

  expect(getRun).toHaveBeenCalledTimes(2);
  expect(actions).toContainEqual(expect.objectContaining({
    type: "record", record: completedRunFixture,
  }));
  expect(stream).toHaveBeenCalledTimes(1);
});

it.each([failedRunFixture, timedOutRunFixture, interruptedRunFixture])(
  "does not connect when GET is already terminal: $status",
  async (record) => {
    const stream = vi.fn();
    await observeResearchRun({
      runId: record.run_id,
      signal: new AbortController().signal,
      dispatch: vi.fn(),
      getRun: vi.fn().mockResolvedValue(record),
      streamEvents: stream,
      wait: vi.fn(),
    });
    expect(stream).not.toHaveBeenCalled();
  },
);
```

- [ ] **Step 2: Add failing reconnect, cursor, and calibration tests**

Add separate tests asserting:

```ts
expect(stream.mock.calls[1][0].lastEventId).toBe("7-0");
expect(wait.mock.calls.map(([milliseconds]) => milliseconds)).toEqual([
  500, 1000, 2000, 4000, 8000,
]);
expect(stream).toHaveBeenCalledTimes(6); // initial plus five reconnects
```

Cover these exact branches:

- network error → GET returns `running` → reconnect with cursor;
- network error → GET returns `completed` → no wait and no reconnect;
- cursor-bearing `SseHttpError(500)` → GET remains running → exactly one cursorless retry;
- stream or GET 404 → one `not_found` action and stop;
- `SseProtocolError` → `protocol_error`, GET calibration, no cursor advancement;
- aborted signal → no error action and no retry;
- event from a prior Run is irrelevant because all actions include the observed Run ID.

- [ ] **Step 3: Run the observer test and verify RED**

Run: `cd frontend && npm test -- tests/run-observer.test.ts`

Expected: FAIL because the coordinator does not exist.

- [ ] **Step 4: Implement the observation loop**

Use explicit constants and injected boundaries:

```ts
export const reconnectDelays = [500, 1_000, 2_000, 4_000, 8_000] as const;

export async function observeResearchRun(options: ObserveResearchRunOptions) {
  let cursor: string | undefined;
  let cursorFallbackUsed = false;
  let reconnectAttempt = 0;
  let record = options.initialRecord ?? await readRecord(options);
  if (!record || options.signal.aborted) return;
  options.dispatch({ type: "record", runId: options.runId, record });
  if (isTerminalRunStatus(record.status)) return;

  while (!options.signal.aborted) {
    options.dispatch({
      type: "connection",
      runId: options.runId,
      connection: reconnectAttempt === 0 ? "connecting" : "reconnecting",
      error: null,
    });
    let terminalEventSeen = false;
    try {
      await options.streamEvents({
        runId: options.runId,
        lastEventId: cursor,
        signal: options.signal,
        onEvent(event) {
          cursor = event.id;
          terminalEventSeen ||= isTerminalEvent(event.event);
          options.dispatch({ type: "event", runId: options.runId, event });
          options.dispatch({ type: "connection", runId: options.runId, connection: "connected", error: null });
        },
      });
    } catch (error) {
      if (isAbort(error, options.signal)) return;
      options.dispatch(connectionFailure(options.runId, error));
      if (error instanceof SseHttpError && error.status === 404) {
        options.dispatch({ type: "not_found", runId: options.runId });
        options.onNotFound?.(options.runId);
        return;
      }
      if (error instanceof SseHttpError && cursor && !cursorFallbackUsed) {
        cursor = undefined;
        cursorFallbackUsed = true;
      }
    }

    record = await readRecord(options);
    if (!record || options.signal.aborted) return;
    options.dispatch({ type: "record", runId: options.runId, record });
    if (terminalEventSeen || isTerminalRunStatus(record.status)) {
      if (isTerminalRunStatus(record.status)) return;
    }
    options.dispatch({ type: "connection", runId: options.runId, connection: "disconnected", error: "事件流已断开，等待重连。" });
    if (reconnectAttempt >= reconnectDelays.length) return;
    await options.wait(reconnectDelays[reconnectAttempt], options.signal);
    reconnectAttempt += 1;
  }
}
```

Implement `readRecord` so `ApiHttpError(404)` dispatches not-found and calls `onNotFound`; `ApiProtocolError` becomes a protocol-specific connection error; network/other HTTP failures preserve the selected Run. Implement the default wait with an abort listener that clears its timeout.

- [ ] **Step 5: Run observer, reducer, and transport tests and verify GREEN**

Run: `cd frontend && npm test -- tests/run-observer.test.ts tests/run-reducer.test.ts tests/research-events-client.test.ts tests/sse-parser.test.ts`

Expected: all four suites PASS.

- [ ] **Step 6: Commit the observation coordinator**

```bash
git add frontend/components/workbench/run-observer.ts frontend/tests/run-observer.test.ts
git commit -m "feat(frontend): reconcile and reconnect research runs"
```

### Task 6: Add React Lifecycle and Session Recovery

**Files:**
- Create: `frontend/tests/run-selection.test.ts`
- Create: `frontend/components/workbench/run-selection.ts`
- Create: `frontend/components/workbench/use-research-run.ts`
- Create: `frontend/tests/run-observation.test.tsx`

- [ ] **Step 1: Write failing sessionStorage tests**

```ts
it("persists only the current Run ID", () => {
  storeSelectedRunId("run-1", sessionStorage);
  expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("run-1");
  expect(readSelectedRunId(sessionStorage)).toBe("run-1");
  clearSelectedRunId(sessionStorage);
  expect(readSelectedRunId(sessionStorage)).toBeNull();
});

it("degrades safely when storage access throws", () => {
  const blocked = {
    getItem: () => { throw new DOMException("blocked"); },
    setItem: () => { throw new DOMException("blocked"); },
    removeItem: () => { throw new DOMException("blocked"); },
  } as unknown as Storage;
  expect(readSelectedRunId(blocked)).toBeNull();
  expect(() => storeSelectedRunId("run", blocked)).not.toThrow();
  expect(() => clearSelectedRunId(blocked)).not.toThrow();
});
```

- [ ] **Step 2: Run the storage test and verify RED**

Run: `cd frontend && npm test -- tests/run-selection.test.ts`

Expected: FAIL because the storage module does not exist.

- [ ] **Step 3: Implement safe, versioned selection persistence**

```ts
export const RESEARCH_RUN_STORAGE_KEY = "insightagent.current-research-run.v1";

export function readSelectedRunId(storage: Storage): string | null {
  try {
    const value = storage.getItem(RESEARCH_RUN_STORAGE_KEY)?.trim();
    return value || null;
  } catch { return null; }
}

export function storeSelectedRunId(runId: string, storage: Storage): void {
  try { storage.setItem(RESEARCH_RUN_STORAGE_KEY, runId); } catch { /* unavailable */ }
}

export function clearSelectedRunId(storage: Storage): void {
  try { storage.removeItem(RESEARCH_RUN_STORAGE_KEY); } catch { /* unavailable */ }
}
```

- [ ] **Step 4: Write failing Hook lifecycle tests**

In `run-observation.test.tsx`, mock the coordinator boundary and require one active generation:

```ts
it("aborts observation on unmount", async () => {
  let observedSignal: AbortSignal | undefined;
  const observe = vi.fn(async (options) => {
    observedSignal = options.signal;
    await new Promise<void>((resolve) => options.signal.addEventListener("abort", () => resolve()));
  });
  const { unmount } = renderHook(() => useResearchRun({ runId: "run-1" }, { observe }));
  await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
  unmount();
  expect(observedSignal?.aborted).toBe(true);
});

it("aborts the old Run before observing a replacement", async () => {
  const signals = new Map<string, AbortSignal>();
  const observe = vi.fn(async (options) => { signals.set(options.runId, options.signal); });
  const { rerender } = renderHook(
    ({ runId }) => useResearchRun({ runId }, { observe }),
    { initialProps: { runId: "old-run" } },
  );
  rerender({ runId: "new-run" });
  await waitFor(() => expect(observe).toHaveBeenCalledTimes(2));
  expect(signals.get("old-run")?.aborted).toBe(true);
  expect(signals.get("new-run")?.aborted).toBe(false);
});
```

- [ ] **Step 5: Implement the one-effect Hook**

```ts
export function useResearchRun(
  selection: ActiveResearchRun | null,
  dependencies: UseResearchRunDependencies = {},
) {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  useEffect(() => {
    if (!selection) {
      dispatch({ type: "select", runId: null });
      return;
    }
    dispatch({ type: "select", runId: selection.runId });
    const controller = new AbortController();
    void (dependencies.observe ?? observeResearchRun)({
      runId: selection.runId,
      initialRecord: selection.initialRecord,
      signal: controller.signal,
      dispatch,
      onNotFound: dependencies.onNotFound,
    });
    return () => controller.abort();
  }, [selection?.runId, selection?.initialRecord, dependencies.observe, dependencies.onNotFound]);
  return state;
}
```

Define the dependency seam without weakening production types:

```ts
type UseResearchRunDependencies = {
  observe?: typeof observeResearchRun;
  onNotFound?: (runId: string) => void;
};
```

Memoize default callbacks at module scope and ensure `InsightWorkbench` supplies a stable `onNotFound` callback so React renders do not resubscribe.

- [ ] **Step 6: Run lifecycle and storage tests and verify GREEN**

Run: `cd frontend && npm test -- tests/run-selection.test.ts tests/run-observation.test.tsx`

Expected: storage and Hook lifecycle tests PASS.

- [ ] **Step 7: Commit persistence and Hook lifecycle**

```bash
git add frontend/components/workbench/run-selection.ts frontend/components/workbench/use-research-run.ts frontend/tests/run-selection.test.ts frontend/tests/run-observation.test.tsx
git commit -m "feat(frontend): restore one observed run per tab"
```

### Task 7: Wire the Live Workbench and Preserve Demo Isolation

**Files:**
- Modify: `frontend/components/workbench/workbench-types.ts`
- Modify: `frontend/components/workbench/insight-workbench.tsx`
- Modify: `frontend/components/workbench/research-workspace.tsx`
- Modify: `frontend/components/workbench/run-inspector.tsx`
- Modify: `frontend/app/globals.css`
- Modify: `frontend/tests/mocks/handlers.ts`
- Modify: `frontend/tests/query-submission.test.tsx`
- Modify: `frontend/tests/run-state.test.tsx`
- Modify: `frontend/tests/run-observation.test.tsx`

- [ ] **Step 1: Replace the old smart-Research test with a failing hydration test**

Require the follow-up Run GET before showing `thread_id`:

```ts
function finiteEventStream(runId: string, frames: string[]) {
  return http.get(`${runEndpoint}/${runId}/events`, () =>
    new HttpResponse(new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    }), {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      },
    }),
  );
}

it("hydrates thread_id after smart routing creates only a run_id", async () => {
  let runReads = 0;
  mockServer.use(
    http.post(queryEndpoint, () => HttpResponse.json(researchQueryResponseFixture, { status: 202 })),
    http.get(`${runEndpoint}/demo-run`, () => {
      runReads += 1;
      return HttpResponse.json(queuedRunFixture);
    }),
    finiteEventStream("demo-run", []),
  );
  const { user } = await renderReadyWorkbench();
  await user.click(screen.getByRole("button", { name: "智能提问" }));
  await user.click(screen.getByRole("button", { name: "提交智能提问" }));
  expect(await screen.findByText("demo-thread")).toBeInTheDocument();
  expect(runReads).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Add failing UI tests for live events and terminal states**

Use MSW `ReadableStream` responses to emit only real frames. Assert:

```ts
expect(await screen.findByText("run.started")).toBeInTheDocument();
expect(await screen.findByText("workflow.progress · planning")).toBeInTheDocument();
expect(screen.getByText("SSE 已连接")).toBeInTheDocument();
expect(screen.queryByText(/%/)).not.toBeInTheDocument();
expect(screen.queryByText(/Evidence Grader|Planner Task|Retriever/)).not.toBeInTheDocument();
```

For a terminal event, make the second GET return `completedRunFixture`, then assert the plain text contains the literal `# Research Report`, no Markdown heading is created from it, no citations are rendered, and no reconnect request occurs. Add parameterized failed/timed-out/interrupted GET tests and assert Resume never sends an HTTP request.

- [ ] **Step 3: Add failing remount, 404, and rapid-switch tests**

Cover:

- store `demo-run`, unmount, remount, then verify GET and SSE observation resume;
- GET 404 clears the storage key and shows “运行不存在”;
- submit/select `new-run` while an old stream is pending, release the old event, and assert it never appears;
- unmount while SSE is open and assert the request signal becomes aborted.

- [ ] **Step 4: Run UI tests and verify RED**

Run: `cd frontend && npm test -- tests/query-submission.test.tsx tests/run-observation.test.tsx tests/run-state.test.tsx`

Expected: FAIL because the production workbench still renders submission-only facts.

- [ ] **Step 5: Implement active selection in the workbench**

Replace `ConfirmedRun` with:

```ts
export type ActiveResearchRun = {
  runId: string;
  source: "smart" | "explicit" | "restored";
  question: string | null;
  initialRecord?: RunRecord;
};
```

Initialize production state from `sessionStorage`; do not touch storage in Demo mode. Smart Research stores `{runId, source: "smart", question}` without invented metadata. Explicit Research stores the same selection plus `initialRecord: result`. Direct/Analyze clears the active selection and storage. Call `useResearchRun` once with the active selection. The not-found callback must clear only when its Run ID still matches the current selection.

- [ ] **Step 6: Render truthful live workspace state**

Derive copy from `state.observedStatus ?? state.record?.status`. Use labels:

```ts
const RUN_LABELS: Record<RunStatus, string> = {
  queued: "任务已排队",
  running: "研究运行中",
  completed: "研究已完成",
  failed: "研究失败",
  timed_out: "研究超时",
  interrupted: "研究已中断",
};
```

Show the stored in-memory question when available, otherwise “重新挂载后任务文本不可用”. For progress, render only `stage` and string `message` fields received from `workflow.progress`. Render `record.final_output` inside a `<pre className="report-preview">` so React escapes it and does not interpret Markdown. Do not derive percentages.

- [ ] **Step 7: Render the real Run Inspector**

Display `record.run_id`, `record.thread_id`, observed/record status, connection label, backend `record.error`, and `state.events`. Use the event ID as key and show the exact type plus a conservative note generated only from known string/boolean/number data. Unknown events remain visible without semantic claims.

The Resume button remains `type="button"` with no request handler. It may be visually enabled for `interrupted` or `timed_out`, but its adjacent copy must say that this phase does not send Resume requests. Live citation/source content remains empty.

- [ ] **Step 8: Add minimal supporting styles**

Add styles for `.connection-chip`, `.progress-fact`, `.report-preview`, and wrapping inspector IDs. Reuse existing color variables and spacing; do not redesign the grid, typography system, or Demo views.

- [ ] **Step 9: Run the UI slices and verify GREEN**

Run: `cd frontend && npm test -- tests/query-submission.test.tsx tests/run-observation.test.tsx tests/run-state.test.tsx tests/workbench.test.tsx tests/responsive.test.tsx`

Expected: live and Demo suites PASS; no act warnings or unhandled requests appear.

- [ ] **Step 10: Commit the workbench integration**

```bash
git add frontend/components/workbench frontend/app/globals.css frontend/tests/mocks/handlers.ts frontend/tests/query-submission.test.tsx frontend/tests/run-state.test.tsx frontend/tests/run-observation.test.tsx
git commit -m "feat(frontend): observe live research run state"
```

### Task 8: Verify the Next.js Rewrite Streams SSE in Real Time

**Files:**
- Create: `frontend/scripts/verify-sse-proxy.mjs`
- Modify: `frontend/package.json`
- Modify: `frontend/README.md`

- [ ] **Step 1: Add a controlled non-MSW proxy verification script**

The script must:

1. start a Node HTTP backend on `127.0.0.1` with an ephemeral port;
2. expose `/health` and `/v1/research/runs/proxy-run/events`;
3. write the first SSE frame immediately, delay the second by 750 ms, then end;
4. spawn `next dev --webpack` with `INSIGHTAGENT_BACKEND_ORIGIN` pointed at the controlled backend;
5. fetch through the Next.js origin and assert the first frame arrives at least 400 ms before stream completion;
6. assert `content-type` starts with `text/event-stream` and `cache-control` contains `no-cache`;
7. abort the first browser-side request and assert the backend receives `close`;
8. reconnect with `Last-Event-ID: 1-0` and assert the backend sees exactly `1-0`;
9. terminate both local processes in `finally`.

Use Node `assert`, `http`, `child_process.spawn`, `events.once`, and native `fetch`; do not import MSW or call FastAPI. The essential timing assertion is:

```js
const startedAt = performance.now();
const first = await reader.read();
assert.match(decoder.decode(first.value), /id: 1-0/);
const firstAt = performance.now();
const remainder = await readToEnd(reader, decoder);
const completedAt = performance.now();
assert.ok(firstAt - startedAt < 600, "proxy buffered the first SSE frame");
assert.ok(completedAt - firstAt >= 400, "test backend did not preserve the delayed boundary");
assert.match(remainder, /id: 2-0/);
```

- [ ] **Step 2: Add the package command**

```json
"test:sse-proxy": "node scripts/verify-sse-proxy.mjs"
```

- [ ] **Step 3: Run the proxy verifier**

Run: `cd frontend && npm run test:sse-proxy`

Expected: PASS with explicit lines for early-frame streaming, content type/cache headers, abort propagation, and `Last-Event-ID: 1-0`. If it fails, preserve the timing/header output as reproducible evidence before changing `next.config.ts`; make only the smallest proxy change proven necessary, then rerun.

- [ ] **Step 4: Document the verification command and boundary**

In `frontend/README.md`, document that `sessionStorage` retains only the current Run ID, and that `npm run test:sse-proxy` starts only controlled local Node/Next services and never creates a Research Run.

- [ ] **Step 5: Commit the proxy verification**

```bash
git add frontend/scripts/verify-sse-proxy.mjs frontend/package.json frontend/README.md frontend/next.config.ts
git commit -m "test(frontend): verify SSE rewrite streaming"
```

Do not stage `frontend/next.config.ts` if the existing rewrite passes unchanged.

### Task 9: Full Verification and Real-Service Smoke Test

**Files:**
- Modify only if failures reveal a scoped defect in the files above.

- [ ] **Step 1: Run protocol and observation focused suites**

Run:

```bash
cd frontend
npm test -- tests/api-client.test.ts tests/protocol.test.ts tests/sse-parser.test.ts tests/research-events-client.test.ts tests/run-reducer.test.ts tests/run-observer.test.ts tests/run-selection.test.ts tests/run-observation.test.tsx tests/query-submission.test.tsx tests/run-state.test.tsx
```

Expected: all focused suites PASS with zero unhandled requests, stream errors, or React act warnings.

- [ ] **Step 2: Run all frontend quality gates**

Run each command separately from `frontend/` and record its fresh output:

```bash
npm run typecheck
npm run lint
npm test
npm run build
npm run test:sse-proxy
```

Expected: every command exits 0. Report the exact Vitest test/file counts and production build result.

- [ ] **Step 3: Inspect the repository Python instructions and environments**

Run:

```bash
sed -n '36,225p' README.md
conda env list
```

Expected: confirm the repository's named `insight-agent` environment exists before running Python. Do not install packages, create a venv, or alter any Conda environment. If the named environment does not exist, stop the real-service portion and report that boundary.

- [ ] **Step 4: Discover existing service state before creating work**

Check liveness/readiness and running processes without mutation:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --silent --show-error http://127.0.0.1:8000/ready
ps -ef | rg 'uvicorn|next dev|next-server' || true
```

If an existing Run ID is available from the current session or service logs, use it. Do not enumerate Redis directly or create a new run until GET proves no known Run is usable.

- [ ] **Step 5: Start only missing local services using documented commands**

When the confirmed `insight-agent` environment and required infrastructure are available, run FastAPI exactly as documented:

```bash
conda run --no-capture-output -n insight-agent \
  python -m uvicorn insight_agent.runtime.app:app \
  --host 127.0.0.1 --port 8000
```

Start the frontend from `frontend/` with `npm run dev`. Keep process IDs and stop only processes started by this task.

- [ ] **Step 6: Perform the bounded real-service smoke test**

Through `http://127.0.0.1:3000`, verify:

1. GET a known RunRecord and record `run_id`, `thread_id`, and status.
2. Open its SSE stream and read at least one real `id/event/data` frame.
3. Abort that request.
4. Reconnect with the received ID in `Last-Event-ID` and record the next replayed frame or clean terminal close.
5. GET the RunRecord again and record the calibrated status and `final_output` availability.
6. Open the browser workbench only if needed to confirm the same facts are rendered.

If no known Run exists, create at most one tightly scoped Research run using already indexed local material. Do not repeatedly retry a paid or Web-backed workflow. Record whether the created run ends completed, failed, timed out, interrupted, or remains running when observation stops.

- [ ] **Step 7: Review the requirement checklist and diff**

Run:

```bash
git diff --check
git status --short
git diff --stat 358a621..HEAD
```

Confirm: no Python agent/runtime algorithm files changed; no Resume request exists; no mock production fallback, fake percentages, citations, node stages, timestamps, thread IDs, or reports were introduced.

- [ ] **Step 8: Commit only scoped verification fixes if needed**

If verification required code changes, repeat the failing command, then the full relevant gate, before committing:

```bash
git diff --name-only
git add frontend/components/workbench/run-observer.ts frontend/tests/run-observer.test.ts
git commit -m "fix(frontend): harden research run observation"
```

The two paths above illustrate an observer fix. If the fresh diff identifies a different scoped file, substitute its exact printed path and do not stage unrelated files.

If no fix was required, do not create an empty commit.

## Plan Self-Review

- Every design requirement maps to a task: Run GET (Task 1), incremental SSE (Tasks 2–3), state semantics and dedupe (Task 4), reconciliation/reconnect/cursor fallback (Task 5), Strict Mode/unmount/remount/switch safety (Task 6), live three-column UI and honesty constraints (Task 7), real rewrite streaming evidence (Task 8), and full gates/real-service smoke (Task 9).
- Runtime event `data` is always treated as the SSE payload object, never as a complete RuntimeEvent or RunRecord.
- Only GET RunRecord supplies authoritative `thread_id`, timestamps, error, and final output.
- No task modifies Python backend behavior or calls Resume.
- The plan adds no package dependency and contains no unresolved placeholder.
