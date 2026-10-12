# Frontend Phase Three Run Observation Design

## Scope

Connect the existing three-column Next.js workbench to the real research-run read endpoint and SSE stream. A submitted Research run must expose its authoritative metadata, backend status, observed events, connection health, and final output. Observation must survive a same-tab page refresh or component remount and recover safely from a dropped stream.

This phase does not call the Resume endpoint, parse citations, render Markdown, change runtime or agent algorithms, fabricate node-level progress, or introduce a second realtime transport. Existing mock scenarios remain isolated to explicit demo/test usage and never back production state.

## Confirmed Backend Contract

- `GET /v1/research/runs/{run_id}` returns a complete `RunRecord` with `run_id`, `thread_id`, `status`, timestamps, nullable `error`, and nullable `final_output`; an unknown run returns 404.
- `GET /v1/research/runs/{run_id}/events` responds as `text/event-stream`, sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, accepts `Last-Event-ID`, and replays events strictly after that cursor.
- Each SSE frame contains `id`, `event`, and JSON `data`. The JSON value is only `RuntimeEvent.data`; it is not a complete `RuntimeEvent` or `RunRecord` and has no `run_id` or `created_at`.
- Emitted event types currently include `run.queued`, `run.started`, `workflow.progress`, `run.completed`, `run.failed`, `run.timed_out`, and `run.interrupted`. Resume may also emit `run.resuming`, which remains an unknown-but-displayable event in this phase because no Resume request is sent.
- Current workflow progress is intentionally coarse. The runner emits stages such as `planning`, `resuming`, and `completed`, with optional messages. `workflow.progress` with stage `completed` is not a terminal Run transition.
- The server closes the stream after a terminal event. If status becomes terminal before that event is read, it makes one final read before closing.

## Architecture

The implementation uses three focused layers.

1. `frontend/lib/api-client.ts` adds `getResearchRun(runId, signal)` on top of the existing JSON/Zod request pipeline.
2. A standalone client under `frontend/lib/research-events/` owns incremental SSE decoding and one HTTP stream connection. It has no React dependency.
3. A reducer and observation hook under `frontend/components/workbench/` own one current Run's state and lifecycle. The workbench supplies a selected Run ID; the hook queries, subscribes, reconciles, retries, and returns presentation-ready facts.

The existing three-column components remain responsible for display only. They receive the observed Run state rather than creating transport side effects themselves. No global state library or new backend endpoint is introduced.

## Run Selection and Remount Recovery

Only the current `run_id` is persisted in `sessionStorage`, under a versioned frontend-owned key. A successful Research submission replaces that key. A production workbench mount reads it once and starts observation. This restores the current run after a same-tab refresh without inventing or persisting metadata that belongs to the backend.

A Direct or Analyze result clears the active Research selection because the workbench has switched to a synchronous answer. A 404 from the authoritative Run endpoint clears the persisted ID and exposes a not-found state. Other failures leave the ID available for retry. Demo scenarios do not read or write session storage.

## Run State Model

The reducer keeps backend execution state separate from transport state.

Authoritative execution data consists of the latest validated `RunRecord`. Its status is one of `queued`, `running`, `completed`, `failed`, `timed_out`, or `interrupted`. SSE events may provide immediate presentation hints, but terminal state and `final_output` are confirmed by a subsequent GET.

Connection state consists of `idle`, `connecting`, `connected`, `reconnecting`, `disconnected`, `protocol_error`, or `not_found`. It never changes a nonterminal Run to `failed` merely because the network dropped.

The reducer also stores:

- the selected Run ID and an observation generation;
- observed SSE events in arrival order;
- a set of handled event IDs for deduplication;
- the last successfully handled event ID;
- the latest real workflow-progress payload;
- a safe observer error suitable for the UI.

Reducer actions carry the Run ID or generation. Actions from an obsolete observation are ignored, preventing a delayed GET, stream read, or retry from overwriting a newly selected Run.

## SSE Parsing and Validation

The standalone parser consumes decoded text incrementally and retains an unfinished line or frame between network chunks. It accepts LF and CRLF, ignores comment lines, treats a blank line as the dispatch boundary, joins repeated `data` fields with a newline, and supports standard `id` and `event` fields. A final incomplete event without a dispatching blank line is not emitted.

For each dispatched frame, the client JSON-decodes the joined data and validates `{id, event, data}` with the existing Zod SSE schema. Data must be a JSON object because that is the backend contract. A malformed JSON payload or schema mismatch is a protocol error, distinct from an HTTP response or network interruption.

Only validated events advance the stored cursor. Duplicate IDs are ignored before they reach the visible timeline. Unknown event types are retained with their actual type and a conservative payload summary; they cannot crash the reducer or imply an unsupported workflow stage.

The stream request uses `fetch`, `Accept: text/event-stream`, `Cache-Control: no-cache`, and the caller's `AbortSignal`. When a cursor exists it adds `Last-Event-ID`. It rejects a non-2xx response as a typed HTTP error and rejects an unexpected content type as a protocol error.

## Observation, Reconciliation, and Reconnection

Observation follows this sequence:

1. GET the selected Run and validate the `RunRecord`.
2. If it is terminal, render it and do not open SSE.
3. If it is nonterminal, open exactly one SSE connection.
4. Apply validated events. `run.queued` and `run.started` provide prompt status feedback; `workflow.progress` stores only its real payload.
5. On a terminal event, close the stream and GET again. The returned RunRecord is authoritative for status, error, and final output.

When the stream ends without a terminal event or throws a network error, connection state becomes disconnected and an immediate GET calibrates the Run. A terminal GET stops all retries. A nonterminal GET schedules another connection with bounded exponential delays of 500 ms, 1 s, 2 s, 4 s, and 8 s. After five consecutive unsuccessful reconnects, the observer remains disconnected and stops automatic retries; remounting or selecting the Run again starts a fresh observation cycle.

The first reconnect sends the last handled event ID. If a cursor-bearing request receives a non-404 HTTP failure that can represent an expired or invalid replay cursor, the observer first performs GET calibration and then permits one cursorless replay attempt. Existing event IDs suppress duplicates. A 404 is terminal for observation and clears persisted selection. Protocol errors are displayed distinctly and trigger GET calibration, but malformed event data is never committed or used to advance the cursor.

An effect owns one `AbortController` and all retry timers for one observation generation. Cleanup increments the generation, aborts the active fetch, and clears timers. React Strict Mode's cleanup therefore tears down the first development-only subscription before the second effect owns a new one. Selecting another Run performs the same cleanup, so the old Run cannot update the new Run.

## Workbench Presentation

The visual structure and established styling remain intact.

The center Research Workspace displays the real Run status, the submitted research question for the current in-memory submission, the most recent real workflow-progress stage/message, and a conservative connection notice. Completed runs show that a final report is available and render `final_output` as escaped plain text. They do not parse Markdown or citations. A remounted run may not have its original question because the backend RunRecord does not expose it; the UI states that the task text is unavailable instead of fabricating one.

The Run Inspector displays real `run_id`, `thread_id`, Run status, SSE connection status, backend error, and an ordered timeline made only from validated SSE events. Interrupted and timed-out states may visually enable the existing Resume affordance, but the button remains non-operational and explicitly states that Resume communication is outside this phase. The sources area remains empty for live production runs.

## Error Semantics

- Run GET 404: mark not found, clear session storage, stop observation.
- Other Run GET HTTP failures: retain the selected Run and show a status-query error.
- SSE 404: mark not found, clear session storage, stop observation.
- SSE network interruption or premature EOF: reconcile with GET; retry only if the Run remains nonterminal.
- SSE non-404 HTTP failure: reconcile, then retry within the bounded policy; a cursorless fallback is allowed once when a replay cursor was sent.
- SSE protocol failure: retain already validated events, do not advance past the malformed event, reconcile with GET, and present a protocol-specific connection state.
- Abort caused by cleanup or Run switching: silent cancellation, with no error state or retry.

## Testing Strategy

Tests follow red-green-refactor and use the existing Vitest, Testing Library, and MSW setup where appropriate.

Pure parser/client tests cover frames split across arbitrary chunks, CRLF, comments, blank lines, repeated data lines, IDs, malformed JSON, schema-invalid data, headers, abort, content type, and non-2xx responses.

Reducer and observation tests cover:

- queued to running to completed;
- failed, timed out, and interrupted terminal states;
- `workflow.progress: completed` remaining nonterminal;
- event-ID deduplication and cursor advancement only after validation;
- Last-Event-ID reconnect;
- network disconnect followed by authoritative GET calibration;
- terminal calibration stopping reconnects;
- invalid-cursor cursorless fallback;
- finite exponential retry behavior with fake timers;
- unmount abort and timer cleanup;
- rapid Run switching and stale-result suppression;
- sessionStorage remount recovery and 404 cleanup.

MSW UI/API tests cover RunRecord retrieval, the extra GET after smart Research submission to obtain `thread_id`, explicit Research observation, real progress and timeline display, terminal/error presentation, final-output plain-text preview, and the absence of fabricated percentages, planner tasks, evidence nodes, citations, or mock reports.

A controlled streaming integration test uses a real local streaming HTTP server and the Next.js development or production proxy rather than MSW. It measures receipt of an early frame before the stream closes, verifies `text/event-stream`, `Cache-Control`, `Last-Event-ID` forwarding, and abort-driven disconnect. This provides evidence for whether the existing rewrite streams correctly before any proxy change is considered.

## Verification and Real-Service Smoke Test

Before any Python setup or execution, inspect repository instructions and `conda env list`. Use the already confirmed repository Conda environment if one exists; do not alter an environment without approval.

Frontend verification runs typecheck, ESLint, the complete Vitest suite, and the production build. The controlled streaming test is recorded separately from MSW coverage.

For the real-service smoke test, first query any known existing Run ID that is still present. If none exists, create at most one narrowly scoped Research run using indexed local material. Through the Next.js origin, verify RunRecord retrieval, at least one real event, connection abort, reconnect with the last cursor, and GET calibration. Record the Run's final observed status. Do not repeatedly trigger paid Web Research.

## Non-Goals

- Calling `POST /v1/research/runs/{run_id}/resume`
- Citation parsing or source extraction
- Markdown or rich report rendering
- Invented percentage, planner-task, retriever, grader, or evidence progress
- Backend runtime, Agent, Planner, Retriever, Evidence, Reporting, or Evaluation changes
- WebSocket or an additional realtime service
- Multi-Run history or cross-tab persistence
- Zustand, Redux, or another global state dependency
