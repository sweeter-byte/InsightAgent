import { describe, expect, it } from "vitest";

import {
  initialRunState,
  isTerminalEvent,
  isTerminalRunStatus,
  runReducer,
  selectRun,
  terminalRunStatuses,
} from "@/components/workbench/run-state";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";

function record(
  runId: string,
  status: RunRecord["status"] = "queued",
  error: string | null = null,
): RunRecord {
  return {
    run_id: runId,
    thread_id: `thread-${runId}`,
    status,
    created_at: "2026-10-12T00:00:00Z",
    updated_at: "2026-10-12T00:01:00Z",
    error,
    final_output: null,
  };
}

function event(
  id: string,
  eventName: string,
  data: SseEvent["data"] = {},
): SseEvent {
  return { id, event: eventName, data };
}

describe("runReducer", () => {
  it("observes a queued event from an unknown initial status", () => {
    const state = runReducer(selectRun(initialRunState, "run-1"), {
      type: "event",
      runId: "run-1",
      event: event("queued", "run.queued"),
    });

    expect(state.observedStatus).toBe("queued");
  });

  it("observes queued record, started event, and completed event", () => {
    let state = selectRun(initialRunState, "run-1");
    const queued = record("run-1");

    state = runReducer(state, {
      type: "record",
      runId: "run-1",
      record: queued,
    });
    expect(state.record).toBe(queued);
    expect(state.observedStatus).toBe("queued");

    state = runReducer(state, {
      type: "event",
      runId: "run-1",
      event: event("event-1", "run.started"),
    });
    expect(state.observedStatus).toBe("running");

    state = runReducer(state, {
      type: "event",
      runId: "run-1",
      event: event("event-2", "run.completed"),
    });
    expect(state.observedStatus).toBe("completed");
    expect(state.events.map(({ id }) => id)).toEqual(["event-1", "event-2"]);
    expect(state.lastEventId).toBe("event-2");
  });

  it.each([
    ["run.failed", "failed"],
    ["run.timed_out", "timed_out"],
    ["run.interrupted", "interrupted"],
  ] as const)("maps %s to %s", (eventName, status) => {
    const state = runReducer(selectRun(initialRunState, "run-1"), {
      type: "event",
      runId: "run-1",
      event: event("terminal", eventName),
    });

    expect(state.observedStatus).toBe(status);
  });

  it("keeps workflow.progress data verbatim without treating completed stage as terminal", () => {
    const progress = { stage: "completed", detail: { sources: 3 } };
    const running = runReducer(selectRun(initialRunState, "run-1"), {
      type: "record",
      runId: "run-1",
      record: record("run-1", "running"),
    });

    const state = runReducer(running, {
      type: "event",
      runId: "run-1",
      event: event("progress-1", "workflow.progress", progress),
    });

    expect(state.latestProgress).toBe(progress);
    expect(state.observedStatus).toBe("running");
    expect(isTerminalEvent(state.events[0]!)).toBe(false);
  });

  it("appends unknown events without changing execution status", () => {
    const running = runReducer(selectRun(initialRunState, "run-1"), {
      type: "record",
      runId: "run-1",
      record: record("run-1", "running"),
    });
    const unknown = event("unknown-1", "source.found", { url: "/source" });

    const state = runReducer(running, {
      type: "event",
      runId: "run-1",
      event: unknown,
    });

    expect(state.events).toEqual([unknown]);
    expect(state.observedStatus).toBe("running");
  });

  it("deduplicates event IDs without updating event-derived state", () => {
    const progress = event("same-id", "workflow.progress", { stage: "search" });
    const first = runReducer(selectRun(initialRunState, "run-1"), {
      type: "event",
      runId: "run-1",
      event: progress,
    });

    const duplicate = runReducer(first, {
      type: "event",
      runId: "run-1",
      event: event("same-id", "run.completed", { stage: "completed" }),
    });

    expect(duplicate).toBe(first);
    expect(duplicate.events).toEqual([progress]);
    expect(duplicate.lastEventId).toBe("same-id");
    expect(duplicate.latestProgress).toBe(progress.data);
    expect(duplicate.observedStatus).toBeNull();
  });

  it("ignores every stale-run action by object identity", () => {
    const state = selectRun(initialRunState, "run-current");

    expect(
      runReducer(state, {
        type: "record",
        runId: "run-old",
        record: record("run-old", "completed"),
      }),
    ).toBe(state);
    expect(
      runReducer(state, {
        type: "event",
        runId: "run-old",
        event: event("old-event", "run.completed"),
      }),
    ).toBe(state);
    expect(
      runReducer(state, {
        type: "connection",
        runId: "run-old",
        connection: "protocol_error",
        error: "old error",
      }),
    ).toBe(state);
  });

  it("defends against a record whose run_id mismatches the action", () => {
    const state = selectRun(initialRunState, "run-1");

    expect(
      runReducer(state, {
        type: "record",
        runId: "run-1",
        record: record("run-other", "completed"),
      }),
    ).toBe(state);
  });

  it("does not regress running or terminal state during queued replay", () => {
    const running = runReducer(selectRun(initialRunState, "run-1"), {
      type: "event",
      runId: "run-1",
      event: event("started", "run.started"),
    });
    const replayedQueued = runReducer(running, {
      type: "event",
      runId: "run-1",
      event: event("queued-late", "run.queued"),
    });
    expect(replayedQueued.observedStatus).toBe("running");

    const terminal = runReducer(replayedQueued, {
      type: "event",
      runId: "run-1",
      event: event("completed", "run.completed"),
    });
    const replayedStarted = runReducer(terminal, {
      type: "event",
      runId: "run-1",
      event: event("started-late", "run.started"),
    });
    const queuedAfterTerminal = runReducer(replayedStarted, {
      type: "event",
      runId: "run-1",
      event: event("queued-later", "run.queued"),
    });

    expect(replayedStarted.observedStatus).toBe("completed");
    expect(queuedAfterTerminal.observedStatus).toBe("completed");
  });

  it("allows terminal events to replace a prior terminal observation", () => {
    const completed = runReducer(selectRun(initialRunState, "run-1"), {
      type: "event",
      runId: "run-1",
      event: event("completed", "run.completed"),
    });
    const failed = runReducer(completed, {
      type: "event",
      runId: "run-1",
      event: event("failed", "run.failed"),
    });
    expect(failed.observedStatus).toBe("failed");

    const authoritative = record("run-1", "completed");
    const refreshed = runReducer(failed, {
      type: "record",
      runId: "run-1",
      record: authoritative,
    });
    expect(refreshed.observedStatus).toBe("completed");
    expect(refreshed.record).toBe(authoritative);
  });

  it("keeps connection state separate from execution state", () => {
    const runningRecord = record("run-1", "running");
    const running = runReducer(selectRun(initialRunState, "run-1"), {
      type: "record",
      runId: "run-1",
      record: runningRecord,
    });
    const disconnected = runReducer(running, {
      type: "connection",
      runId: "run-1",
      connection: "disconnected",
      error: "stream ended",
    });

    expect(disconnected.observedStatus).toBe("running");
    expect(disconnected.record).toBe(runningRecord);
    expect(disconnected.connection).toBe("disconnected");
    expect(disconnected.error).toBe("stream ended");

    const missing = runReducer(disconnected, {
      type: "connection",
      runId: "run-1",
      connection: "not_found",
      error: "run not found",
    });
    expect(missing.runId).toBe("run-1");
    expect(missing.record).toBeNull();
    expect(missing.observedStatus).toBe("running");
    expect(missing.connection).toBe("not_found");
    expect(missing.error).toBe("run not found");
  });

  it("selecting a new run or null clears all prior-run observations", () => {
    let state = runReducer(selectRun(initialRunState, "run-old"), {
      type: "event",
      runId: "run-old",
      event: event("event-old", "workflow.progress", { stage: "search" }),
    });
    state = runReducer(state, {
      type: "connection",
      runId: "run-old",
      connection: "protocol_error",
      error: "bad frame",
    });

    const next = selectRun(state, "run-new");
    expect(next).toEqual({
      runId: "run-new",
      record: null,
      observedStatus: null,
      connection: "idle",
      events: [],
      seenEventIds: new Set(),
      lastEventId: null,
      latestProgress: null,
      error: null,
    });

    expect(
      runReducer(next, { type: "select", runId: null }),
    ).toEqual(initialRunState);
  });
});

describe("terminal helpers", () => {
  it("contains exactly the protocol terminal run statuses", () => {
    expect(terminalRunStatuses).toEqual([
      "completed",
      "failed",
      "timed_out",
      "interrupted",
    ]);
    expect(isTerminalRunStatus("queued")).toBe(false);
    expect(isTerminalRunStatus("running")).toBe(false);
    for (const status of terminalRunStatuses) {
      expect(isTerminalRunStatus(status)).toBe(true);
    }
  });

  it.each([
    ["run.completed", true],
    ["run.failed", true],
    ["run.timed_out", true],
    ["run.interrupted", true],
    ["run.started", false],
    ["run.resuming", false],
    ["workflow.progress", false],
  ])("classifies %s terminal=%s", (eventName, terminal) => {
    expect(isTerminalEvent(event("event-1", eventName))).toBe(terminal);
  });
});
