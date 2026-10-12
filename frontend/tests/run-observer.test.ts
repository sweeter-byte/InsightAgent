import { describe, expect, it, vi } from "vitest";

import {
  abortableWait,
  observeResearchRun,
  reconnectDelays,
} from "@/components/workbench/run-observer";
import {
  initialRunState,
  runReducer,
  selectRun,
  type RunAction,
} from "@/components/workbench/run-state";
import { ApiHttpError, ApiNetworkError, ApiProtocolError } from "@/lib/api-client";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";
import {
  SseHttpError,
  SseNetworkError,
  SseProtocolError,
} from "@/lib/research-events/client";

const RUN_ID = "run-1";
const GENERATION = 7;

function record(
  status: RunRecord["status"],
  overrides: Partial<RunRecord> = {},
): RunRecord {
  return {
    run_id: RUN_ID,
    thread_id: "thread-1",
    status,
    created_at: "2026-10-12T00:00:00Z",
    updated_at: "2026-10-12T00:01:00Z",
    error: null,
    final_output: null,
    ...overrides,
  };
}

function event(id: string, name: string, data: SseEvent["data"] = {}): SseEvent {
  return { id, event: name, data };
}

function harness() {
  let state = selectRun(initialRunState, RUN_ID, GENERATION);
  const actions: RunAction[] = [];
  const dispatch = (action: RunAction) => {
    actions.push(action);
    state = runReducer(state, action);
  };
  return { actions, dispatch, state: () => state };
}

describe("observeResearchRun", () => {
  it("GETs queued state, observes events, and takes terminal output only from final GET", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("queued"))
      .mockResolvedValueOnce(
        record("completed", { final_output: "authoritative report" }),
      );
    const streamEvents = vi.fn(async ({ onEvent }) => {
      onEvent(event("1", "run.started"));
      onEvent(event("2", "workflow.progress", { stage: "writing" }));
      onEvent(
        event("3", "run.completed", { final_output: "untrusted event output" }),
      );
    });

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
    });

    expect(getRun).toHaveBeenCalledTimes(2);
    expect(streamEvents).toHaveBeenCalledTimes(1);
    expect(h.state().record?.final_output).toBe("authoritative report");
    expect(h.state().events.map(({ id }) => id)).toEqual(["1", "2", "3"]);
    expect(h.actions.map((action) => action.type)).toEqual([
      "record",
      "connection",
      "connection",
      "event",
      "event",
      "event",
      "record",
    ]);
    expect(h.actions.every((action) => action.generation === GENERATION)).toBe(true);
    expect(h.actions.every((action) => action.runId === RUN_ID)).toBe(true);
  });

  it("aborts the current attempt after a terminal event and immediately reconciles", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(
        record("completed", { final_output: "authoritative report" }),
      );
    const streamEvents = vi.fn(
      ({ signal, onEvent }): Promise<void> =>
        new Promise((resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
          onEvent(event("terminal", "run.completed"));
          onEvent(event("after-terminal", "workflow.progress"));
          if (signal?.aborted) resolve();
        }),
    );

    const observation = observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
    });
    const timeout = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("observer did not abort terminal stream")), 50);
    });

    await expect(Promise.race([observation, timeout])).resolves.toBeUndefined();
    expect(getRun).toHaveBeenCalledTimes(2);
    expect(h.state().record?.final_output).toBe("authoritative report");
    expect(h.state().events.map(({ id }) => id)).toEqual(["terminal"]);
    expect(
      h.actions.filter(
        (action) =>
          action.type === "connection" &&
          ["disconnected", "protocol_error", "not_found"].includes(
            action.connection,
          ),
      ),
    ).toEqual([]);
  });

  it("uses a matching initial record without GET and skips streams for every terminal status", async () => {
    for (const status of ["completed", "failed", "timed_out", "interrupted"] as const) {
      const h = harness();
      const getRun = vi.fn();
      const streamEvents = vi.fn();
      await observeResearchRun({
        runId: RUN_ID,
        generation: GENERATION,
        signal: new AbortController().signal,
        dispatch: h.dispatch,
        initialRecord: record(status),
        getRun,
        streamEvents,
      });
      expect(getRun).not.toHaveBeenCalled();
      expect(streamEvents).not.toHaveBeenCalled();
      expect(h.state().observedStatus).toBe(status);
    }
  });

  it("ignores a mismatched initial record and GETs the selected run", async () => {
    const h = harness();
    const authoritative = record("completed");
    const getRun = vi.fn().mockResolvedValue(authoritative);
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      initialRecord: record("running", { run_id: "other-run" }),
      getRun,
      streamEvents: vi.fn(),
    });
    expect(getRun).toHaveBeenCalledWith(RUN_ID, expect.any(AbortSignal));
    expect(h.state().record).toBe(authoritative);
  });

  it("does not treat workflow.progress stage completed as terminal", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("completed"));
    const streamEvents = vi
      .fn()
      .mockImplementationOnce(async ({ onEvent }) => {
        onEvent(event("1", "workflow.progress", { stage: "completed" }));
      })
      .mockResolvedValueOnce(undefined);
    const wait = vi.fn().mockResolvedValue(undefined);

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait,
    });

    expect(streamEvents).toHaveBeenCalledTimes(2);
    expect(wait).toHaveBeenCalledWith(500, expect.any(AbortSignal));
  });

  it("marks an ended stream disconnected when GET disproves a terminal candidate", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("completed"));
    const streamEvents = vi
      .fn()
      .mockImplementationOnce(async ({ onEvent }) => {
        onEvent(event("candidate", "run.completed"));
      })
      .mockResolvedValueOnce(undefined);

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait: vi.fn().mockResolvedValue(undefined),
    });

    const firstRecordAfterCandidate = h.actions.findIndex(
      (action, index) => index > 0 && action.type === "record",
    );
    expect(h.actions[firstRecordAfterCandidate + 1]).toEqual(
      expect.objectContaining({ type: "connection", connection: "disconnected" }),
    );
  });

  it("reconciles a network disconnect, waits, and reconnects with the last cursor", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("completed"));
    const streamEvents = vi
      .fn()
      .mockImplementationOnce(async ({ onEvent }) => {
        onEvent(event("cursor-1", "run.started"));
        throw new SseNetworkError();
      })
      .mockResolvedValueOnce(undefined);
    const wait = vi.fn().mockResolvedValue(undefined);

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait,
    });

    expect(wait).toHaveBeenCalledTimes(1);
    expect(wait).toHaveBeenCalledWith(500, expect.any(AbortSignal));
    expect(streamEvents.mock.calls[1]?.[0].lastEventId).toBe("cursor-1");
    expect(h.actions).toContainEqual(
      expect.objectContaining({ type: "connection", connection: "disconnected" }),
    );
  });

  it("stops immediately when reconciliation finds a terminal run", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("failed", { error: "authoritative" }));
    const streamEvents = vi.fn().mockRejectedValue(new SseNetworkError());
    const wait = vi.fn();
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait,
    });
    expect(streamEvents).toHaveBeenCalledTimes(1);
    expect(wait).not.toHaveBeenCalled();
    expect(h.state().record?.error).toBe("authoritative");
  });

  it("uses exactly five backoff delays and at most six streams", async () => {
    const h = harness();
    const getRun = vi.fn().mockResolvedValue(record("running"));
    const streamEvents = vi.fn().mockResolvedValue(undefined);
    const wait = vi.fn().mockResolvedValue(undefined);
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait,
    });
    expect(streamEvents).toHaveBeenCalledTimes(6);
    expect(wait.mock.calls.map(([ms]) => ms)).toEqual([...reconnectDelays]);
  });

  it("allows one cursorless replay after cursor HTTP failure without duplicate dispatch or cursor rollback", async () => {
    const h = harness();
    const getRun = vi
      .fn()
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("running"))
      .mockResolvedValueOnce(record("completed"));
    const streamEvents = vi
      .fn()
      .mockImplementationOnce(async ({ onEvent }) => {
        onEvent(event("10", "run.started"));
      })
      .mockRejectedValueOnce(new SseHttpError(409))
      .mockImplementationOnce(async ({ onEvent }) => {
        onEvent(event("10", "run.started"));
        onEvent(event("11", "workflow.progress", { stage: "search" }));
      });

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait: vi.fn().mockResolvedValue(undefined),
    });

    expect(streamEvents.mock.calls.map(([options]) => options.lastEventId)).toEqual([
      undefined,
      "10",
      undefined,
    ]);
    expect(h.state().events.map(({ id }) => id)).toEqual(["10", "11"]);
    expect(h.state().lastEventId).toBe("11");
  });

  it("uses the sixth and final stream as cursorless fallback after a fifth-stream HTTP failure", async () => {
    const h = harness();
    let gets = 0;
    let streams = 0;
    const getRun = vi.fn(async () =>
      record(++gets === 7 ? "completed" : "running"),
    );
    const streamEvents = vi.fn(async ({ onEvent }) => {
      streams += 1;
      if (streams === 1) onEvent(event("cursor-1", "run.started"));
      if (streams === 5) throw new SseHttpError(409);
    });

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait: vi.fn().mockResolvedValue(undefined),
    });

    expect(streamEvents).toHaveBeenCalledTimes(6);
    expect(streamEvents.mock.calls[4]?.[0].lastEventId).toBe("cursor-1");
    expect(streamEvents.mock.calls[5]?.[0].lastEventId).toBeUndefined();
  });

  it("does not open a seventh stream when the sixth cursor-bearing stream fails", async () => {
    const h = harness();
    const getRun = vi.fn().mockResolvedValue(record("running"));
    let streams = 0;
    const streamEvents = vi.fn(async ({ onEvent }) => {
      streams += 1;
      if (streams === 1) onEvent(event("cursor-1", "run.started"));
      if (streams === 6) throw new SseHttpError(409);
    });

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait: vi.fn().mockResolvedValue(undefined),
    });

    expect(streamEvents).toHaveBeenCalledTimes(6);
    expect(streamEvents.mock.calls[5]?.[0].lastEventId).toBe("cursor-1");
  });

  it.each([
    ["SSE", new SseHttpError(404)],
    ["GET", new ApiHttpError(404, "secret detail")],
  ])("handles %s 404 once", async (source, failure) => {
    const h = harness();
    const onNotFound = vi.fn();
    const getRun =
      source === "GET"
        ? vi.fn().mockRejectedValue(failure)
        : vi.fn().mockResolvedValue(record("running"));
    const streamEvents =
      source === "SSE" ? vi.fn().mockRejectedValue(failure) : vi.fn();
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun,
      streamEvents,
      wait: vi.fn(),
      onNotFound,
    });
    expect(onNotFound).toHaveBeenCalledOnce();
    expect(onNotFound).toHaveBeenCalledWith(RUN_ID);
    expect(h.state().connection).toBe("not_found");
    expect(h.state().error).not.toContain("secret");
    expect(streamEvents).toHaveBeenCalledTimes(source === "SSE" ? 1 : 0);
  });

  it.each([
    [new SseProtocolError(), "protocol_error"],
    [new ApiProtocolError(), "protocol_error"],
    [new SseNetworkError(), "disconnected"],
    [new ApiNetworkError(), "disconnected"],
    [new SseHttpError(503), "disconnected"],
    [new ApiHttpError(503, "private backend detail"), "disconnected"],
  ] as const)("classifies %s as %s with a safe message", async (failure, connection) => {
    const h = harness();
    const isApiFailure = failure instanceof ApiProtocolError || failure instanceof ApiNetworkError || failure instanceof ApiHttpError;
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun: isApiFailure
        ? vi.fn().mockRejectedValue(failure)
        : vi.fn().mockResolvedValueOnce(record("running")).mockResolvedValueOnce(record("completed")),
      streamEvents: isApiFailure ? vi.fn() : vi.fn().mockRejectedValue(failure),
      wait: vi.fn(),
    });
    const matching = h.actions.find(
      (action) => action.type === "connection" && action.connection === connection,
    );
    expect(matching).toBeDefined();
    if (matching?.type === "connection") {
      expect(matching.error).not.toContain("private backend detail");
      expect(matching.error).toMatch(/[\u4e00-\u9fff]/);
    }
    expect(h.state().observedStatus).not.toBe("failed");
  });

  it("silently exits on abort during a stream and during a wait", async () => {
    for (const phase of ["stream", "wait"] as const) {
      const h = harness();
      const controller = new AbortController();
      const streamEvents = vi.fn(async () => {
        if (phase === "stream") {
          controller.abort();
          throw new DOMException("Aborted", "AbortError");
        }
      });
      const wait = vi.fn(async () => {
        controller.abort();
        throw new DOMException("Aborted", "AbortError");
      });
      await observeResearchRun({
        runId: RUN_ID,
        generation: GENERATION,
        signal: controller.signal,
        dispatch: h.dispatch,
        getRun: vi.fn().mockResolvedValue(record("running")),
        streamEvents,
        wait,
      });
      const errors = h.actions.filter(
        (action) =>
          action.type === "connection" &&
          ["disconnected", "protocol_error", "not_found"].includes(
            action.connection,
          ),
      );
      expect(errors).toHaveLength(phase === "stream" ? 0 : 1);
      if (phase === "stream") expect(wait).not.toHaveBeenCalled();
    }
  });

  it("cannot miss a parent abort triggered synchronously by connecting dispatch", async () => {
    const h = harness();
    const controller = new AbortController();
    const dispatch = (action: RunAction) => {
      h.dispatch(action);
      if (action.type === "connection" && action.connection === "connecting") {
        controller.abort();
      }
    };
    const streamEvents = vi.fn(
      ({ signal }): Promise<void> =>
        signal?.aborted
          ? Promise.reject(new DOMException("Aborted", "AbortError"))
          : new Promise(() => undefined),
    );
    const wait = vi.fn();

    const observation = observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: controller.signal,
      dispatch,
      getRun: vi.fn().mockResolvedValue(record("running")),
      streamEvents,
      wait,
    });
    const timeout = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("observer missed synchronous abort")), 50);
    });

    await expect(Promise.race([observation, timeout])).resolves.toBeUndefined();
    if (streamEvents.mock.calls.length > 0) {
      expect(streamEvents.mock.calls[0]?.[0].signal?.aborted).toBe(true);
    }
    expect(wait).not.toHaveBeenCalled();
    expect(
      h.actions.some(
        (action) =>
          action.type === "connection" &&
          ["disconnected", "protocol_error", "not_found"].includes(
            action.connection,
          ),
      ),
    ).toBe(false);
  });

  it("does nothing when the parent signal is already aborted", async () => {
    const h = harness();
    const controller = new AbortController();
    controller.abort();
    const getRun = vi.fn();
    const streamEvents = vi.fn();

    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: controller.signal,
      dispatch: h.dispatch,
      initialRecord: record("running"),
      getRun,
      streamEvents,
    });

    expect(h.actions).toEqual([]);
    expect(getRun).not.toHaveBeenCalled();
    expect(streamEvents).not.toHaveBeenCalled();
  });

  it("never opens concurrent stream attempts", async () => {
    const h = harness();
    let active = 0;
    let maximum = 0;
    const streamEvents = vi.fn(async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await Promise.resolve();
      active -= 1;
    });
    let gets = 0;
    await observeResearchRun({
      runId: RUN_ID,
      generation: GENERATION,
      signal: new AbortController().signal,
      dispatch: h.dispatch,
      getRun: vi.fn(async () => record(++gets >= 3 ? "completed" : "running")),
      streamEvents,
      wait: vi.fn().mockResolvedValue(undefined),
    });
    expect(maximum).toBe(1);
  });
});

describe("abortableWait", () => {
  it("clears its timer and rejects AbortError when aborted", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const pending = abortableWait(1_000, controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(vi.getTimerCount()).toBe(0);
    vi.useRealTimers();
  });
});
