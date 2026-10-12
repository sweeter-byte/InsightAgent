import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ObserveResearchRunOptions } from "@/components/workbench/run-observer";
import { useResearchRun } from "@/components/workbench/use-research-run";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";

type Observer = (options: ObserveResearchRunOptions) => Promise<void>;

function record(runId: string, status: RunRecord["status"] = "running"): RunRecord {
  return {
    run_id: runId,
    thread_id: `thread-${runId}`,
    status,
    created_at: "2026-10-12T00:00:00Z",
    updated_at: "2026-10-12T00:01:00Z",
    error: null,
    final_output: null,
  };
}

function deferred() {
  let resolve!: () => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<void>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

describe("useResearchRun", () => {
  it("selects and observes a run with one positive generation", async () => {
    const pending = deferred();
    const observe = vi.fn<Observer>(() => pending.promise);
    const { result, unmount } = renderHook(() =>
      useResearchRun({ runId: "run-1" }, { observe }),
    );

    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const options = observe.mock.calls[0][0];
    expect(options.runId).toBe("run-1");
    expect(options.generation).toBeGreaterThan(0);
    expect(result.current).toMatchObject({
      runId: "run-1",
      generation: options.generation,
    });
    options.dispatch({
      type: "connection",
      runId: "run-1",
      generation: options.generation,
      connection: "connected",
    });
    await waitFor(() => expect(result.current.connection).toBe("connected"));

    unmount();
    pending.resolve();
  });

  it("aborts its observation on unmount", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const { unmount } = renderHook(() =>
      useResearchRun({ runId: "run-1" }, { observe }),
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const signal = observe.mock.calls[0][0].signal;

    unmount();

    expect(signal.aborted).toBe(true);
  });

  it("aborts the old run before observing a new selection", async () => {
    const starts: Array<{ runId: string; generation: number; priorAborted: boolean }> = [];
    let previousSignal: AbortSignal | undefined;
    const observe = vi.fn<Observer>((options) => {
      starts.push({
        runId: options.runId,
        generation: options.generation,
        priorAborted: previousSignal?.aborted ?? true,
      });
      previousSignal = options.signal;
      return new Promise(() => undefined);
    });
    const { result, rerender, unmount } = renderHook(
      ({ runId }) => useResearchRun({ runId }, { observe }),
      { initialProps: { runId: "run-old" } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const old = observe.mock.calls[0][0];

    rerender({ runId: "run-new" });

    await waitFor(() => expect(observe).toHaveBeenCalledTimes(2));
    const next = observe.mock.calls[1][0];
    expect(old.signal.aborted).toBe(true);
    expect(next.signal.aborted).toBe(false);
    expect(next.generation).not.toBe(old.generation);
    expect(starts).toEqual([
      { runId: "run-old", generation: old.generation, priorAborted: true },
      { runId: "run-new", generation: next.generation, priorAborted: true },
    ]);
    expect(result.current).toMatchObject({
      runId: "run-new",
      generation: next.generation,
    });
    unmount();
  });

  it("restarts the same run with a new generation when its seed changes", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const firstSeed = record("run-1", "queued");
    const secondSeed = record("run-1", "running");
    const { result, rerender, unmount } = renderHook(
      ({ initialRecord }) =>
        useResearchRun({ runId: "run-1", initialRecord }, { observe }),
      { initialProps: { initialRecord: firstSeed } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const first = observe.mock.calls[0][0];

    rerender({ initialRecord: secondSeed });

    await waitFor(() => expect(observe).toHaveBeenCalledTimes(2));
    const second = observe.mock.calls[1][0];
    expect(first.signal.aborted).toBe(true);
    expect(second.generation).not.toBe(first.generation);
    expect(second.initialRecord).toBe(secondSeed);
    expect(result.current.generation).toBe(second.generation);
    unmount();
  });

  it("aborts and clears state when the selection becomes null", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const { result, rerender } = renderHook(
      ({ selected }: { selected: boolean }) =>
        useResearchRun(selected ? { runId: "run-1" } : null, { observe }),
      { initialProps: { selected: true } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const first = observe.mock.calls[0][0];
    act(() => {
      first.dispatch({
        type: "record",
        runId: "run-1",
        generation: first.generation,
        record: record("run-1"),
      });
    });

    rerender({ selected: false });

    await waitFor(() => expect(result.current.runId).toBeNull());
    expect(first.signal.aborted).toBe(true);
    expect(result.current.record).toBeNull();
    expect(result.current.connection).toBe("idle");
    expect(result.current.generation).toBeGreaterThan(first.generation);
    expect(observe).toHaveBeenCalledTimes(1);
  });

  it("never overlaps live observations in Strict Mode", async () => {
    let active = 0;
    let starts = 0;
    const observe = vi.fn<Observer>(({ signal }) => {
      expect(active).toBe(0);
      active += 1;
      starts += 1;
      signal.addEventListener("abort", () => {
        active -= 1;
      }, { once: true });
      return new Promise(() => undefined);
    });
    const { unmount } = renderHook(
      () => useResearchRun({ runId: "run-1" }, { observe }),
      { reactStrictMode: true },
    );

    await waitFor(() => expect(starts).toBeGreaterThanOrEqual(2));
    expect(active).toBe(1);
    const generations = observe.mock.calls.map(([options]) => options.generation);
    expect(new Set(generations).size).toBe(generations.length);

    unmount();
    expect(active).toBe(0);
  });

  it("ignores stale actions after restarting the same run", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const firstSeed = record("run-1", "queued");
    const secondSeed = record("run-1", "running");
    const { result, rerender, unmount } = renderHook(
      ({ initialRecord }) =>
        useResearchRun({ runId: "run-1", initialRecord }, { observe }),
      { initialProps: { initialRecord: firstSeed } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const stale = observe.mock.calls[0][0];
    rerender({ initialRecord: secondSeed });
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(2));
    const current = observe.mock.calls[1][0];
    const staleEvent: SseEvent = {
      id: "stale-event",
      event: "workflow.progress",
      data: { stage: "stale" },
    };

    act(() => {
      current.dispatch({
        type: "record",
        runId: "run-1",
        generation: current.generation,
        record: secondSeed,
      });
      stale.dispatch({
        type: "record",
        runId: "run-1",
        generation: stale.generation,
        record: record("run-1", "failed"),
      });
      stale.dispatch({
        type: "event",
        runId: "run-1",
        generation: stale.generation,
        event: staleEvent,
      });
    });

    expect(stale.signal.aborted).toBe(true);
    expect(result.current.record).toBe(secondSeed);
    expect(result.current.events).toEqual([]);
    unmount();
  });

  it("turns an unexpected observer rejection into a safe disconnected state", async () => {
    const pending = deferred();
    const observe = vi.fn<Observer>(() => pending.promise);
    const { result } = renderHook(() =>
      useResearchRun({ runId: "run-1" }, { observe }),
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));

    await act(async () => {
      pending.reject(new Error("observer exploded"));
      await pending.promise.catch(() => undefined);
    });

    await waitFor(() => expect(result.current.connection).toBe("disconnected"));
    expect(result.current.error).toBeTruthy();
  });

  it("silently ignores an abort rejection", async () => {
    const pending = deferred();
    const observe = vi.fn<Observer>(() => pending.promise);
    const { result, rerender } = renderHook(
      ({ selected }: { selected: boolean }) =>
        useResearchRun(selected ? { runId: "run-1" } : null, { observe }),
      { initialProps: { selected: true } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));

    rerender({ selected: false });
    const abortError = new Error("Aborted");
    abortError.name = "AbortError";
    await act(async () => {
      pending.reject(abortError);
      await pending.promise.catch(() => undefined);
    });

    expect(result.current).toMatchObject({
      runId: null,
      connection: "idle",
      error: null,
    });
  });
});
