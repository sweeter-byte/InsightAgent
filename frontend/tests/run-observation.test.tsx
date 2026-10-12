import { useLayoutEffect } from "react";
import { act, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import type { ObserveResearchRunOptions } from "@/components/workbench/run-observer";
import { RESEARCH_RUN_STORAGE_KEY } from "@/components/workbench/run-selection";
import { useResearchRun } from "@/components/workbench/use-research-run";
import {
  completedRunFixture,
  queuedRunFixture,
} from "@/lib/protocol/fixtures";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";

import { mockServer } from "./mocks/server";

const runEndpoint = "http://localhost:3000/v1/research/runs";

function sseFrame(id: string, event: string, data: Record<string, unknown> = {}): string {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function finiteEventStream(chunks: string[]): Response {
  const encoder = new TextEncoder();
  return new HttpResponse(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

function controlledEventStream() {
  const encoder = new TextEncoder();
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
  let stopped = false;
  let resolveAborted!: () => void;
  const aborted = new Promise<void>((resolve) => {
    resolveAborted = resolve;
  });
  const markStopped = () => {
    if (stopped) return;
    stopped = true;
    resolveAborted();
  };
  const closeForAbort = () => {
    if (stopped) return;
    try {
      controller?.close();
    } catch {
      // The consumer may have canceled the stream in the same turn.
    }
    markStopped();
  };
  const stream = new ReadableStream<Uint8Array>({
    start(next) {
      controller = next;
    },
    cancel() {
      markStopped();
    },
  });
  return {
    response(signal: AbortSignal): Response {
      signal.addEventListener("abort", () => {
        closeForAbort();
      }, { once: true });
      controller?.enqueue(encoder.encode(": stream-open\n\n"));
      return new HttpResponse(stream, {
        headers: { "Content-Type": "text/event-stream" },
      });
    },
    observeClientSignal(signal: AbortSignal | null | undefined) {
      signal?.addEventListener("abort", closeForAbort, { once: true });
    },
    enqueue(frame: string): boolean {
      if (stopped || !controller) return false;
      controller.enqueue(encoder.encode(frame));
      return true;
    },
    close() {
      if (stopped || !controller) return;
      stopped = true;
      controller.close();
    },
    aborted,
    get stopped() {
      return stopped;
    },
  };
}

function observeEventFetchSignals(
  onSignal: (url: string, signal: AbortSignal | null | undefined) => void,
) {
  const realFetch = globalThis.fetch.bind(globalThis);
  return vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
    const url = input instanceof Request ? input.url : String(input);
    if (url.includes("/v1/research/runs/") && url.endsWith("/events")) {
      onSignal(url, init?.signal);
      // MSW's Node interceptor asserts internally if an open mock socket is
      // completed after Undici has already processed the abort. The real
      // application signal is observed above; a non-aborting transport signal
      // lets the controlled stream close cleanly in the same abort turn.
      const transport = new AbortController();
      init?.signal?.addEventListener("abort", () => {
        window.setTimeout(() => transport.abort(), 0);
      }, { once: true });
      return realFetch(input, { ...init, signal: transport.signal });
    }
    return realFetch(input, init);
  });
}

async function renderReadyWorkbench() {
  const user = userEvent.setup();
  const result = render(<InsightWorkbench />);
  await screen.findByText("后端已就绪");
  return { user, ...result };
}

async function setQuestion(user: ReturnType<typeof userEvent.setup>, value: string) {
  const input = screen.getByLabelText("研究问题");
  await user.clear(input);
  await user.type(input, value);
}

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

beforeEach(() => sessionStorage.clear());
afterEach(() => vi.restoreAllMocks());

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
    act(() => {
      options.dispatch({
        type: "connection",
        runId: "run-1",
        generation: options.generation,
        connection: "connected",
      });
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

  it("uses the latest inline not-found callback without restarting observation", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const firstCallback = vi.fn();
    const latestCallback = vi.fn();
    const callbacks = [firstCallback, latestCallback];
    const { result, rerender, unmount } = renderHook(
      ({ callbackIndex }: { callbackIndex: number }) =>
        useResearchRun(
          { runId: "run-1" },
          {
            observe,
            onNotFound: (runId) => callbacks[callbackIndex](runId),
          },
        ),
      { initialProps: { callbackIndex: 0 } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const observation = observe.mock.calls[0][0];
    const generation = observation.generation;

    act(() => {
      observation.dispatch({
        type: "connection",
        runId: "run-1",
        generation,
        connection: "connected",
      });
    });
    expect(result.current.connection).toBe("connected");
    expect(observe).toHaveBeenCalledTimes(1);
    expect(observation.signal.aborted).toBe(false);
    expect(result.current.generation).toBe(generation);

    rerender({ callbackIndex: 1 });
    expect(observe).toHaveBeenCalledTimes(1);
    expect(observation.signal.aborted).toBe(false);
    expect(result.current.generation).toBe(generation);

    act(() => observation.onNotFound?.("run-1"));
    expect(firstCallback).not.toHaveBeenCalled();
    expect(latestCallback).toHaveBeenCalledWith("run-1");
    unmount();
  });

  it("updates the not-found callback before later layout effects run", async () => {
    let observation: ObserveResearchRunOptions | undefined;
    const observe = vi.fn<Observer>((options) => {
      observation = options;
      return new Promise(() => undefined);
    });
    const firstCallback = vi.fn();
    const latestCallback = vi.fn();
    const callbacks = [firstCallback, latestCallback];
    const { result, rerender, unmount } = renderHook(
      ({ callbackIndex, trigger }: { callbackIndex: number; trigger: boolean }) => {
        const state = useResearchRun(
          { runId: "run-1" },
          {
            observe,
            onNotFound: (runId) => callbacks[callbackIndex](runId),
          },
        );
        useLayoutEffect(() => {
          if (trigger) observation?.onNotFound?.("run-1");
        }, [callbackIndex, trigger]);
        return state;
      },
      { initialProps: { callbackIndex: 0, trigger: false } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const activeObservation = observe.mock.calls[0][0];
    const generation = activeObservation.generation;

    rerender({ callbackIndex: 1, trigger: true });

    expect(firstCallback).not.toHaveBeenCalled();
    expect(latestCallback).toHaveBeenCalledWith("run-1");
    expect(observe).toHaveBeenCalledTimes(1);
    expect(activeObservation.signal.aborted).toBe(false);
    expect(result.current.generation).toBe(generation);
    unmount();
  });

  it("does not forward a late not-found result from an obsolete selection", async () => {
    const observe = vi.fn<Observer>(() => new Promise(() => undefined));
    const onNotFound = vi.fn();
    const { rerender, unmount } = renderHook(
      ({ runId }) => useResearchRun({ runId }, { observe, onNotFound }),
      { initialProps: { runId: "run-old" } },
    );
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(1));
    const old = observe.mock.calls[0][0];

    rerender({ runId: "run-new" });
    await waitFor(() => expect(observe).toHaveBeenCalledTimes(2));
    const current = observe.mock.calls[1][0];
    act(() => old.onNotFound?.("run-old"));
    expect(onNotFound).not.toHaveBeenCalled();

    act(() => current.onNotFound?.("run-new"));
    expect(onNotFound).toHaveBeenCalledWith("run-new");
    unmount();
  });
});

describe("InsightWorkbench live run observation", () => {
  it("renders the complete explicit Research lifecycle from a real chunked SSE stream", async () => {
    const streamText = [
      sseFrame("1-0", "run.queued"),
      sseFrame("2-0", "run.started", { resume: false }),
      sseFrame("3-0", "workflow.progress", {
        stage: "planning",
        message: "Building the evidence plan",
      }),
      sseFrame("4-0", "run.completed", { final_output: "present" }),
    ].join("");
    const chunks = [
      streamText.slice(0, 17),
      streamText.slice(17, 73),
      streamText.slice(73, 151),
      streamText.slice(151),
    ];
    let getCalls = 0;
    mockServer.use(
      http.post(runEndpoint, () => HttpResponse.json(queuedRunFixture, { status: 202 })),
      http.get(`${runEndpoint}/demo-run/events`, () => finiteEventStream(chunks)),
      http.get(`${runEndpoint}/demo-run`, () => {
        getCalls += 1;
        return HttpResponse.json(completedRunFixture);
      }),
    );
    const { user } = await renderReadyWorkbench();
    await setQuestion(user, "Trace the complete lifecycle");

    await user.click(screen.getByRole("button", { name: "提交深度研究" }));

    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(screen.getByText("workflow.progress · planning")).toBeInTheDocument();
    expect(screen.getAllByText("Building the evidence plan").length).toBeGreaterThan(0);
    expect(screen.getByText("THREAD ID").nextElementSibling).toHaveTextContent("demo-thread");
    expect(screen.getAllByText("SSE 已连接").length).toBeGreaterThan(0);
    const eventList = document.querySelector(".event-list");
    expect(eventList).not.toBeNull();
    const items = within(eventList as HTMLElement).getAllByRole("listitem");
    expect(items).toHaveLength(4);
    expect(items.map((item) => item.textContent)).toEqual([
      expect.stringContaining("1-0run.queued"),
      expect.stringContaining("2-0run.started"),
      expect.stringContaining("3-0workflow.progress · planning"),
      expect.stringContaining("4-0run.completed"),
    ]);
    const report = screen.getByTestId("report-preview");
    expect(report).toHaveTextContent("# Research Report");
    expect(report.tagName).toBe("PRE");
    expect(screen.queryByRole("heading", { name: "Research Report" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Chapter 18|\[1\]|\d+%|Planner Task|Evidence Grader|Retriever/)).not.toBeInTheDocument();
    expect(getCalls).toBe(1);
  });

  it("restores a genuinely submitted run after unmount and remount", async () => {
    const firstStream = controlledEventStream();
    let firstClientSignalSeen = false;
    const fetchSpy = observeEventFetchSignals((url, signal) => {
      if (!firstClientSignalSeen && url.includes("restore-run")) {
        firstClientSignalSeen = true;
        firstStream.observeClientSignal(signal);
      }
    });
    const restoredRun = {
      ...queuedRunFixture,
      run_id: "restore-run",
      thread_id: "restore-thread",
    };
    const completedRestoredRun = {
      ...completedRunFixture,
      run_id: "restore-run",
      thread_id: "restore-thread",
    };
    let eventCalls = 0;
    let getCalls = 0;
    mockServer.use(
      http.post(runEndpoint, () => HttpResponse.json(restoredRun, { status: 202 })),
      http.get(`${runEndpoint}/restore-run/events`, ({ request }) => {
        eventCalls += 1;
        if (eventCalls === 1) return firstStream.response(request.signal);
        return finiteEventStream([
          sseFrame("restored-start", "run.started"),
          sseFrame("restored-complete", "run.completed"),
        ]);
      }),
      http.get(`${runEndpoint}/restore-run`, () => {
        getCalls += 1;
        return HttpResponse.json(getCalls === 1 ? restoredRun : completedRestoredRun);
      }),
    );
    const first = await renderReadyWorkbench();
    await setQuestion(first.user, "Persist this exact run");
    await first.user.click(screen.getByRole("button", { name: "提交深度研究" }));
    await waitFor(() => expect(eventCalls).toBe(1));
    await waitFor(() => expect(firstClientSignalSeen).toBe(true));
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("restore-run");

    first.unmount();
    await firstStream.aborted;
    render(<InsightWorkbench />);

    expect(await screen.findByText("重新挂载后任务文本不可用")).toBeInTheDocument();
    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(screen.getByText("THREAD ID").nextElementSibling).toHaveTextContent("restore-thread");
    expect(eventCalls).toBe(2);
    expect(getCalls).toBe(2);
    fetchSpy.mockRestore();
  });

  it("aborts the old live stream when two Research runs are submitted rapidly", async () => {
    const oldStream = controlledEventStream();
    let oldClientSignalSeen = false;
    const fetchSpy = observeEventFetchSignals((url, signal) => {
      if (url.includes("old-run")) {
        oldClientSignalSeen = true;
        oldStream.observeClientSignal(signal);
      }
    });
    const oldRun = { ...queuedRunFixture, run_id: "old-run", thread_id: "old-thread" };
    const newRun = { ...queuedRunFixture, run_id: "new-run", thread_id: "new-thread" };
    const completedNewRun = {
      ...completedRunFixture,
      run_id: "new-run",
      thread_id: "new-thread",
      final_output: "NEW REPORT",
    };
    let postCalls = 0;
    mockServer.use(
      http.post(runEndpoint, () => {
        postCalls += 1;
        return HttpResponse.json(postCalls === 1 ? oldRun : newRun, { status: 202 });
      }),
      http.get(`${runEndpoint}/old-run/events`, ({ request }) => oldStream.response(request.signal)),
      http.get(`${runEndpoint}/new-run/events`, () => finiteEventStream([
        sseFrame("new-start", "run.started"),
        sseFrame("new-complete", "run.completed"),
      ])),
      http.get(`${runEndpoint}/new-run`, () => HttpResponse.json(completedNewRun)),
    );
    const { user } = await renderReadyWorkbench();
    await setQuestion(user, "Old question");
    await user.click(screen.getByRole("button", { name: "提交深度研究" }));
    await screen.findAllByText("old-run");
    await waitFor(() => expect(oldClientSignalSeen).toBe(true));
    await setQuestion(user, "New question");
    await user.click(screen.getByRole("button", { name: "提交深度研究" }));

    await oldStream.aborted;
    expect(oldStream.enqueue(sseFrame("old-failed", "run.failed", { error: "OLD ERROR" }))).toBe(false);
    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("new-run").length).toBeGreaterThan(0);
    expect(screen.getByTestId("report-preview")).toHaveTextContent("NEW REPORT");
    expect(screen.queryByText("old-failed")).not.toBeInTheDocument();
    expect(screen.queryByText("OLD ERROR")).not.toBeInTheDocument();
    fetchSpy.mockRestore();
  });

  it("aborts an open real SSE request when the workbench unmounts", async () => {
    const stream = controlledEventStream();
    let clientSignalSeen = false;
    const fetchSpy = observeEventFetchSignals((_url, signal) => {
      clientSignalSeen = true;
      stream.observeClientSignal(signal);
    });
    mockServer.use(
      http.post(runEndpoint, () => HttpResponse.json(queuedRunFixture, { status: 202 })),
      http.get(`${runEndpoint}/demo-run/events`, ({ request }) => stream.response(request.signal)),
    );
    const workbench = await renderReadyWorkbench();
    await workbench.user.click(screen.getByRole("button", { name: "提交深度研究" }));
    await screen.findAllByText("demo-run");
    await waitFor(() => expect(clientSignalSeen).toBe(true));

    workbench.unmount();

    await stream.aborted;
    expect(stream.stopped).toBe(true);
    fetchSpy.mockRestore();
  });
});
