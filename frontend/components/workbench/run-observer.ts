import type { RunAction } from "./run-state";
import { isTerminalEvent, isTerminalRunStatus } from "./run-state";
import {
  api,
  ApiHttpError,
  ApiNetworkError,
  ApiProtocolError,
} from "@/lib/api-client";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";
import {
  SseHttpError,
  SseNetworkError,
  SseProtocolError,
  streamResearchEvents,
} from "@/lib/research-events/client";

export const reconnectDelays = [500, 1_000, 2_000, 4_000, 8_000] as const;

type GetRun = (runId: string, signal?: AbortSignal) => Promise<RunRecord>;

type StreamEvents = (options: {
  runId: string;
  lastEventId?: string;
  signal?: AbortSignal;
  onEvent: (event: SseEvent) => void;
}) => Promise<void>;

export type ObserveResearchRunOptions = {
  runId: string;
  generation: number;
  signal: AbortSignal;
  dispatch: (action: RunAction) => void;
  initialRecord?: RunRecord;
  getRun?: GetRun;
  streamEvents?: StreamEvents;
  wait?: (ms: number, signal: AbortSignal) => Promise<void>;
  onNotFound?: (runId: string) => void;
};

type UnscopedRunAction<Action = RunAction> = Action extends RunAction
  ? Omit<Action, "runId" | "generation">
  : never;

function abortError(): Error {
  const error = new Error("Aborted");
  error.name = "AbortError";
  return error;
}

function isAbort(error: unknown, signal: AbortSignal): boolean {
  return (
    signal.aborted ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError")
  );
}

export function abortableWait(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError());

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      reject(abortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function connectionFailure(error: unknown): {
  connection: "protocol_error" | "disconnected";
  message: string;
} {
  if (error instanceof SseProtocolError || error instanceof ApiProtocolError) {
    return {
      connection: "protocol_error",
      message: "后端返回了无法识别的运行数据，请检查前后端版本。",
    };
  }
  if (
    error instanceof SseNetworkError ||
    error instanceof ApiNetworkError ||
    error instanceof SseHttpError ||
    error instanceof ApiHttpError
  ) {
    return {
      connection: "disconnected",
      message: "运行连接暂时中断，请稍后重试。",
    };
  }
  return {
    connection: "disconnected",
    message: "运行连接出现异常，请稍后重试。",
  };
}

export async function observeResearchRun({
  runId,
  generation,
  signal,
  dispatch,
  initialRecord,
  getRun = api.getResearchRun,
  streamEvents = streamResearchEvents,
  wait = abortableWait,
  onNotFound,
}: ObserveResearchRunOptions): Promise<void> {
  if (signal.aborted) return;

  let notFoundReported = false;
  let cursor: string | undefined;
  const seenEventIds = new Set<string>();
  let fallbackUsed = false;
  let nextAttemptCursorless = false;

  const send = (action: UnscopedRunAction) => {
    dispatch({ ...action, runId, generation } as RunAction);
  };

  const reportNotFound = () => {
    if (notFoundReported) return;
    notFoundReported = true;
    send({
      type: "connection",
      connection: "not_found",
      error: "未找到该研究运行。",
    });
    onNotFound?.(runId);
  };

  const reportFailure = (error: unknown) => {
    const failure = connectionFailure(error);
    send({
      type: "connection",
      connection: failure.connection,
      error: failure.message,
    });
  };

  const fetchRecord = async (): Promise<RunRecord | null> => {
    try {
      const current = await getRun(runId, signal);
      if (signal.aborted) throw abortError();
      send({ type: "record", record: current });
      return current;
    } catch (error) {
      if (isAbort(error, signal)) throw error;
      if (error instanceof ApiHttpError && error.status === 404) {
        reportNotFound();
        return null;
      }
      reportFailure(error);
      return null;
    }
  };

  try {
    let current: RunRecord | null;
    if (initialRecord?.run_id === runId) {
      current = initialRecord;
      send({ type: "record", record: current });
    } else {
      current = await fetchRecord();
    }

    if (current === null || isTerminalRunStatus(current.status)) return;

    for (let attempt = 0; attempt <= reconnectDelays.length; attempt += 1) {
      if (signal.aborted) return;

      const cursorlessAttempt = nextAttemptCursorless;
      nextAttemptCursorless = false;
      const lastEventId = cursorlessAttempt ? undefined : cursor;
      send({
        type: "connection",
        connection: attempt === 0 ? "connecting" : "reconnecting",
      });

      let streamFailure: unknown = null;
      let terminalCandidate = false;
      let connected = false;
      const attemptController = new AbortController();
      const abortAttempt = () => attemptController.abort();
      signal.addEventListener("abort", abortAttempt, { once: true });
      try {
        await streamEvents({
          runId,
          lastEventId,
          signal: attemptController.signal,
          onEvent: (incoming) => {
            if (terminalCandidate) return;
            if (signal.aborted) throw abortError();
            if (!connected) {
              send({ type: "connection", connection: "connected" });
              connected = true;
            }
            if (seenEventIds.has(incoming.id)) return;
            send({ type: "event", event: incoming });
            seenEventIds.add(incoming.id);
            cursor = incoming.id;
            if (isTerminalEvent(incoming)) {
              terminalCandidate = true;
              attemptController.abort();
            }
          },
        });
      } catch (error) {
        if (signal.aborted) return;
        if (terminalCandidate && isAbort(error, attemptController.signal)) {
          // A terminal event closes this attempt so the GET below can establish
          // the authoritative status, error, and final output.
        } else if (isAbort(error, signal)) {
          return;
        } else if (error instanceof SseHttpError && error.status === 404) {
          reportNotFound();
          return;
        } else {
          streamFailure = error;
          reportFailure(error);
        }
      } finally {
        signal.removeEventListener("abort", abortAttempt);
      }

      if (signal.aborted) return;
      if (streamFailure === null && !terminalCandidate) {
        send({
          type: "connection",
          connection: "disconnected",
          error: "运行事件流已结束，正在重新连接。",
        });
      }

      current = await fetchRecord();
      if (current === null || isTerminalRunStatus(current.status)) return;
      if (streamFailure === null && terminalCandidate) {
        send({
          type: "connection",
          connection: "disconnected",
          error: "运行事件流已结束，正在重新连接。",
        });
      }

      if (
        streamFailure instanceof SseHttpError &&
        streamFailure.status !== 404 &&
        lastEventId !== undefined &&
        !fallbackUsed
      ) {
        fallbackUsed = true;
        nextAttemptCursorless = true;
      }

      if (attempt === reconnectDelays.length) return;
      await wait(reconnectDelays[attempt]!, signal);
    }
  } catch (error) {
    if (isAbort(error, signal)) return;
    reportFailure(error);
  }
}
