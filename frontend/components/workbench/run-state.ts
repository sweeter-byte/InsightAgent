import type { RunRecord, RunStatus, SseEvent } from "@/lib/protocol/types";

export type RunConnectionStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "protocol_error"
  | "not_found";

export type ResearchRunState = {
  runId: string | null;
  generation: number;
  record: RunRecord | null;
  observedStatus: RunStatus | null;
  connection: RunConnectionStatus;
  events: readonly SseEvent[];
  seenEventIds: ReadonlySet<string>;
  lastEventId: string | null;
  latestProgress: SseEvent["data"] | null;
  error: string | null;
};

export type RunAction =
  | { type: "select"; runId: string | null; generation: number }
  | { type: "record"; runId: string; generation: number; record: RunRecord }
  | { type: "event"; runId: string; generation: number; event: SseEvent }
  | {
      type: "connection";
      runId: string;
      generation: number;
      connection: RunConnectionStatus;
      error?: string | null;
    };

export const initialRunState: ResearchRunState = {
  runId: null,
  generation: 0,
  record: null,
  observedStatus: null,
  connection: "idle",
  events: [],
  seenEventIds: new Set(),
  lastEventId: null,
  latestProgress: null,
  error: null,
};

export const terminalRunStatuses = [
  "completed",
  "failed",
  "timed_out",
  "interrupted",
] as const satisfies readonly RunStatus[];

const terminalStatusSet: ReadonlySet<RunStatus> = new Set(terminalRunStatuses);

const eventStatuses: Readonly<Partial<Record<string, RunStatus>>> = {
  "run.queued": "queued",
  "run.started": "running",
  "run.completed": "completed",
  "run.failed": "failed",
  "run.timed_out": "timed_out",
  "run.interrupted": "interrupted",
};

export function isTerminalRunStatus(status: RunStatus | null): boolean {
  return status !== null && terminalStatusSet.has(status);
}

export function isTerminalEvent(event: SseEvent): boolean {
  const status = eventStatuses[event.event];
  return status !== undefined && isTerminalRunStatus(status);
}

export function selectRun(
  _state: ResearchRunState,
  runId: string | null,
  generation: number,
): ResearchRunState {
  return {
    ...initialRunState,
    runId,
    generation,
    events: [],
    seenEventIds: new Set(),
  };
}

function statusAfterEvent(
  current: RunStatus | null,
  eventName: string,
): RunStatus | null {
  const next = eventStatuses[eventName];
  if (next === undefined) return current;

  if (next === "queued" && (current === "running" || isTerminalRunStatus(current))) {
    return current;
  }
  if (next === "running" && isTerminalRunStatus(current)) return current;

  return next;
}

export function runReducer(
  state: ResearchRunState,
  action: RunAction,
): ResearchRunState {
  if (action.type === "select") {
    return selectRun(state, action.runId, action.generation);
  }
  if (action.runId !== state.runId || action.generation !== state.generation) {
    return state;
  }

  switch (action.type) {
    case "record":
      if (action.record.run_id !== action.runId) return state;
      return {
        ...state,
        record: action.record,
        observedStatus: action.record.status,
        error: action.record.error,
      };

    case "event": {
      if (state.seenEventIds.has(action.event.id)) return state;

      const seenEventIds = new Set(state.seenEventIds);
      seenEventIds.add(action.event.id);
      return {
        ...state,
        observedStatus: statusAfterEvent(
          state.observedStatus,
          action.event.event,
        ),
        events: [...state.events, action.event],
        seenEventIds,
        lastEventId: action.event.id,
        latestProgress:
          action.event.event === "workflow.progress"
            ? action.event.data
            : state.latestProgress,
      };
    }

    case "connection":
      return {
        ...state,
        record: action.connection === "not_found" ? null : state.record,
        connection: action.connection,
        error: action.error ?? null,
      };
  }
}
