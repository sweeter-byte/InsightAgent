"use client";

import { useEffect, useReducer, useRef } from "react";

import {
  observeResearchRun,
  type ObserveResearchRunOptions,
} from "./run-observer";
import {
  initialRunState,
  runReducer,
  type ResearchRunState,
} from "./run-state";
import type { RunRecord } from "@/lib/protocol/types";

export type ResearchRunSelection = {
  runId: string;
  initialRecord?: RunRecord;
};

type ResearchRunObserver = (
  options: ObserveResearchRunOptions,
) => Promise<void>;

export type UseResearchRunOptions = {
  observe?: ResearchRunObserver;
  onNotFound?: (runId: string) => void;
};

function isAbort(error: unknown, signal: AbortSignal): boolean {
  return (
    signal.aborted ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError")
  );
}

export function useResearchRun(
  selection: ResearchRunSelection | null,
  options: UseResearchRunOptions = {},
): ResearchRunState {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const generationRef = useRef(0);
  const observe = options.observe ?? observeResearchRun;
  const onNotFound = options.onNotFound;
  const runId = selection?.runId ?? null;
  const initialRecord = selection?.initialRecord;

  useEffect(() => {
    const controller = new AbortController();
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    dispatch({ type: "select", runId, generation });

    if (runId !== null) {
      const reportUnexpectedFailure = (error: unknown) => {
        if (isAbort(error, controller.signal)) return;
        dispatch({
          type: "connection",
          runId,
          generation,
          connection: "disconnected",
          error: "运行连接出现异常，请稍后重试。",
        });
      };

      try {
        void observe({
          runId,
          generation,
          signal: controller.signal,
          dispatch,
          initialRecord,
          onNotFound,
        }).catch(reportUnexpectedFailure);
      } catch (error) {
        reportUnexpectedFailure(error);
      }
    }

    return () => controller.abort();
  }, [runId, initialRecord, observe, onNotFound]);

  return state;
}
