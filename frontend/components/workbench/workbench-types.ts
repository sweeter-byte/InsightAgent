import type {
  MaterialImportResult,
  ReadinessResponse,
} from "@/lib/protocol/types";

export type AsyncPhase = "idle" | "submitting" | "succeeded" | "failed";

export type ConnectionState = {
  phase:
    | "checking"
    | "ready"
    | "not_ready"
    | "unreachable"
    | "protocol_error";
  healthAlive: boolean;
  readiness: ReadinessResponse | null;
  error: string | null;
};

export type SessionMaterial = MaterialImportResult & {
  fileName: string;
};

export type ConfirmedRun = {
  runId: string;
  threadId: string | null;
  status: string;
  source: "smart" | "explicit";
};

export type SynchronousAnswer = {
  intent: "direct" | "analyze";
  text: string;
};
