import type {
  MaterialImportResult,
  ReadinessResponse,
  RunRecord,
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

export type ActiveResearchRun = {
  runId: string;
  source: "smart" | "explicit" | "restored";
  question: string | null;
  initialRecord?: RunRecord;
};

export type SynchronousAnswer = {
  intent: "direct" | "analyze";
  text: string;
};
