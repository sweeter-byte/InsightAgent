import type { z } from "zod";

import type {
  apiErrorSchema,
  componentReadinessSchema,
  createResearchRunRequestSchema,
  healthResponseSchema,
  materialImportResultSchema,
  queryRequestSchema,
  queryResponseSchema,
  readinessResponseSchema,
  runRecordSchema,
  runStatusSchema,
  runtimeEventSchema,
  sseEventSchema,
} from "./schemas";

export type ApiErrorBody = z.infer<typeof apiErrorSchema>;
export type CreateResearchRunRequest = z.infer<
  typeof createResearchRunRequestSchema
>;
export type QueryRequest = z.infer<typeof queryRequestSchema>;
export type QueryResponse = z.infer<typeof queryResponseSchema>;
export type MaterialImportResult = z.infer<typeof materialImportResultSchema>;
export type RunStatus = z.infer<typeof runStatusSchema>;
export type RunRecord = z.infer<typeof runRecordSchema>;
export type RuntimeEvent = z.infer<typeof runtimeEventSchema>;
export type SseEvent = z.infer<typeof sseEventSchema>;
export type HealthResponse = z.infer<typeof healthResponseSchema>;
export type ComponentReadiness = z.infer<typeof componentReadinessSchema>;
export type ReadinessResponse = z.infer<typeof readinessResponseSchema>;
