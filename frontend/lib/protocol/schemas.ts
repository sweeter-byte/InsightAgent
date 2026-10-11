import { z } from "zod";

// Backend identifiers are plain strings; do not add client-only format rules.
const identifierSchema = z.string();
const utcDateTimeSchema = z.iso.datetime({ offset: true });

export const queryRequestSchema = z
  .object({
    query: z.string().trim().min(1),
  })
  .strict();

export const createResearchRunRequestSchema = queryRequestSchema;

export const apiErrorSchema = z
  .object({
    detail: z.string(),
  })
  .strict();

export const queryResponseSchema = z
  .object({
    request_id: identifierSchema,
    intent: z.enum(["direct", "analyze", "research"]),
    status: z.enum(["completed", "queued"]),
    answer: z.string().nullable(),
    run_id: identifierSchema.nullable(),
  })
  .strict();

export const materialImportResultSchema = z
  .object({
    material_id: identifierSchema,
    document_ids: z.array(identifierSchema),
    status: z.literal("indexed"),
    chunk_count: z.number().int(),
    deduplicated: z.boolean(),
  })
  .strict();

export const runStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
  "timed_out",
  "interrupted",
]);

export const runRecordSchema = z
  .object({
    run_id: identifierSchema,
    thread_id: identifierSchema,
    status: runStatusSchema,
    created_at: utcDateTimeSchema,
    updated_at: utcDateTimeSchema,
    error: z.string().nullable(),
    final_output: z.string().nullable(),
  })
  .strict();

/** Internal public event model. SSE transmits the separate envelope below. */
export const runtimeEventSchema = z
  .object({
    event_id: identifierSchema.nullable(),
    run_id: identifierSchema,
    type: identifierSchema,
    created_at: utcDateTimeSchema,
    data: z.record(z.string(), z.unknown()),
  })
  .strict();

export const sseEventSchema = z
  .object({
    id: identifierSchema,
    event: identifierSchema,
    data: z.record(z.string(), z.unknown()),
  })
  .strict();

export const healthResponseSchema = z
  .object({
    status: z.literal("ok"),
  })
  .strict();

export const componentReadinessSchema = z
  .object({
    status: z.enum(["ready", "unavailable", "available", "unconfigured"]),
    detail: z.string().nullable(),
  })
  .strict();

export const readinessResponseSchema = z
  .object({
    status: z.enum(["ready", "not_ready"]),
    required: z.record(z.string(), componentReadinessSchema),
    optional: z.record(z.string(), componentReadinessSchema),
  })
  .strict();
