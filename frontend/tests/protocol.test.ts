import { describe, expect, it } from "vitest";

import {
  completedRunFixture,
  directQueryResponseFixture,
  healthResponseFixture,
  materialImportResultFixture,
  queryRequestFixture,
  readinessResponseFixture,
  researchQueryResponseFixture,
  runtimeEventFixture,
  sseEventFixtures,
  sseTextFixture,
} from "@/lib/protocol/fixtures";
import {
  healthResponseSchema,
  materialImportResultSchema,
  queryRequestSchema,
  queryResponseSchema,
  readinessResponseSchema,
  runRecordSchema,
  runtimeEventSchema,
  sseEventSchema,
} from "@/lib/protocol/schemas";

describe("Chapter 18 HTTP protocol", () => {
  it("serves fixed HTTP examples through the MSW contract harness", async () => {
    const response = await fetch("http://127.0.0.1:8000/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(queryRequestFixture),
    });

    expect(response.status).toBe(202);
    expect(queryResponseSchema.parse(await response.json())).toEqual(
      researchQueryResponseFixture,
    );
  });

  it("parses the accepted request and response fixtures", () => {
    expect(queryRequestSchema.parse(queryRequestFixture)).toEqual(queryRequestFixture);
    expect(queryResponseSchema.parse(directQueryResponseFixture)).toEqual(
      directQueryResponseFixture,
    );
    expect(queryResponseSchema.parse(researchQueryResponseFixture)).toEqual(
      researchQueryResponseFixture,
    );
    expect(materialImportResultSchema.parse(materialImportResultFixture)).toEqual(
      materialImportResultFixture,
    );
    expect(runRecordSchema.parse(completedRunFixture)).toEqual(completedRunFixture);
    expect(runtimeEventSchema.parse(runtimeEventFixture)).toEqual(runtimeEventFixture);
    expect(healthResponseSchema.parse(healthResponseFixture)).toEqual(
      healthResponseFixture,
    );
    expect(readinessResponseSchema.parse(readinessResponseFixture)).toEqual(
      readinessResponseFixture,
    );
  });

  it("keeps run_id and thread_id as separate required fields", () => {
    const run = runRecordSchema.parse(completedRunFixture);

    expect(run.run_id).toBe("demo-run");
    expect(run.thread_id).toBe("demo-thread");
    expect(run.run_id).not.toBe(run.thread_id);
  });

  it("rejects illegal run states and missing required fields", () => {
    expect(
      runRecordSchema.safeParse({ ...completedRunFixture, status: "paused" }).success,
    ).toBe(false);
    expect(
      runRecordSchema.safeParse({ ...completedRunFixture, thread_id: undefined }).success,
    ).toBe(false);
    expect(queryResponseSchema.safeParse({ intent: "direct" }).success).toBe(false);
  });

  it("normalizes request whitespace like the Pydantic validator", () => {
    expect(queryRequestSchema.parse({ query: "  explain checkpoints  " })).toEqual({
      query: "explain checkpoints",
    });
    expect(queryRequestSchema.safeParse({ query: "   " }).success).toBe(false);
  });
});

describe("runtime SSE boundary", () => {
  it("parses the event envelopes found in backend SSE tests", () => {
    for (const event of sseEventFixtures) {
      expect(sseEventSchema.parse(event)).toEqual(event);
    }
  });

  it("records that SSE data omits run_id and created_at", () => {
    expect(sseTextFixture).toContain("event: workflow.progress");
    expect(sseTextFixture).not.toContain("run_id");
    expect(sseTextFixture).not.toContain("created_at");
  });
});
