import { describe, expect, it } from "vitest";

import {
  createSseParser,
  type ParsedSseFrame,
} from "@/lib/research-events/sse-parser";

function collectEvents() {
  const events: ParsedSseFrame[] = [];
  const parser = createSseParser((event) => events.push(event));

  return { events, parser };
}

describe("createSseParser", () => {
  it("waits for a complete blank line when fields and JSON span chunks", () => {
    const { events, parser } = collectEvents();

    for (const chunk of [
      "i",
      "d: 1",
      "-0\nev",
      "ent: run.started\ndata: {\"res",
      "ume\":false}\n",
    ]) {
      parser.feed(chunk);
      expect(events).toEqual([]);
    }

    parser.feed("\n");

    expect(events).toEqual([
      { id: "1-0", event: "run.started", data: '{"resume":false}' },
    ]);
  });

  it("handles CRLF split across chunks, comments, and multiple data lines", () => {
    const { events, parser } = collectEvents();

    for (const chunk of [
      ": keep-alive\r",
      "\nid: 2-0\r\nevent: workflow.prog",
      "ress\r\ndata: first\r\ndata: sec",
      "ond\r",
      "\n\r",
      "\n",
    ]) {
      parser.feed(chunk);
    }

    expect(events).toEqual([
      { id: "2-0", event: "workflow.progress", data: "first\nsecond" },
    ]);
  });

  it("defaults the event name and ignores unknown fields", () => {
    const { events, parser } = collectEvents();

    parser.feed("id:3-0\nunknown: discarded\ndata:no-space-after-colon\n\n");

    expect(events).toEqual([
      { id: "3-0", event: "message", data: "no-space-after-colon" },
    ]);
  });

  it("dispatches an explicitly empty data field", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: 4-0\ndata:\n\n");

    expect(events).toEqual([{ id: "4-0", event: "message", data: "" }]);
  });

  it("ignores an id field containing NUL", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: valid\nid: invalid\0id\ndata: payload\n\n");

    expect(events).toEqual([
      { id: "valid", event: "message", data: "payload" },
    ]);
  });

  it("keeps the latest event id for later frames without an id field", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: persistent\ndata: first\n\ndata: second\n\n");

    expect(events).toEqual([
      { id: "persistent", event: "message", data: "first" },
      { id: "persistent", event: "message", data: "second" },
    ]);
  });

  it("updates the latest event id from a frame without data", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: id-only\n\ndata: later\n\n");

    expect(events).toEqual([
      { id: "id-only", event: "message", data: "later" },
    ]);
  });

  it("does not leak an id-only frame's explicit-id metadata into the next data frame", () => {
    const received: Array<{
      event: ParsedSseFrame;
      metadata: { hasExplicitId: boolean };
    }> = [];
    const parser = createSseParser((event, metadata) => {
      received.push({ event, metadata });
    });

    parser.feed("id: cursor-only\n\ndata: later\n\n");

    expect(received).toEqual([
      {
        event: { id: "cursor-only", event: "message", data: "later" },
        metadata: { hasExplicitId: false },
      },
    ]);
  });

  it("clears the latest event id with an empty id field", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: old\ndata: before\n\nid:\n\ndata: after\n\n");

    expect(events).toEqual([
      { id: "old", event: "message", data: "before" },
      { id: "", event: "message", data: "after" },
    ]);
  });

  it("clears frame data before invoking a callback that throws", () => {
    const events: ParsedSseFrame[] = [];
    let shouldThrow = true;
    const parser = createSseParser((event) => {
      if (shouldThrow) {
        shouldThrow = false;
        throw new Error("consumer failed");
      }
      events.push(event);
    });

    expect(() => parser.feed("id: first\ndata: stale\n\n")).toThrow(
      "consumer failed",
    );
    parser.feed("id: second\ndata: fresh\n\n");

    expect(events).toEqual([
      { id: "second", event: "message", data: "fresh" },
    ]);
  });

  it("does not dispatch a frame without a data field", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: 5-0\nevent: heartbeat\n\n");

    expect(events).toEqual([]);
  });

  it("discards an unterminated frame and residual line on finish", () => {
    const { events, parser } = collectEvents();

    parser.feed("id: 6-0\nevent: run.finished\ndata: {\"ok\":true}");
    parser.finish();

    expect(events).toEqual([]);
  });
});
