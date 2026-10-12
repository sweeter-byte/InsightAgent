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
