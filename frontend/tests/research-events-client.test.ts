import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SseHttpError,
  SseNetworkError,
  SseProtocolError,
  streamResearchEvents,
} from "@/lib/research-events/client";

const encoder = new TextEncoder();
const appBase = "http://localhost:3000";

function eventStream(
  chunks: string[],
  options: { status?: number; contentType?: string } = {},
): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    }),
    {
      status: options.status ?? 200,
      headers: {
        "Content-Type": options.contentType ?? "text/event-stream",
      },
    },
  );
}

function fetchReturning(response: Response) {
  return vi.fn<typeof fetch>().mockResolvedValue(response);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("streamResearchEvents", () => {
  it("decodes a frame split across byte chunks and dispatches the validated envelope", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const bytes = encoder.encode(
          'id: event-1\nevent: workflow.progress\ndata: {"step":"检索"}\n\n',
        );
        controller.enqueue(bytes.slice(0, 9));
        controller.enqueue(bytes.slice(9, 41));
        controller.enqueue(bytes.slice(41, 44));
        controller.enqueue(bytes.slice(44));
        controller.close();
      },
    });
    const onEvent = vi.fn();

    await streamResearchEvents({
      runId: "run-1",
      onEvent,
      fetchImpl: fetchReturning(
        new Response(stream, {
          headers: { "Content-Type": "text/event-stream; charset=utf-8" },
        }),
      ),
    });

    expect(onEvent).toHaveBeenCalledOnce();
    expect(onEvent).toHaveBeenCalledWith({
      id: "event-1",
      event: "workflow.progress",
      data: { step: "检索" },
    });
  });

  it("uses an encoded same-origin URL and the complete streaming request contract", async () => {
    const fetchImpl = fetchReturning(eventStream([]));
    const controller = new AbortController();

    await streamResearchEvents({
      runId: "run/with spaces?and#symbols",
      lastEventId: "event-7",
      signal: controller.signal,
      onEvent: vi.fn(),
      fetchImpl,
    });

    expect(fetchImpl).toHaveBeenCalledWith(
      `${appBase}/v1/research/runs/run%2Fwith%20spaces%3Fand%23symbols/events`,
      {
        method: "GET",
        headers: {
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          "Last-Event-ID": "event-7",
        },
        cache: "no-store",
        signal: controller.signal,
      },
    );
  });

  it.each([undefined, ""])(
    "omits Last-Event-ID when its value is %s",
    async (lastEventId) => {
      const fetchImpl = fetchReturning(eventStream([]));

      await streamResearchEvents({
        runId: "run-1",
        lastEventId,
        onEvent: vi.fn(),
        fetchImpl,
      });

      expect(fetchImpl).toHaveBeenCalledWith(
        `${appBase}/v1/research/runs/run-1/events`,
        expect.objectContaining({
          headers: {
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
          },
        }),
      );
    },
  );

  it("keeps the route relative outside the browser", async () => {
    vi.stubGlobal("window", undefined);
    const fetchImpl = fetchReturning(eventStream([]));

    await streamResearchEvents({
      runId: "server run",
      onEvent: vi.fn(),
      fetchImpl,
    });

    expect(fetchImpl.mock.calls[0]?.[0]).toBe(
      "/v1/research/runs/server%20run/events",
    );
  });

  it("joins multiple data lines before parsing JSON", async () => {
    const onEvent = vi.fn();

    await streamResearchEvents({
      runId: "run-1",
      onEvent,
      fetchImpl: fetchReturning(
        eventStream([
          'id: event-2\nevent: source.found\ndata: {"source":\ndata: "manual"}\n\n',
        ]),
      ),
    });

    expect(onEvent).toHaveBeenCalledWith({
      id: "event-2",
      event: "source.found",
      data: { source: "manual" },
    });
  });

  it.each([404, 500])("classifies HTTP %i as SseHttpError", async (status) => {
    const error = await streamResearchEvents({
      runId: "run-1",
      onEvent: vi.fn(),
      fetchImpl: fetchReturning(new Response(null, { status })),
    }).catch((caught) => caught);

    expect(error).toBeInstanceOf(SseHttpError);
    expect(error).toMatchObject({ name: "SseHttpError", status });
  });

  it("rejects an unexpected content type regardless of a successful status", async () => {
    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: vi.fn(),
        fetchImpl: fetchReturning(
          eventStream([], { contentType: "application/json" }),
        ),
      }),
    ).rejects.toBeInstanceOf(SseProtocolError);
  });

  it("accepts a case-insensitive event-stream content type with parameters", async () => {
    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: vi.fn(),
        fetchImpl: fetchReturning(
          eventStream([], { contentType: "Text/Event-Stream; Charset=UTF-8" }),
        ),
      }),
    ).resolves.toBeUndefined();
  });

  it("rejects a successful response without a body", async () => {
    const response = {
      ok: true,
      status: 200,
      headers: new Headers({ "Content-Type": "text/event-stream" }),
      body: null,
    } as Response;

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: vi.fn(),
        fetchImpl: fetchReturning(response),
      }),
    ).rejects.toBeInstanceOf(SseProtocolError);
  });

  it.each([
    ["malformed JSON", "id: event-3\ndata: {oops}\n\n"],
    ["array data", "id: event-3\ndata: []\n\n"],
    ["null data", "id: event-3\ndata: null\n\n"],
    ["string data", 'id: event-3\ndata: "text"\n\n'],
    ["missing id", 'event: update\ndata: {"ok":true}\n\n'],
  ])("classifies %s as a protocol error without dispatch", async (_name, frame) => {
    const onEvent = vi.fn();

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent,
        fetchImpl: fetchReturning(eventStream([frame])),
      }),
    ).rejects.toBeInstanceOf(SseProtocolError);
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("rejects a data frame that only inherits the preceding frame's id", async () => {
    const onEvent = vi.fn();

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent,
        fetchImpl: fetchReturning(
          eventStream([
            'id: first\nevent: update\ndata: {"sequence":1}\n\n' +
              'event: update\ndata: {"sequence":2}\n\n',
          ]),
        ),
      }),
    ).rejects.toBeInstanceOf(SseProtocolError);
    expect(onEvent).toHaveBeenCalledOnce();
    expect(onEvent).toHaveBeenCalledWith({
      id: "first",
      event: "update",
      data: { sequence: 1 },
    });
  });

  it("dispatches a later frame that explicitly repeats the preceding id", async () => {
    const onEvent = vi.fn();

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent,
        fetchImpl: fetchReturning(
          eventStream([
            'id: repeated\nevent: update\ndata: {"sequence":1}\n\n' +
              'id: repeated\nevent: update\ndata: {"sequence":2}\n\n',
          ]),
        ),
      }),
    ).resolves.toBeUndefined();
    expect(onEvent).toHaveBeenCalledTimes(2);
    expect(onEvent).toHaveBeenNthCalledWith(2, {
      id: "repeated",
      event: "update",
      data: { sequence: 2 },
    });
  });

  it("classifies a fetch rejection as SseNetworkError", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockRejectedValue(new TypeError("connection failed"));

    await expect(
      streamResearchEvents({ runId: "run-1", onEvent: vi.fn(), fetchImpl }),
    ).rejects.toBeInstanceOf(SseNetworkError);
  });

  it("classifies a reader failure as SseNetworkError", async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(new Error("socket closed"));
      },
    });

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: vi.fn(),
        fetchImpl: fetchReturning(
          new Response(stream, {
            headers: { "Content-Type": "text/event-stream" },
          }),
        ),
      }),
    ).rejects.toBeInstanceOf(SseNetworkError);
  });

  it("preserves AbortError from fetch", async () => {
    const abortError = new DOMException("cancelled", "AbortError");
    const fetchImpl = vi.fn<typeof fetch>().mockRejectedValue(abortError);

    await expect(
      streamResearchEvents({ runId: "run-1", onEvent: vi.fn(), fetchImpl }),
    ).rejects.toBe(abortError);
  });

  it("preserves AbortError from the reader", async () => {
    const abortError = new DOMException("cancelled", "AbortError");
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(abortError);
      },
    });

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: vi.fn(),
        fetchImpl: fetchReturning(
          new Response(stream, {
            headers: { "Content-Type": "text/event-stream" },
          }),
        ),
      }),
    ).rejects.toBe(abortError);
  });

  it("propagates consumer errors and cancels the reader", async () => {
    const consumerError = new Error("consumer failed");
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode('id: event-4\ndata: {"ok":true}\n\n'),
        );
      },
      cancel,
    });

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent: () => {
          throw consumerError;
        },
        fetchImpl: fetchReturning(
          new Response(stream, {
            headers: { "Content-Type": "text/event-stream" },
          }),
        ),
      }),
    ).rejects.toBe(consumerError);
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("resolves at EOF and discards an unterminated frame", async () => {
    const onEvent = vi.fn();

    await expect(
      streamResearchEvents({
        runId: "run-1",
        onEvent,
        fetchImpl: fetchReturning(
          eventStream(['id: partial\ndata: {"ignored":true}']),
        ),
      }),
    ).resolves.toBeUndefined();
    expect(onEvent).not.toHaveBeenCalled();
  });
});
