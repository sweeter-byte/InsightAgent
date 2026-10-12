import { apiRoutes } from "../api-client";
import { sseEventSchema } from "../protocol/schemas";
import type { SseEvent } from "../protocol/types";
import { createSseParser } from "./sse-parser";

export class SseHttpError extends Error {
  readonly name = "SseHttpError";

  constructor(readonly status: number) {
    super(`HTTP ${status}`);
  }
}

export class SseNetworkError extends Error {
  readonly name = "SseNetworkError";

  constructor() {
    super("Unable to read the research event stream");
  }
}

export class SseProtocolError extends Error {
  readonly name = "SseProtocolError";

  constructor() {
    super("The research event stream returned an unexpected response");
  }
}

type StreamResearchEventsOptions = {
  runId: string;
  lastEventId?: string;
  signal?: AbortSignal;
  onEvent: (event: SseEvent) => void;
  fetchImpl?: typeof fetch;
};

function requestUrl(path: string): string {
  if (typeof window === "undefined") return path;
  return new URL(path, window.location.origin).toString();
}

function isAbort(error: unknown, signal?: AbortSignal): boolean {
  return (
    signal?.aborted === true ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError")
  );
}

export async function streamResearchEvents({
  runId,
  lastEventId,
  signal,
  onEvent,
  fetchImpl = fetch,
}: StreamResearchEventsOptions): Promise<void> {
  const headers: Record<string, string> = {
    Accept: "text/event-stream",
    "Cache-Control": "no-cache",
  };
  if (lastEventId) headers["Last-Event-ID"] = lastEventId;

  let response: Response;
  try {
    response = await fetchImpl(requestUrl(apiRoutes.researchEvents(runId)), {
      method: "GET",
      headers,
      cache: "no-store",
      signal,
    });
  } catch (error) {
    if (isAbort(error, signal)) throw error;
    throw new SseNetworkError();
  }

  if (!response.ok) throw new SseHttpError(response.status);

  const contentType = response.headers.get("content-type");
  if (!contentType || !/^text\/event-stream\b/i.test(contentType)) {
    throw new SseProtocolError();
  }
  if (!response.body) throw new SseProtocolError();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser((frame) => {
    let event: SseEvent;
    try {
      if (!frame.id) throw new Error("SSE event is missing an id");
      event = sseEventSchema.parse({
        id: frame.id,
        event: frame.event,
        data: JSON.parse(frame.data),
      });
    } catch {
      throw new SseProtocolError();
    }
    onEvent(event);
  });

  try {
    while (true) {
      let result: ReadableStreamReadResult<Uint8Array>;
      try {
        result = await reader.read();
      } catch (error) {
        if (isAbort(error, signal)) throw error;
        throw new SseNetworkError();
      }

      if (result.done) {
        parser.finish();
        return;
      }
      parser.feed(decoder.decode(result.value, { stream: true }));
    }
  } catch (error) {
    try {
      await reader.cancel();
    } catch {
      // Preserve the original stream, protocol, or consumer error.
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
}
