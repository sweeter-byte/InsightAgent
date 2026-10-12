import { resolveBackendOrigin } from "@/lib/backend-origin";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  const { runId } = await context.params;
  const backendOrigin = resolveBackendOrigin(
    process.env.INSIGHTAGENT_BACKEND_ORIGIN,
  );
  const target = new URL(
    `/v1/research/runs/${encodeURIComponent(runId)}/events`,
    backendOrigin,
  );
  target.search = new URL(request.url).search;

  const upstreamController = new AbortController();
  const abortUpstream = () => upstreamController.abort(request.signal.reason);
  request.signal.addEventListener("abort", abortUpstream, { once: true });

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      cache: "no-store",
      headers: {
        accept: "text/event-stream",
        ...(request.headers.has("last-event-id")
          ? { "Last-Event-ID": request.headers.get("last-event-id") ?? "" }
          : {}),
      },
      signal: upstreamController.signal,
    });
  } catch (error) {
    request.signal.removeEventListener("abort", abortUpstream);
    throw error;
  }

  if (!upstream.body) {
    request.signal.removeEventListener("abort", abortUpstream);
    return new Response(null, {
      headers: upstream.headers,
      status: upstream.status,
      statusText: upstream.statusText,
    });
  }

  const reader = upstream.body.getReader();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          request.signal.removeEventListener("abort", abortUpstream);
          controller.close();
        } else {
          controller.enqueue(value);
        }
      } catch (error) {
        request.signal.removeEventListener("abort", abortUpstream);
        if (upstreamController.signal.aborted) controller.close();
        else controller.error(error);
      }
    },
    async cancel(reason) {
      request.signal.removeEventListener("abort", abortUpstream);
      upstreamController.abort(reason);
      await reader.cancel(reason).catch(() => {});
    },
  });

  return new Response(body, {
    headers: upstream.headers,
    status: upstream.status,
    statusText: upstream.statusText,
  });
}
