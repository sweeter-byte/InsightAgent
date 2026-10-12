import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendDirectory = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
const nextBinary = path.join(
  frontendDirectory,
  "node_modules",
  "next",
  "dist",
  "bin",
  "next",
);

const delay = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function withTimeout(promise, milliseconds, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function listen(server, port = 0) {
  server.listen(port, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  return address.port;
}

async function reservePort() {
  const server = http.createServer();
  const port = await listen(server);
  await new Promise((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  return port;
}

function createControlledBackend() {
  let abortFirstRequest;
  const firstRequestAborted = new Promise((resolve) => {
    abortFirstRequest = resolve;
  });
  const lastEventIds = [];
  let eventRequestCount = 0;

  const server = http.createServer((request, response) => {
    const pathname = new URL(request.url ?? "/", "http://127.0.0.1")
      .pathname;

    if (pathname === "/health") {
      response.writeHead(200, { "content-type": "application/json" });
      response.end('{"status":"ok"}');
      return;
    }

    if (pathname !== "/v1/research/runs/proxy-run/events") {
      response.writeHead(404);
      response.end();
      return;
    }

    eventRequestCount += 1;
    const requestNumber = eventRequestCount;
    lastEventIds.push(request.headers["last-event-id"]);
    response.socket?.setNoDelay(true);
    response.writeHead(200, {
      "cache-control": "no-cache",
      connection: "keep-alive",
      "content-type": "text/event-stream; charset=utf-8",
    });
    response.flushHeaders();
    response.write('id: 1-0\nevent: progress\ndata: {"step":1}\n\n');

    const terminalTimer = setTimeout(() => {
      response.end(
        'id: 2-0\nevent: completed\ndata: {"status":"completed"}\n\n',
      );
    }, 750);

    response.once("close", () => {
      clearTimeout(terminalTimer);
      if (requestNumber === 1 && !response.writableEnded) {
        abortFirstRequest();
      }
    });
  });

  return { firstRequestAborted, lastEventIds, server };
}

async function waitForNext(origin, child, logs) {
  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        `Next.js exited before becoming ready (${child.exitCode}).\n${logs.join("")}`,
      );
    }
    try {
      const response = await fetch(`${origin}/health`, {
        signal: AbortSignal.timeout(1_000),
      });
      if (response.ok) return;
    } catch {
      // Next.js is still starting.
    }
    await delay(200);
  }
  throw new Error(`Next.js did not become ready.\n${logs.join("")}`);
}

async function readFirstFrame(reader, decoder) {
  let text = "";
  while (!text.includes("\n\n")) {
    const { done, value } = await reader.read();
    assert.equal(done, false, "SSE ended before its first frame");
    text += decoder.decode(value, { stream: true });
  }
  return text;
}

async function readToEnd(reader, decoder) {
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) return text + decoder.decode();
    text += decoder.decode(value, { stream: true });
  }
}

async function stopChild(child, exitPromise) {
  if (child.exitCode !== null || child.signalCode !== null) return;

  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }

  try {
    await withTimeout(exitPromise, 5_000, "Next.js did not stop after SIGTERM");
  } catch {
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch {
      child.kill("SIGKILL");
    }
    await exitPromise;
  }
}

async function main() {
  const backend = createControlledBackend();
  let nextProcess;
  let nextExit;

  try {
    const backendPort = await listen(backend.server);
    const nextPort = await reservePort();
    const nextOrigin = `http://127.0.0.1:${nextPort}`;
    const logs = [];

    nextProcess = spawn(
      process.execPath,
      [nextBinary, "dev", "--webpack", "--hostname", "127.0.0.1", "--port", String(nextPort)],
      {
        cwd: frontendDirectory,
        detached: true,
        env: {
          ...process.env,
          INSIGHTAGENT_BACKEND_ORIGIN: `http://127.0.0.1:${backendPort}`,
          NEXT_TELEMETRY_DISABLED: "1",
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    nextExit = once(nextProcess, "exit");
    nextProcess.stdout.on("data", (chunk) => logs.push(chunk.toString()));
    nextProcess.stderr.on("data", (chunk) => logs.push(chunk.toString()));

    await waitForNext(nextOrigin, nextProcess, logs);

    const abortController = new AbortController();
    const abortResponse = await fetch(
      `${nextOrigin}/v1/research/runs/proxy-run/events?attempt=abort`,
      { signal: abortController.signal },
    );
    assert.match(
      abortResponse.headers.get("content-type") ?? "",
      /^text\/event-stream\b/i,
    );
    assert.match(abortResponse.headers.get("cache-control") ?? "", /no-cache/i);
    const abortReader = abortResponse.body?.getReader();
    assert.ok(abortReader, "abort response did not expose a readable body");
    const abortFirst = await readFirstFrame(abortReader, new TextDecoder());
    assert.match(abortFirst, /id: 1-0/);
    abortController.abort();
    await abortReader.cancel(abortController.signal.reason).catch(() => {});
    console.log("PASS headers: text/event-stream and no-cache");
    await withTimeout(
      backend.firstRequestAborted,
      1_000,
      "upstream connection remained open",
    );
    console.log("PASS abort: backend observed the proxied connection close");

    const startedAt = performance.now();
    const response = await fetch(
      `${nextOrigin}/v1/research/runs/proxy-run/events?attempt=reconnect`,
      { headers: { "Last-Event-ID": "1-0" } },
    );
    const reader = response.body?.getReader();
    assert.ok(reader, "reconnect response did not expose a readable body");
    const decoder = new TextDecoder();
    const first = await readFirstFrame(reader, decoder);
    const firstAt = performance.now();
    assert.match(first, /id: 1-0/);
    const remainder = await readToEnd(reader, decoder);
    const completedAt = performance.now();

    assert.ok(
      firstAt - startedAt < 600,
      `proxy buffered the first SSE frame (${Math.round(firstAt - startedAt)}ms)`,
    );
    assert.ok(
      completedAt - firstAt >= 400,
      "test backend did not preserve the delayed boundary",
    );
    assert.match(remainder, /id: 2-0/);
    assert.equal(backend.lastEventIds.at(-1), "1-0");
    console.log(
      `PASS streaming: first frame ${Math.round(firstAt - startedAt)}ms, completion ${Math.round(completedAt - firstAt)}ms later`,
    );
    console.log("PASS reconnect: Last-Event-ID: 1-0 reached the backend");
  } finally {
    if (nextProcess && nextExit) await stopChild(nextProcess, nextExit);
    backend.server.closeAllConnections();
    if (backend.server.listening) {
      await new Promise((resolve) => backend.server.close(resolve));
    }
  }
}

await main();
