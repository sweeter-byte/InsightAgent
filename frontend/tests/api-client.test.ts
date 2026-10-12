import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import {
  ApiHttpError,
  ApiNetworkError,
  ApiProtocolError,
  api,
} from "@/lib/api-client";
import {
  healthResponseFixture,
  materialImportResultFixture,
  queuedRunFixture,
  readinessResponseFixture,
} from "@/lib/protocol/fixtures";

import { mockServer } from "./mocks/server";

const appBase = "http://localhost:3000";

describe("API client", () => {
  it("parses health and readiness through same-origin paths", async () => {
    mockServer.use(
      http.get(`${appBase}/health`, () => HttpResponse.json(healthResponseFixture)),
      http.get(`${appBase}/ready`, () => HttpResponse.json(readinessResponseFixture)),
    );

    await expect(api.health()).resolves.toEqual(healthResponseFixture);
    await expect(api.readiness()).resolves.toEqual(readinessResponseFixture);
  });

  it("parses a not-ready 503 body as readiness data", async () => {
    const notReady = {
      status: "not_ready",
      required: {
        redis: { status: "unavailable", detail: "redis unavailable" },
      },
      optional: {
        vision: { status: "unconfigured", detail: null },
      },
    } as const;
    mockServer.use(
      http.get(`${appBase}/ready`, () =>
        HttpResponse.json(notReady, { status: 503 }),
      ),
    );

    await expect(api.readiness()).resolves.toEqual(notReady);
  });

  it("sends only the normalized query to the unified query endpoint", async () => {
    let receivedJson: unknown;
    mockServer.use(
      http.post(`${appBase}/v1/query`, async ({ request }) => {
        receivedJson = await request.json();
        return HttpResponse.json({
          request_id: "request-direct-1",
          intent: "direct",
          status: "completed",
          answer: "direct answer",
          run_id: null,
        });
      }),
    );

    await expect(api.query("  explain checkpoints  ")).resolves.toMatchObject({
      intent: "direct",
      status: "completed",
    });
    expect(receivedJson).toEqual({ query: "explain checkpoints" });
  });

  it("uses FormData with exactly one file field and lets fetch set the boundary", async () => {
    let formKeys: string[] = [];
    let uploadedFile: FormDataEntryValue | null = null;
    let contentType: string | null = null;
    mockServer.use(
      http.post(`${appBase}/v1/materials`, async ({ request }) => {
        const form = await request.formData();
        formKeys = [...form.keys()];
        uploadedFile = form.get("file");
        contentType = request.headers.get("content-type");
        return HttpResponse.json(materialImportResultFixture, { status: 201 });
      }),
    );

    const file = new File(["# material"], "material.md", {
      type: "text/markdown",
    });
    await expect(api.uploadMaterial(file)).resolves.toEqual(
      materialImportResultFixture,
    );
    expect(formKeys).toEqual(["file"]);
    expect(uploadedFile).toMatchObject({
      size: 10,
      type: "text/markdown",
    });
    expect(contentType).toMatch(/^multipart\/form-data; boundary=/);
  });

  it("posts explicit research to the run endpoint and parses the RunRecord", async () => {
    let receivedJson: unknown;
    mockServer.use(
      http.post(`${appBase}/v1/research/runs`, async ({ request }) => {
        receivedJson = await request.json();
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
    );

    await expect(api.createResearchRun("  compare systems  ")).resolves.toEqual(
      queuedRunFixture,
    );
    expect(receivedJson).toEqual({ query: "compare systems" });
  });

  it("gets an authoritative research run without using the browser cache", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    mockServer.use(
      http.get(`${appBase}/v1/research/runs/demo-run`, () =>
        HttpResponse.json(queuedRunFixture),
      ),
    );

    try {
      await expect(api.getResearchRun("demo-run")).resolves.toEqual(
        queuedRunFixture,
      );
      expect(fetchSpy).toHaveBeenCalledWith(
        `${appBase}/v1/research/runs/demo-run`,
        expect.objectContaining({ cache: "no-store" }),
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it("URL-encodes the research run identifier at the route boundary", async () => {
    const runId = "run/with spaces?and#symbols";
    let requestedUrl: string | undefined;
    mockServer.use(
      http.get(`${appBase}/v1/research/runs/*`, ({ request }) => {
        requestedUrl = request.url;
        return HttpResponse.json(queuedRunFixture);
      }),
    );

    await expect(api.getResearchRun(runId)).resolves.toEqual(queuedRunFixture);
    expect(requestedUrl).toBe(
      `${appBase}/v1/research/runs/run%2Fwith%20spaces%3Fand%23symbols`,
    );
  });

  it("posts Resume with an encoded run identifier and parses the queued RunRecord", async () => {
    const runId = "run/with spaces?and#symbols";
    let requestedUrl: string | undefined;
    let requestedMethod: string | undefined;
    mockServer.use(
      http.post(`${appBase}/v1/research/runs/*/resume`, ({ request }) => {
        requestedUrl = request.url;
        requestedMethod = request.method;
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
    );

    await expect(api.resumeResearchRun(runId)).resolves.toEqual(queuedRunFixture);
    expect(requestedMethod).toBe("POST");
    expect(requestedUrl).toBe(
      `${appBase}/v1/research/runs/run%2Fwith%20spaces%3Fand%23symbols/resume`,
    );
  });

  it.each([404, 409])("preserves Resume HTTP %s as an ApiHttpError", async (status) => {
    mockServer.use(
      http.post(`${appBase}/v1/research/runs/demo-run/resume`, () =>
        HttpResponse.json({ detail: "resume rejected" }, { status }),
      ),
    );

    const error = await api.resumeResearchRun("demo-run").catch((caught) => caught);
    expect(error).toEqual(
      expect.objectContaining({
        name: "ApiHttpError",
        status,
        detail: "resume rejected",
      }),
    );
  });

  it("maps a Resume transport failure to ApiNetworkError", async () => {
    mockServer.use(
      http.post(`${appBase}/v1/research/runs/demo-run/resume`, () =>
        HttpResponse.error(),
      ),
    );

    await expect(api.resumeResearchRun("demo-run")).rejects.toBeInstanceOf(
      ApiNetworkError,
    );
  });

  it("preserves a 404 response as an ApiHttpError", async () => {
    mockServer.use(
      http.get(`${appBase}/v1/research/runs/missing-run`, () =>
        HttpResponse.json({ detail: "run not found" }, { status: 404 }),
      ),
    );

    const error = await api.getResearchRun("missing-run").catch((caught) => caught);

    expect(error).toBeInstanceOf(ApiHttpError);
    expect(error).toEqual(
      expect.objectContaining({
        name: "ApiHttpError",
        status: 404,
        detail: "run not found",
      }),
    );
  });

  it("rejects a malformed successful research run response", async () => {
    mockServer.use(
      http.get(`${appBase}/v1/research/runs/malformed-run`, () =>
        HttpResponse.json({ run_id: "malformed-run" }),
      ),
    );

    await expect(api.getResearchRun("malformed-run")).rejects.toBeInstanceOf(
      ApiProtocolError,
    );
  });

  it("preserves AbortError when fetching a research run is cancelled", async () => {
    const controller = new AbortController();
    controller.abort();

    await expect(
      api.getResearchRun("demo-run", controller.signal),
    ).rejects.toMatchObject({ name: "AbortError" });
  });

  it("distinguishes HTTP, network, and protocol failures", async () => {
    mockServer.use(
      http.post(`${appBase}/v1/query`, () =>
        HttpResponse.json(
          { detail: "unable to route query" },
          { status: 502 },
        ),
      ),
      http.get(`${appBase}/health`, () => HttpResponse.error()),
      http.post(`${appBase}/v1/research/runs`, () =>
        HttpResponse.json({ run_id: "missing-fields" }, { status: 202 }),
      ),
    );

    await expect(api.query("hello")).rejects.toEqual(
      expect.objectContaining({
        name: "ApiHttpError",
        status: 502,
        detail: "unable to route query",
      }),
    );
    await expect(api.health()).rejects.toBeInstanceOf(ApiNetworkError);
    await expect(api.createResearchRun("hello")).rejects.toBeInstanceOf(
      ApiProtocolError,
    );
  });

  it("preserves AbortError when the caller cancels a request", async () => {
    const controller = new AbortController();
    controller.abort();

    await expect(api.health(controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
  });
});
