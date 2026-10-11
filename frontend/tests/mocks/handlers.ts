import { http, HttpResponse } from "msw";

import {
  healthResponseFixture,
  materialImportResultFixture,
  queuedRunFixture,
  readinessResponseFixture,
  researchQueryResponseFixture,
} from "@/lib/protocol/fixtures";

const apiBase = "http://127.0.0.1:8000";
const appBase = "http://localhost:3000";

export const handlers = [
  http.get(`${apiBase}/health`, () => HttpResponse.json(healthResponseFixture)),
  http.get(`${apiBase}/ready`, () => HttpResponse.json(readinessResponseFixture)),
  http.post(`${apiBase}/v1/query`, () =>
    HttpResponse.json(researchQueryResponseFixture, { status: 202 }),
  ),
  http.get(`${appBase}/health`, () => HttpResponse.json(healthResponseFixture)),
  http.get(`${appBase}/ready`, () => HttpResponse.json(readinessResponseFixture)),
  http.post(`${appBase}/v1/materials`, () =>
    HttpResponse.json(materialImportResultFixture, { status: 201 }),
  ),
  http.post(`${appBase}/v1/query`, () =>
    HttpResponse.json(researchQueryResponseFixture, { status: 202 }),
  ),
  http.post(`${appBase}/v1/research/runs`, () =>
    HttpResponse.json(queuedRunFixture, { status: 202 }),
  ),
];
