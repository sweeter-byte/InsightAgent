import { http, HttpResponse } from "msw";

import {
  healthResponseFixture,
  readinessResponseFixture,
  researchQueryResponseFixture,
} from "@/lib/protocol/fixtures";

const apiBase = "http://127.0.0.1:8000";

export const handlers = [
  http.get(`${apiBase}/health`, () => HttpResponse.json(healthResponseFixture)),
  http.get(`${apiBase}/ready`, () => HttpResponse.json(readinessResponseFixture)),
  http.post(`${apiBase}/v1/query`, () =>
    HttpResponse.json(researchQueryResponseFixture, { status: 202 }),
  ),
];
