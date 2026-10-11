/**
 * Phase-one boundary only. Network execution and SSE lifecycle belong to phase two.
 */
export const apiRoutes = {
  health: "/health",
  ready: "/ready",
  materials: "/v1/materials",
  query: "/v1/query",
  researchRuns: "/v1/research/runs",
  researchRun: (runId: string) => `/v1/research/runs/${runId}`,
  researchEvents: (runId: string) => `/v1/research/runs/${runId}/events`,
  researchResume: (runId: string) => `/v1/research/runs/${runId}/resume`,
} as const;

export const defaultApiBaseUrl = "http://127.0.0.1:8000";

export function resolveApiUrl(path: string, baseUrl = defaultApiBaseUrl): URL {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return new URL(path.replace(/^\//, ""), normalizedBase);
}
