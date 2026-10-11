import { z } from "zod";

import {
  apiErrorSchema,
  createResearchRunRequestSchema,
  healthResponseSchema,
  materialImportResultSchema,
  queryRequestSchema,
  queryResponseSchema,
  readinessResponseSchema,
  runRecordSchema,
} from "./protocol/schemas";
import type {
  HealthResponse,
  MaterialImportResult,
  QueryResponse,
  ReadinessResponse,
  RunRecord,
} from "./protocol/types";

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

export class ApiNetworkError extends Error {
  readonly name = "ApiNetworkError";

  constructor() {
    super("Unable to reach the InsightAgent backend");
  }
}

export class ApiHttpError extends Error {
  readonly name = "ApiHttpError";

  constructor(
    readonly status: number,
    readonly detail: string | null,
  ) {
    super(detail ?? `HTTP ${status}`);
  }
}

export class ApiProtocolError extends Error {
  readonly name = "ApiProtocolError";

  constructor() {
    super("The InsightAgent backend returned an unexpected response");
  }
}

type JsonRequestOptions<T> = {
  path: string;
  schema: z.ZodType<T>;
  init?: RequestInit;
  acceptedErrorStatuses?: readonly number[];
};

function requestUrl(path: string): string {
  if (typeof window === "undefined") {
    return path;
  }
  return new URL(path, window.location.origin).toString();
}

async function readErrorDetail(response: Response): Promise<string | null> {
  try {
    const parsed = apiErrorSchema.safeParse(await response.json());
    return parsed.success ? parsed.data.detail : null;
  } catch {
    return null;
  }
}

async function requestJson<T>({
  path,
  schema,
  init,
  acceptedErrorStatuses = [],
}: JsonRequestOptions<T>): Promise<T> {
  let response: Response;
  try {
    response = await fetch(requestUrl(path), init);
  } catch (error) {
    if (
      init?.signal?.aborted ||
      (error instanceof Error && error.name === "AbortError")
    ) {
      throw error;
    }
    throw new ApiNetworkError();
  }

  if (!response.ok && !acceptedErrorStatuses.includes(response.status)) {
    throw new ApiHttpError(response.status, await readErrorDetail(response));
  }

  try {
    return schema.parse(await response.json());
  } catch {
    throw new ApiProtocolError();
  }
}

function jsonRequest(body: unknown, signal?: AbortSignal): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  };
}

export const api = {
  health(signal?: AbortSignal): Promise<HealthResponse> {
    return requestJson({
      path: apiRoutes.health,
      schema: healthResponseSchema,
      init: { signal },
    });
  },

  readiness(signal?: AbortSignal): Promise<ReadinessResponse> {
    return requestJson({
      path: apiRoutes.ready,
      schema: readinessResponseSchema,
      init: { signal },
      acceptedErrorStatuses: [503],
    });
  },

  uploadMaterial(file: File, signal?: AbortSignal): Promise<MaterialImportResult> {
    const form = new FormData();
    form.append("file", file, file.name);
    return requestJson({
      path: apiRoutes.materials,
      schema: materialImportResultSchema,
      init: { method: "POST", body: form, signal },
    });
  },

  query(query: string, signal?: AbortSignal): Promise<QueryResponse> {
    const body = queryRequestSchema.parse({ query });
    return requestJson({
      path: apiRoutes.query,
      schema: queryResponseSchema,
      init: jsonRequest(body, signal),
    });
  },

  createResearchRun(query: string, signal?: AbortSignal): Promise<RunRecord> {
    const body = createResearchRunRequestSchema.parse({ query });
    return requestJson({
      path: apiRoutes.researchRuns,
      schema: runRecordSchema,
      init: jsonRequest(body, signal),
    });
  },
};

export type ApiOperation = "connection" | "upload" | "query";

export function describeApiError(error: unknown, operation: ApiOperation): string {
  if (error instanceof ApiNetworkError) {
    return "无法连接 InsightAgent 后端，请检查服务后重试。";
  }
  if (error instanceof ApiProtocolError) {
    return "后端返回了无法识别的响应，请检查前后端版本。";
  }
  if (error instanceof ApiHttpError) {
    if (operation === "upload") {
      if (error.status === 413) return "文件超过后端允许的上传大小。";
      if (error.status === 422 && error.detail?.includes("VISION_")) {
        return "图片解析不可用，请先配置 Vision Provider。";
      }
      if (error.status === 422) return "资料格式无效、内容为空或无法解析。";
      if (error.status === 503) return "知识服务暂不可用，请稍后重试。";
    }
    if (operation === "query") {
      if (error.status === 502) return "上游模型未能完成本次请求，请重试。";
      if (error.status === 503) return "研究运行服务暂不可用，请稍后重试。";
    }
    return `请求失败（HTTP ${error.status}）。`;
  }
  return "请求未能完成，请重试。";
}
