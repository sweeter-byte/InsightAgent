import { describe, expect, it } from "vitest";

import {
  createBackendRewrites,
  resolveBackendOrigin,
} from "@/lib/backend-origin";

describe("backend origin configuration", () => {
  it("uses the local FastAPI origin by default", () => {
    expect(resolveBackendOrigin(undefined)).toBe("http://127.0.0.1:8000");
    expect(resolveBackendOrigin("  ")).toBe("http://127.0.0.1:8000");
  });

  it("normalizes an explicitly configured HTTP origin", () => {
    expect(resolveBackendOrigin("https://api.example.test:8443/")).toBe(
      "https://api.example.test:8443",
    );
  });

  it.each([
    "file:///tmp/backend",
    "https://user:pass@example.test",
    "https://example.test/path",
    "https://example.test?target=evil",
    "https://example.test#fragment",
  ])("rejects an unsafe backend target: %s", (value) => {
    expect(() => resolveBackendOrigin(value)).toThrow(
      "INSIGHTAGENT_BACKEND_ORIGIN",
    );
  });

  it("creates only fixed application API rewrites", () => {
    expect(createBackendRewrites("http://backend.test:8000")).toEqual([
      {
        source: "/health",
        destination: "http://backend.test:8000/health",
      },
      {
        source: "/ready",
        destination: "http://backend.test:8000/ready",
      },
      {
        source: "/v1/:path*",
        destination: "http://backend.test:8000/v1/:path*",
      },
    ]);
  });
});
