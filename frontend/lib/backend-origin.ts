export const defaultBackendOrigin = "http://127.0.0.1:8000";

export function resolveBackendOrigin(value: string | undefined): string {
  const configured = value?.trim() || defaultBackendOrigin;
  let url: URL;
  try {
    url = new URL(configured);
  } catch {
    throw new Error("INSIGHTAGENT_BACKEND_ORIGIN must be a valid HTTP(S) origin");
  }

  const isHttp = url.protocol === "http:" || url.protocol === "https:";
  const isOriginOnly =
    !url.username &&
    !url.password &&
    url.pathname === "/" &&
    !url.search &&
    !url.hash;
  if (!isHttp || !isOriginOnly) {
    throw new Error(
      "INSIGHTAGENT_BACKEND_ORIGIN must be an HTTP(S) origin without credentials, path, query, or fragment",
    );
  }
  return url.origin;
}

export function createBackendRewrites(origin: string) {
  return [
    { source: "/health", destination: `${origin}/health` },
    { source: "/ready", destination: `${origin}/ready` },
    { source: "/v1/:path*", destination: `${origin}/v1/:path*` },
  ];
}
