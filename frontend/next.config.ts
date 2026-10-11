import type { NextConfig } from "next";

import {
  createBackendRewrites,
  resolveBackendOrigin,
} from "./lib/backend-origin";

const backendOrigin = resolveBackendOrigin(
  process.env.INSIGHTAGENT_BACKEND_ORIGIN,
);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  agentRules: false,
  async rewrites() {
    return createBackendRewrites(backendOrigin);
  },
  // Next 16.4 defaults to the detached TypeScript CLI path. In this managed
  // environment that child exits before its captured stdout is delivered.
  // TypeScript 5.9 still exposes the compiler API, so use that stable path.
  experimental: {
    useTypeScriptCli: false,
  },
};

export default nextConfig;
