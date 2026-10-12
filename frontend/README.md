# InsightAgent Frontend

Next.js workbench for the local InsightAgent backend.

## Commands

```bash
npm run dev
npm run typecheck
npm run lint
npm test
npm run test:sse-proxy
npm run build
```

The frontend runs on `http://127.0.0.1:3000`. It proxies the fixed `/health`,
`/ready`, and `/v1/*` paths to `http://127.0.0.1:8000` by default.

To use another backend origin, set the server-only variable before starting or
building Next.js:

```bash
INSIGHTAGENT_BACKEND_ORIGIN=https://insightagent.example.test npm run dev
```

The value must be an HTTP(S) origin without credentials, a path, query, or
fragment. It is read by `next.config.ts`; it is not exposed as a `NEXT_PUBLIC_*`
browser variable. Provider and model credentials stay in the backend process.

## Current boundary

- Health, readiness, material upload, unified query, and explicit Research Run
  creation use the real Chapter 18 HTTP API.
- JSON responses are validated with the Zod schemas in `lib/protocol`.
- Uploaded material records exist only in the current browser session because
  the backend has no `GET /v1/materials` endpoint.
- `sessionStorage` retains only the current Research Run ID so a same-tab reload
  can resume observation; authoritative status and output are fetched again.
- Mock run data is available only through the explicit `demoScenario` component
  prop used by tests or deliberate demos; production defaults do not show it.
- No provider or LLM credentials are used in browser code.

`npm run test:sse-proxy` starts only a controlled local Node SSE backend and a
local Next.js development server. It verifies streaming, headers, and reconnect
cursor forwarding through the frontend proxy; it never creates a Research Run
or calls an external service. A narrow Route Handler streams the SSE path because
the general Next 16 rewrite buffers it until completion. The verifier also checks
that aborting the downstream fetch closes the backend connection. Other `/v1/*`
requests keep using the rewrite.

Resume communication and Citation parsing remain intentionally deferred.

The production build uses Next.js' supported `--webpack` path. Turbopack's PostCSS
worker attempts an internal port bind that is denied by this managed environment.
