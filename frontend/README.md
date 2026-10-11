# InsightAgent Frontend

Next.js workbench for the local InsightAgent backend.

## Commands

```bash
npm run dev
npm run typecheck
npm run lint
npm test
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
- Mock run data is available only through the explicit `demoScenario` component
  prop used by tests or deliberate demos; production defaults do not show it.
- No provider or LLM credentials are used in browser code.

SSE, run polling, Resume communication, Citation parsing, and live final-report
rendering remain intentionally deferred.

The production build uses Next.js' supported `--webpack` path. Turbopack's PostCSS
worker attempts an internal port bind that is denied by this managed environment.
