# InsightAgent Frontend

Phase-one Next.js workspace for the local InsightAgent backend.

## Commands

```bash
npm run dev
npm run typecheck
npm run lint
npm test
npm run build
```

The frontend runs on `http://127.0.0.1:3000`; the existing backend remains on
`http://127.0.0.1:8000`.

## Current boundary

- UI data is explicitly marked as Mock.
- `lib/protocol` contains the Chapter 18 TypeScript/Zod contract and fixtures.
- `lib/api-client.ts` defines URLs only. It does not send HTTP requests or open SSE.
- The material list is session-preview data, not a persistent `GET /v1/materials` view.
- No provider or LLM credentials are used in browser code.

HTTP upload/query/run polling and SSE lifecycle are intentionally deferred to phase two.

The production build uses Next.js' supported `--webpack` path. Turbopack's PostCSS
worker attempts an internal port bind that is denied by this managed environment.
