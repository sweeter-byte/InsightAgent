# Third-party notices

## GPT Researcher UI

The workspace information architecture was informed by the open-source GPT Researcher
UI, especially its `ResearchResults` composition and the Question, Report, Sources,
and LogsSection concepts:

- Project: https://github.com/assafelovic/gpt-researcher
- UI path: https://github.com/assafelovic/gpt-researcher/tree/master/frontend/nextjs
- Repository license: Apache License 2.0
- `frontend/nextjs/package.json` license field: MIT

No GPT Researcher source component was copied into this directory. The InsightAgent
components are clean, lightweight implementations against InsightAgent's own HTTP/SSE
contract. GPT Researcher's WebSocket transport, Agent Runtime, model settings, MCP,
authentication, and search backend were not imported.

## Lucide

Icons are provided by `lucide-react` under the ISC License. Package metadata and the
locked dependency version are retained in `package-lock.json`.
