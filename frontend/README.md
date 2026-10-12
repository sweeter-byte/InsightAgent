# InsightAgent Frontend

InsightAgent 的 Next.js 研究工作台。前端通过同源 HTTP 与 SSE 路径连接 FastAPI，不保存任何 LLM、Vision、Web Search 或检索凭据。

## 技术栈

- Next.js 16.4、React 19.3、TypeScript 5.9。
- Tailwind CSS 4、Lucide React。
- Zod 协议校验、React Markdown + GFM + HTML Sanitize。
- Vitest、Testing Library、MSW。

需要 Node.js 20.9 或更高版本。

## 安装与启动

先按仓库根目录 [`README.zh.md`](../README.zh.md) 启动 Redis 与 FastAPI，然后执行：

```bash
cd frontend
npm ci
npm run dev
```

浏览器访问 <http://127.0.0.1:3000>。生产构建与本地启动：

```bash
npm run build
npm start
```

## 后端地址配置

前端默认将 `/health`、`/ready` 和 `/v1/*` 代理到 `http://127.0.0.1:8000`。使用其他后端时，在启动或构建 Next.js 前设置仅服务端可见的 Origin：

```bash
INSIGHTAGENT_BACKEND_ORIGIN=https://insightagent.example.test npm run dev
```

该值必须是无凭据、路径、查询参数和 fragment 的 HTTP(S) Origin。它由 `next.config.ts` 读取，不是 `NEXT_PUBLIC_*` 变量，不会作为配置值暴露给浏览器。

## HTTP 与 SSE 集成

- JSON 与 Multipart 请求通过相对路径调用 Health、Readiness、Material、Query 和 Research Run 接口，并使用 Zod 校验响应。
- Research 事件通过 `GET /v1/research/runs/{run_id}/events` 的 `text/event-stream` 读取。专用 Route Handler 透传流、`Last-Event-ID` 和取消信号，避免通用 Rewrite 缓冲事件。
- 客户端按事件 ID 去重，断线时携带最后事件 ID 重连，并用权威 Run 状态校准终态。
- 最近一次 Run ID 保存在 `sessionStorage`；页面重新挂载后重新获取状态和最终输出。资料列表与原始问题文本不会随刷新恢复。

## 当前功能

- 响应式三栏工作台与 Health/Readiness 状态。
- Markdown、纯文本、PDF，以及后端启用 Vision 后的图片上传。
- Smart Query 的 Direct/Analyze/Research 展示与显式 Deep Research。
- Research Run 状态、SSE 事件、重连、去重和 Resume。
- 安全渲染 GFM Markdown 报告、编号引用与可提取的来源导航。

当前没有认证、多用户隔离、资料历史、完整 Research 历史、PDF 导出或富文本编辑。不要将开发服务直接暴露到不可信网络。

## 测试与构建

```bash
npm run typecheck
npm run lint
npm test
npm run test:sse-proxy
npm run build
```

`npm run test:sse-proxy` 启动受控的本地 Node SSE 后端和 Next.js 开发服务器，验证流式转发、响应头、重连游标与下游取消；它不创建 Research Run，也不访问外部 Provider。

第三方来源与许可说明见 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。
