# InsightAgent Frontend

Next.js 三栏研究工作台，对接 InsightAgent Chapter 18 HTTP/SSE 接口。

## 安装与启动

需要 Node.js 20 或更新版本。在 `frontend` 目录安装锁定依赖并启动开发服务器：

```bash
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:3000`。生产构建与启动：

```bash
npm run build
npm start
```

## 连接后端

前端默认将固定的 `/health`、`/ready` 和 `/v1/*` 同源路径代理到
`http://127.0.0.1:8000`。先按仓库根目录 README 配置并启动 Redis 与 FastAPI：

```bash
redis-server --port 6379
```

```bash
conda run --no-capture-output -n insight-agent \
  python -m uvicorn insight_agent.runtime.app:app \
  --host 127.0.0.1 --port 8000
```

如果使用其他后端地址，在启动或构建 Next.js 时设置仅服务端可见的变量：

```bash
INSIGHTAGENT_BACKEND_ORIGIN=https://insightagent.example.test npm run dev
```

该值必须是无凭据、路径、查询参数和 fragment 的 HTTP(S) origin。它由
`next.config.ts` 读取，不会作为 `NEXT_PUBLIC_*` 变量暴露到浏览器。Provider、
模型和检索凭据只配置在后端进程中。

## 本地 Demo

推荐使用仓库已有的 `examples/chapter18_demo_material.md`：

1. 确认页面顶部 Health / Readiness 显示后端已就绪。
2. 在左侧上传 Markdown 文件，等待“已导入共享知识库”或去重成功提示。
3. 选择“智能提问”，分别提交可路由到 Direct 和 Analyze 的问题，检查同步回答。
4. 选择“深度研究”，提交基于已索引本地资料的问题。
5. 在右侧确认 Run ID、SSE 连接状态和去重后的运行事件。
6. Run 完成后，在中栏检查完整 Markdown 报告，在右侧检查可提取的编号来源。
7. 对 `interrupted` 或 `timed_out` Run，可点击“恢复运行”；前端使用同一 Run ID
   调用 Resume，并继续观察 SSE。

Research 可能调用外部模型或 Web Provider。演示时优先复用已完成的 Run 或已索引
本地资料，避免不必要的多轮外部研究。页面刷新后，同一标签页会从
`sessionStorage` 恢复最近一次 Run 的观察。

## 已实现功能

- 响应式三栏工作台，以及 Health / Readiness 组件状态。
- Markdown、纯文本、PDF 和后端允许的图片资料上传与索引反馈。
- Unified Query 的 Direct / Analyze / Research 展示，以及显式 Research Run 创建。
- Research Run 状态查询、SSE 事件、事件去重、断线重连、`Last-Event-ID` 和状态校准。
- 页面重新挂载后的 Run 观察恢复。
- 使用权威 `RunRecord.final_output` 渲染 GFM Markdown，包括标题、段落、列表、表格、
  引用和代码块；任意 HTML 不会直接注入，外部链接使用安全属性。
- 保留正文 `[1]`、`[2]` 等引用编号，并正常渲染“参考来源”；能保守提取时在右栏
  显示来源导航，提取失败时保留原报告且不虚构来源。
- `interrupted`、`timed_out` Run 的最小 Resume 操作，覆盖 202、404、409 和网络错误。

## 当前限制与未支持功能

- 当前没有认证、用户系统或权限隔离；不要将服务直接暴露到不可信网络。
- 后端使用共享知识库，Query 合同没有 `material_id` 过滤，因此不支持资料级、会话级
  或用户级检索隔离。
- 浏览器只保存当前页面导入结果；后端没有 `GET /v1/materials`，刷新后无法恢复资料列表。
- `sessionStorage` 只保存当前 Research Run ID；重新挂载后会重新获取权威状态和输出，
  但无法恢复原始问题文本。
- 来源区只展示后端报告实际提供的编号、标签和 locator/URL；没有 Evidence 评分、来源摘要、
  悬浮预览或复杂 Citation 管理。
- 不支持认证、研究历史、PDF 导出、富文本编辑、文档管理或细粒度节点进度。
- 前端不配置或保存任何 Provider/LLM 凭据。

## 质量检查

```bash
npm run typecheck
npm run lint
npm test
npm run test:sse-proxy
npm run build
```

`npm run test:sse-proxy` 只启动受控的本地 Node SSE 后端和 Next.js 开发服务器，验证
流式转发、请求头、重连游标与下游取消，不创建 Research Run，也不访问外部服务。
Next 16 的通用 rewrite 会缓冲 SSE，因此事件路径使用窄范围 Route Handler；其他
`/v1/*` 请求仍通过 rewrite。Production Build 使用受支持的 `--webpack` 路径，以避免
受管环境中 Turbopack PostCSS worker 的内部端口限制。

第三方来源与许可证说明保留在 [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。
