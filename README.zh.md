# InsightAgent

InsightAgent 是一个后端研究智能体，提供统一 Query、受控本地资料导入、混合检索和
异步 Research Run。研究工作流使用 **LangGraph** 与 SQLite Checkpoint；Redis 保存
对外可见的 Run 状态和 SSE 事件。仓库顶层 `evals` 包独立提供可重复的离线
Evaluation、Baseline 注册与 Regression 比较。

当前交付仅包含后端，支持三种意图路径：

- `direct`：在当前请求内完成普通回答，不创建 Research Run。
- `analyze`：使用工具和共享本地知识索引同步分析，不创建 Research Run。
- `research`：立即返回 HTTP 202 与 `run_id`，随后通过状态和 SSE 观察执行，最终
  获得 Research Report。

## 架构

```text
POST /v1/materials -> 受控存储 -> 摄取 -> 切块/向量化 -> Qdrant -> 刷新混合检索

POST /v1/query -> IntentRouter
                 |- direct  -> 单次 LLM 回答
                 |- analyze -> ResearchAgent + 本地工具
                 `- research -> ResearchRuntimeService -> LangGraph 工作流
                                 |- Redis 状态 + SSE 事件
                                 `- SQLite Checkpoint + Resume

python -m evals -> 版本化 Dataset + 固定 Fixture -> Evaluation Result
                  -> 显式 Baseline -> Regression 报告
```

`insight_agent.application.build_application` 是组装根。FastAPI lifespan 位于
`insight_agent.runtime.app`，负责只初始化一次共享资源，并按所有权逆序释放。

## 安装

需要 Python 3.11 或更高版本。本仓库统一使用 Conda，不创建项目 `.venv`。

```bash
conda env list
conda create -n insight-agent python=3.11
conda run --no-capture-output -n insight-agent python -m pip install -e ".[dev]"
```

如果 `insight-agent` 环境已经存在，请保留该环境，仅在依赖需要更新时执行安装。

## 配置

```bash
cp .env.example .env
```

至少填写：

```dotenv
LLM_API_KEY=replace-me
LLM_BASE_URL=https://your-openai-compatible-host/v1
LLM_MODEL=your-tool-capable-model
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

`LLM_BASE_URL` 必须提供 OpenAI 兼容的 Chat Completions API。模型需要支持 Analyze
所需的 Tool Calling，以及 Research 工作流所需的结构化决策。`VISION_*` 和
`TAVILY_API_KEY` 可选。

Embedding 与 Reranker 模型可能在首次使用时下载。离线部署应预先准备模型缓存。
不要提交 `.env`。

## 启动 Redis 与服务

分别在两个终端执行：

```bash
redis-server --port 6379
```

```bash
conda run --no-capture-output -n insight-agent \
  python -m uvicorn insight_agent.runtime.app:app \
  --host 127.0.0.1 --port 8000
```

FastAPI lifespan 会打开 Redis、本地 Qdrant、模型客户端、混合检索和 SQLite
Checkpoint。必需资源无法初始化时，服务启动失败。

## Health 与 Readiness

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

`GET /health` 仅表示 HTTP 进程存活。`GET /ready` 检查已经由应用持有的 Redis、
Qdrant、Checkpoint 和配置资源。Web 与 Vision 是可选组件；任一必需组件不可用时
返回 HTTP 503。

## 导入演示资料

上传接口支持 `.md`、`.markdown`、`.txt` 和 `.pdf`；图片还需要完整的
`VISION_*` 配置。文件以流式方式写入 `MATERIAL_UPLOAD_DIR`，受
`MATERIAL_MAX_BYTES` 限制，按内容寻址，并在索引完成后返回。

```bash
curl --fail \
  -F 'file=@examples/chapter18_demo_material.md;type=text/markdown' \
  http://127.0.0.1:8000/v1/materials
```

```json
{
  "material_id": "<sha256>",
  "document_ids": ["<document-id>"],
  "status": "indexed",
  "chunk_count": 1,
  "deduplicated": false
}
```

新资料返回 HTTP 201；相同内容重复上传时返回 HTTP 200 与
`deduplicated: true`。接口不接受服务器本地路径或 URL。

## 统一 Query

### Direct

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"Direct answer only: what is 2 + 2?"}' \
  http://127.0.0.1:8000/v1/query
```

Direct 与 Analyze 均返回 HTTP 200，字段保持一致：

```json
{
  "request_id": "<request-id>",
  "intent": "direct",
  "status": "completed",
  "answer": "4",
  "run_id": null
}
```

### Analyze 已导入资料

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"分析已导入的第 18 章演示资料，并给出准确的 acceptance marker。"}' \
  http://127.0.0.1:8000/v1/query
```

预期回答来自共享本地索引并包含 `IA-CH18-READY`。当前 Query 协议不支持按
`material_id` 过滤。

### 通过统一入口提交 Research

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"研究并比较三种 Agent Memory 方案，说明取舍并给出来源。"}' \
  http://127.0.0.1:8000/v1/query
```

Research 立即返回 HTTP 202：

```json
{
  "request_id": "<request-id>",
  "intent": "research",
  "status": "queued",
  "answer": null,
  "run_id": "<run-id>"
}
```

意图分类由模型完成。需要确定创建 Research 时，应调用下面的显式 Runtime 入口。

## Research Run、状态与 SSE

显式创建异步 Run：

```bash
curl --fail -H 'Content-Type: application/json' \
  -d '{"query":"比较直接回答、资料分析和多步研究三种处理方式。"}' \
  http://127.0.0.1:8000/v1/research/runs
```

HTTP 202 响应包含 `run_id`、`thread_id`、`status`、时间戳、`error` 和
`final_output`。查询状态：

```bash
curl --fail http://127.0.0.1:8000/v1/research/runs/<run_id>
```

状态包括 `queued`、`running`、`completed`、`failed`、`timed_out` 和
`interrupted`。完成后 `final_output` 保存 Research Report。

读取 SSE，直到终止事件：

```bash
curl -N --fail \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/events
```

断线后可从上一个事件之后继续：

```bash
curl -N --fail -H 'Last-Event-ID: <event-id>' \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/events
```

事件包括 `run.queued`、`run.started`、`workflow.progress`、
`run.completed`、`run.failed`、`run.timed_out` 和 `run.interrupted`。

## 恢复中断任务

只有 `interrupted` 或 `timed_out` Run 可以 Resume。恢复保留原始 `run_id` 和
LangGraph `thread_id`，从 SQLite Checkpoint 继续，不重复提交初始 query。

```bash
curl --fail -X POST \
  http://127.0.0.1:8000/v1/research/runs/<run_id>/resume
```

安全复现方式：提交耗时较长的 Research Run，在状态为 `running` 时停止服务，再用
同一个 Redis 数据库和 `CHECKPOINT_PATH` 重启。启动会把遗留的 `running` 记录
协调为 `interrupted`，此时调用 Resume。其他状态返回 HTTP 409，避免重复执行。

## 固定五场景 Demo

真实服务就绪后执行：

```bash
conda run --no-capture-output -n insight-agent \
  bash examples/chapter18_demo.sh
```

脚本覆盖 Health/Readiness、资料导入、Direct、Analyze、显式 Research、状态、SSE
和离线 Evaluation。Resume 必须显式传入真实的中断或超时 Run：

```bash
INTERRUPTED_RUN_ID=<run-id> \
conda run --no-capture-output -n insight-agent \
  bash examples/chapter18_demo.sh
```

外部模型/Provider 部分属于手动 Smoke Test。离线测试通过不能证明外部服务可达。

## 离线 Evaluation

真实入口是仓库顶层 `evals` 包。以下命令不需要 Redis、FastAPI 或外部 API：

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --dataset eval_data/research_eval_smoke_v1.jsonl \
  --fixture evals/fixtures/smoke_v1.json \
  --mode offline \
  --output /tmp/insight-agent-eval/runs \
  --run-id chapter18-candidate
```

CLI 会标记 `MOCK TEST`，该结果不代表真实模型质量。Baseline 必须显式确认：

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --register-baseline /tmp/insight-agent-eval/runs/chapter18-candidate \
  --baseline-output /tmp/insight-agent-eval/baseline.json \
  --baseline-kind mock_only \
  --confirm-baseline
```

执行 Regression 比较：

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --baseline /tmp/insight-agent-eval/baseline.json \
  --candidate /tmp/insight-agent-eval/runs/chapter18-candidate \
  --policy evals/policies/default_v1.json \
  --report-output /tmp/insight-agent-eval/reports
```

比较退出码为：0 通过、1 regression、2 incompatible、3 inconclusive。

## 测试

```bash
conda run --no-capture-output -n insight-agent python -m pytest -q
```

自动测试使用 Fake LLM/Provider、内存 Runtime Store、临时文件和临时 SQLite
Checkpoint。缺少凭证或模型时，不会把未执行的真实 Provider Smoke Test 标为成功。

## 最终 HTTP API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/health` | HTTP 存活检查 |
| `GET` | `/ready` | 必需与可选依赖就绪检查 |
| `POST` | `/v1/materials` | Multipart 资料导入与索引 |
| `POST` | `/v1/query` | Direct / Analyze / Research 统一入口 |
| `POST` | `/v1/research/runs` | 显式创建异步 Research |
| `GET` | `/v1/research/runs/{run_id}` | Run 状态与最终报告 |
| `GET` | `/v1/research/runs/{run_id}/events` | SSE 进度与重放 |
| `POST` | `/v1/research/runs/{run_id}/resume` | 恢复中断/超时任务 |

所有 HTTP 响应均包含 `X-Request-ID`。

## 已知工程限制

- Research Runtime 是单进程实现；Redis 持久化公开状态，但本版本没有分布式 Worker
  Queue。
- SQLite Checkpoint 位于本地。水平扩容前需要共享 Checkpoint 方案。
- 本地知识索引由所有请求共享，`/v1/query` 不支持按资料过滤。
- 资料准备对客户端表现为同步完成，但摄取和索引已通过线程卸载，避免阻塞事件循环。
- 认证、授权、限流和生产 TLS 终止属于仓库之外的部署职责。
