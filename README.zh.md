# InsightAgent

一个直接构建在 OpenAI 兼容 Chat Completions API 之上的、极简的手写 **Research Agent(研究型智能体)**。第一章验证了「最小可称为 agent loop」的那条链路:LLM 决策 → Runtime 执行工具 → 工具结果写回 messages → 再次调用 LLM。第二章把这条 loop 包进 `InsightAgent` 门面,并在其前方接入 **Intent Router(意图路由)** 为每个 query 决定执行路径。第三章搭建了一个独立的 **Multimodal Ingestion(多模态摄取)** 层,把 Text / Markdown / PDF / URL / Image 输入统一转换为 `list[Document]`。

不使用 LangChain,不使用 LangGraph,不使用 OpenAI Agents SDK,不使用任何现成的 Agent Framework。

## 环境要求

- Python **3.11+**
- 一个兼容 OpenAI Chat Completions 协议、且支持 Tool Calling 的 endpoint(OpenAI 官方,或任何遵循同一 `/v1/chat/completions` 协议的网关,如 vLLM、DeepSeek、Qwen、Moonshot 等)。

## 安装

本项目统一使用名为 `insight-agent` 的 conda 环境:

```bash
conda activate insight-agent
pip install -e ".[dev]"
```

## 配置 `.env`

复制模板并填写必填变量。除聊天 endpoint 外，Hybrid Retrieval 还要求显式配置
Cross-Encoder 模型：

```bash
cp .env.example .env
```

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

`insight_agent.application.AppConfig` 是应用启动配置的统一入口。
`AppConfig.from_env()` 加载 `.env`，在组装应用前解析模型、检索、可选
Vision/Web、Checkpoint 与 Runtime 参数。缺少必填的聊天或 Reranker 参数时，
会抛出明确的配置错误。`CHECKPOINT_PATH` 指定 LangGraph 的 SQLite Checkpoint
文件，默认值为 `.insight_agent/checkpoints.sqlite`，相对于当前工作目录解析。

## 应用启动（第十八章）

`insight_agent.application.build_application(config)` 是 CLI 与 HTTP Runtime
共用的唯一核心组装入口（composition root）。它接收显式配置，构建已有业务图，
返回共享同一个 LLM Client 的 `InsightAgent`。CLI 只负责输入、输出与错误适配：
读取 query、调用应用、打印答案或错误，并在退出时关闭应用。HTTP Runtime 在每次
lifespan 启动时只构建一次核心应用，各请求共享其依赖。

未配置 Vision 时，Qdrant 的打开与 Hybrid Retriever 的组装仍延迟到首次本地
检索。配置 Vision 后，启动阶段会打开 Qdrant，执行已有的图片能力探测。只有发现
索引中存在可用图片、且其受支持原图仍可读取时，才在启动阶段组装共享的 Hybrid
Retriever；视觉检索与本地知识库工具随后共享该 Retriever 和探测时打开的 Store。
Embedding 与 CrossEncoder 模型权重仍延迟到检索使用时加载。启动阶段不增加模型
预热，也不在每次请求时重新扫描图片。

资源所有权、构建中途失败后的清理与应用的幂等关闭，集中在核心组装入口及其持有的
资源中。Runtime lifespan 另外持有 Runtime Service 与 Redis Client，在关闭或
启动失败时一并释放这些资源与核心应用。

早期章节的独立组件构造器保留基于环境变量的默认值，作为示例和直接使用组件时的
显式兼容边界。正式入口通过 `AppConfig` 解析配置，再显式传入组件。
`TextChunker` 的 `CHUNK_SIZE`、`CHUNK_OVERLAP` 仍保留旧机制，因为索引写入
流水线不在本阶段范围内。第十八章调整启动组装和资源所有权，已有 HTTP API 与
前端行为保持不变。

## 运行 CLI

```bash
python -m insight_agent
```

启动后进入交互式 REPL:

```text
>> 什么是 Agent Loop？
```

用户无需手动选择模式 —— Intent Router 会对每个 query 分类,决定是直接回答还是交给 ResearchAgent。常识类问题会被路由为 `direct`,由单次 LLM 调用直接作答;而像下面第一章那样的读文件请求会被路由为 `analyze`/`research`,由 `ResearchAgent` 调用 `read_file`、把文件内容作为 tool message、再次询问 LLM,最后打印答复。输入 `exit` 或按 Ctrl-D 退出。

也支持一次性执行:

```bash
python -m insight_agent "读取 README.md，并概括项目当前实现范围"
```

## 运行测试

```bash
pytest
```

默认测试使用脚本化的 `FakeLLMClient`,**不会**访问任何真实 endpoint,不会产生 API 费用。

## 真实 endpoint 手动冒烟测试

不属于 `pytest` 的一部分。配好 `.env` 后,用上面的示例 query 跑一遍 CLI,确认模型确实调用了 `read_file` 并返回正确答案。这是最快验证你配置的 `LLM_BASE_URL` 与 `LLM_MODEL` 是否真的支持 tool calling 的方式。

## 第一章已经实现

- `LLMClient`:读取 env 配置,封装 `openai.OpenAI`,只暴露一个 `chat(messages, tools=None)`,返回 SDK 原始 response。
- `read_file(path) -> str`:读取 UTF-8 文本文件;文件缺失或读取失败时返回 `Error: ...` 字符串而不是抛出底层异常。
- `READ_FILE_SCHEMA`:`read_file` 对应的 OpenAI Tool Calling schema。
- `ToolRegistry`:`name -> callable` 的注册/查询/执行;附带 `build_default_registry()` 与 `default_tool_schemas()`。
- `ResearchAgent`:真正的 Agent Loop;`max_steps` 保护;messages 中的 assistant tool_calls 与 tool result 回写;对未知工具、非法 JSON 参数、工具内部异常等边界的兜底处理。
- `python -m insight_agent` CLI(REPL + 一次性执行)。

## 第二章新增

第二章引入 **Intent Router** 与统一应用门面 `InsightAgent`，在一个共享的
`LLMClient` 上组合 `ToolRegistry`、`ResearchAgent` 与 `IntentRouter`。当前 CLI
通过 `insight_agent.application.build_application` 获取该门面；以下流程描述
后续章节加入研究编排之前的第二章行为：

```text
用户
  ↓
Intent Router
  ├─ direct   → 单次 LLM 调用   (无工具、无 agent loop)
  ├─ analyze  → ResearchAgent
  └─ research → ResearchAgent
```

要点:

- `Intent`:三值枚举(`direct` / `analyze` / `research`),优先级 `research > analyze > direct`。
- `IntentRouter`:一次带分类 prompt 的 LLM 调用 + string→enum 解析;输出为空或非法时安全回退到 `research`。无结构化输出、无关键词启发、无置信度、无混合路由。
- `InsightAgent`:只负责意图分派。`direct` 走单次普通 chat(不进入 agent loop,因此不会产生 `AgentStepsExceeded`);`analyze` 与 `research` 都交给 `ResearchAgent`。
- `analyze` 与 `research` **当前故意复用同一个 `ResearchAgent`** —— 这是本章的设计,不是 bug。保留意图标签是为了后续章节能把它们分别拆成 Document / Multimodal Analysis Pipeline 与 Research Workflow。
- CLI(`python -m insight_agent`)两种模式保持不变;路由自动决定路径,正常输出中不会强制打印 Intent。

## 第三章新增

第三章引入**独立的多模态摄取层**(`insight_agent/ingestion/`)。它是一个数据处理层,**不属于** agent 的决策循环。三个概念层保持清晰分离:

```text
Intent Router        → 决定请求进入哪条高层路径
Multimodal Ingestion → 决定资料如何转换为统一 Document
Agent Loop / Tools   → 决定执行过程中下一步调用什么外部能力
```

五种输入统一归一为一种形态 —— `list[Document]`:

| 输入 | 入口 | 产生的 Document |
|---|---|---|
| Text | `ingest("text", inline_text)` | 1 个 —— 内联文本作为 content |
| Markdown | `ingest_file("x.md")` | 1 个 —— 原样保留 markdown |
| PDF | `ingest_file("x.pdf")` | 每个非空页 1 个(`metadata.page`/`page_count`) |
| URL | `ingest("url", "https://…")` | 1 个 —— 清理后的可见文本(`metadata.title`/`final_url`/`content_type`) |
| Image（PNG / JPG / JPEG / WEBP） | `ingest_file("x.png")` | 1 个 —— VLM 描述作为 content;原图路径保留在 `source` |

- `Document`:含 `content` / `source` / `source_type` / `metadata` 的简单 dataclass(见 `ingestion.models`)。
- 公开 API —— 只从 `insight_agent.ingestion` 导入,不要从 `ingestion.loaders.*` 导入:`ingest`、`ingest_file`、`ingest_text_file`、`infer_file_type`、`safe_ingest`,以及按格式的 `load_*` 逃生舱。
- 本地文件由**程序**按后缀分发(`infer_file_type`);LLM 不参与选择 loader,loader 也**不会**被注册为 Agent tool。
- 单一错误类型 `IngestionError`(继承 `RuntimeError`);`safe_ingest` 把任意底层/第三方异常包装为它并保留 `__cause__`。

视觉配置(仅图片)复用项目统一的 env 机制:`VISION_API_KEY`、`VISION_BASE_URL`、`VISION_MODEL`(见 `.env.example`)。Text / Markdown / PDF 摄取无需 API;URL 摄取需要网络但不需要 key。

基本文件摄取示例:

```python
from insight_agent.ingestion import documents_to_context, ingest_file

documents = ingest_file("path/to/paper.pdf")

for document in documents:
    print("source_type:", document.source_type.value)
    print("source:", document.source)
    print("metadata:", document.metadata)
    print("content preview:", document.content[:200])

context = documents_to_context(documents)
print(context[:500])
```

Text、Markdown 和 PDF 文件无需 API 配置。摄取图片时，例如
`ingest_file("path/to/chart.png")`，需要配置 `VISION_API_KEY`、
`VISION_BASE_URL` 和 `VISION_MODEL`。

### 临时 context 适配器(**不是** RAG)

`documents_to_context(documents)`(位于 `ingestion.context`)把 `list[Document]` 渲染成单个 `[Document N] / source / source_type / metadata / <content>` 文本块,便于当前 LLM/Agent 栈在小规模联调中消费摄取结果。它**不是 RAG**,除字符串拼接外不做任何事:无切块、无 embedding、无向量库、无检索、无排序;无 token 预算、无 context 压缩(内容原样转储)。不要借它把大型 Document 集合直接塞进 Agent context。

## 第三章仍未实现

以下能力刻意留到后续章节:

- Chunking / 文本切分
- Embedding 与任何向量数据库
- 检索 / RAG pipeline、token 预算、context 压缩
- OCR 与扫描件 PDF 处理
- URL loader 的 SSRF 防护、JS 渲染、登录/重 JS 页面抽取
- 把 loader 注册为 LLM tool、或让模型自行选择 loader
- 把摄取层接入 CLI / `InsightAgent` 门面(集成为后续工作)

## 明确未实现（留待后续章节）

以下能力当前不存在，刻意留到后续章节：

- 将 `analyze`/`research` 拆成独立 handler 的 Document / Multimodal Analysis Pipeline 与 Research Workflow
- Web Search、Research Planner、Workflow 编排、LangGraph
- Memory、MCP
- Router 置信度、混合路由、Intent 评测基准
- Multi-Agent 编排、Planner
- 长期 Memory、Context Compact
- Hooks、Permission、Workspace Sandbox
- Path Traversal 防护、文件大小限制、二进制文件支持
- Retry / Timeout / Checkpoint
- FastAPI 或其他服务端组件
- LangChain、OpenAI Agents SDK

## 第五章新增：本地 Vector RAG

第五章在第四章持久化索引之上补齐最小查询链路，同时严格分离索引与检索生命周期：

```text
query → embed_documents([query]) → Qdrant Top-K 搜索
      → RetrievalResult → 结构化 Observation → search_knowledge_base
      → 现有 ResearchAgent Loop
```

- `insight_agent.retrieval.VectorRetriever` 复用索引阶段相同的
  `SentenceTransformerEmbedder` 与 `QdrantVectorStore` 配置。
- `RetrievalResult` 是带当前查询 `score` 的检索模型；索引模型 `Chunk` 不变。
- Qdrant 查询返回 payload、不返回已保存的 vectors；关键来源字段缺失时明确失败。
- 原有扁平 `ToolRegistry` 注册 `search_knowledge_base(query, top_k=5)`，其中
  `1 <= top_k <= 8`，Observation 同时保留正文、来源和 metadata。
- Intent Router 仍只选择 `direct` / `analyze` / `research`，不知道 embedding、
  Qdrant 或 Top-K；进入 Agent Loop 后由模型选择具体工具。

正式查询路径只读取已有 collection，不会在每次提问时重新摄取或建立索引。
可用独立示例验证一次完整的“摄取 → 索引 → 检索”：

```bash
conda run --no-capture-output -n insight-agent \
  python examples/try_vector_rag.py notes/rag.md "为什么分块需要 overlap？" --top-k 5
```

Sentence Transformer 首次使用时可能下载配置的模型。本章不包含稀疏/混合检索、
重排、查询改写、Citation、Web fallback、LangChain 或 LangGraph。

## 第六章新增：混合检索与重排序

第六章在保持 `search_knowledge_base` Tool 不变的前提下，将检索链路升级为：

```text
query
├── Dense Retrieval ─┐
└── BM25 Retrieval ──┴→ RRF → Candidate Cutoff → Cross-Encoder → Final Top-K
```

- Qdrant 仍是 Chunk 的事实来源。Tool 第一次使用时从 Qdrant payload 一次性构建
  内存 BM25 快照，之后的查询复用该快照。
- 资料重新索引后显式调用 `HybridRetriever.refresh()`。刷新先完整构建替代快照，
  再在短锁内交换引用；刷新失败时旧快照继续可用。
- BM25 构建语料和处理 query 共用同一个支持 `jieba` 的 tokenizer，并保留
  `DEEPSEEK_API_KEY`、异常类名和文件路径等技术标识符。正值
  `log(1 + RSJ)` IDF 让极小语料中的唯一词仍具有区分度；没有可检索 token 的
  Chunk 仍可由 Dense Retrieval 召回。
- RRF 只使用排名，按稳定 `chunk_id` 去重；不会直接相加 Dense 与 BM25 原始分数。
- 只有融合后的前 `HYBRID_RERANK_K` 个候选会进入 Cross-Encoder。
  `RERANKER_MODEL` 是必填配置，业务代码没有静默 fallback；推荐中英双语模型为
  `BAAI/bge-reranker-v2-m3`。
- 阶段深度分别由 `HYBRID_DENSE_K`、`HYBRID_SPARSE_K`、
  `HYBRID_RERANK_K`、`HYBRID_FINAL_TOP_K` 与 `HYBRID_RRF_K` 配置。内部召回深度
  可以大于 8；只有 Agent 可见的最终 Tool 参数保持 `1 <= top_k <= 8`。显式 Tool
  参数优先于 `HYBRID_FINAL_TOP_K`。
- DEBUG 日志与可选 trace callback 可观察 Dense、Sparse、RRF、Cross-Encoder 和
  Final 排名，但不会把新增内部评分塞进 Agent Observation。

Tool 名称与 Schema、Registry、Intent Router 和 Agent Loop 均保持不变。本章不实现
Query Rewrite、Citation、Web fallback、自动索引变更检测、LangChain 或 LangGraph。

可通过下面的手动集成示例查看精确标识符与语义查询对应的 Dense、Sparse、RRF、
Cross-Encoder 和 Final 排名：

```bash
conda run --no-capture-output -n insight-agent \
  python examples/try_hybrid_rag.py notes/config.md notes/chunking.md
```

## 第九章新增：网页搜索

第九章把第八章的 `web_entry` 占位分支补成最小公开网页检索链路：

```text
Research Task -> Retrieval Router -> WebRetriever
              -> Tavily Search -> 候选 URL
              -> 既有 URL Loader -> Document -> ResearchState.web_results
```

Tavily 只负责发现候选网页；Provider 不请求 Tavily answer、图片或 raw content。
候选 URL 仍由第三章 URL Loader 获取和清洗。生成的 Document 在保留 `title`、
`final_url`、`content_type` 等 Loader metadata 的同时，增加 `search_query`、
`search_rank`、`search_title`、`search_snippet` 和可选 `search_score`。

Web Search 是可选能力。只有配置以下变量后，Runtime 才会把 `web` 放入
Retrieval Router 的 `available_sources`：

```text
TAVILY_API_KEY=tvly-...
WEB_SEARCH_TIMEOUT=30
WEB_SEARCH_LIMIT=5
WEB_FETCH_LIMIT=3
```

未配置 `TAVILY_API_KEY` 时，local 路由仍可用，但不会开放 WEB。Search Provider
整体失败会明确报错；单个候选网页获取失败则写入
`WebRetrievalResult.failures`，后续候选仍会继续处理。上述结果只是原始检索资料，
不是 Evidence、Citation、置信度或最终报告。

第九章不实现 Query Rewrite、多 Query/多轮搜索、Tavily answer/extract/crawl/
research API、浏览器自动化、来源权威性评分、Evidence/Citation、报告生成或
Vision Retrieval。

## 第十章新增：视觉检索

第十章将已有的 `vision_entry` 补全为面向当前任务的视觉检索：

```text
Research Task -> Hybrid Retrieval（source_type=image）
              -> 按原始图片路径去重
              -> 重新读取仍可访问的原图
              -> VLM 围绕当前任务分析
              -> ResearchState.vision_results[task_id]
```

第三章图片摄取接口保持不变：`describe_image(path)` 仍生成进入文本索引的通用
描述。视觉检索先搜索这些描述，再从 `RetrievalResult.source` 恢复原图，并使用
包含研究目标、当前任务和用户约束的独立 Prompt 重新分析。图片中的文字只作为
不可信资料处理，不能成为可执行指令。

`HybridRetriever.retrieve()` 新增可选的仅限关键字参数 `source_types`。Dense
Retrieval 将其下推为 Qdrant payload filter；BM25 在排序与 Top-K 截断前限制可选
Chunk。不传该参数时，本地检索行为保持不变。

只有现有 `VISION_API_KEY`、`VISION_BASE_URL`、`VISION_MODEL` 配置完整，且启动时
一次性探测到索引中至少一张仍可读取的受支持原图，Runtime 才向 Retrieval Router
声明 VISION 可用。此时本地知识库工具与 Vision Retriever 共享同一个 Hybrid
Retriever；每次路由不会重复扫描知识库。

每个任务的 `VisionRetrievalResult` 分别保存成功分析、结构化的逐图失败，以及独立
的“无候选”状态。一张图片缺失或 VLM 调用失败不会丢弃其他图片的成功结果。这些
结果只是检索材料，不是 Evidence 或 Citation。

最小端到端样例位于 [`examples/vision_workflow.png`](examples/vision_workflow.png)，
旁边保留了可编辑的 SVG。可使用任务：“判断图中的本地、网页和视觉三条路径是否
都会汇合到同一个任务推进节点。”

第十章不实现 CLIP、第二套向量库、OCR、裁剪、检测框、多轮 Vision Agent、
Evidence 评分、引用或报告生成。

## 第十六章新增：Research Runtime

第十六章在已有、可持久化的 Research Workflow 外增加独立的单进程运行层：

```text
POST /v1/research/runs -> Redis Run Registry -> queued
                       -> asyncio Semaphore -> running
                       -> 既有 ResearchCoordinator / SQLite Checkpoint
                       -> Redis Stream 事件 -> SSE
```

Runtime 严格区分 `request_id`、`run_id` 和 LangGraph `thread_id`。Redis 只保存
运行记录与面向客户端的事件；Research Plan、Evidence、Report 和完整
`ResearchState` 仍由第十五章 SQLite Checkpointer 管理。超时或中断的 Run 通过原
`run_id -> thread_id -> checkpoint` 链显式恢复。服务启动时只把遗留的
`running` 记录标记为 `interrupted`，不会自动继续消耗模型额度。

当前 HTTP 入口在 FastAPI lifespan 启动时解析 `AppConfig`，并调用共享的
`build_application` 构建一次核心应用。各请求使用该应用的
`ResearchCoordinator`，请求处理器不重新构建核心依赖。Lifespan 同时持有
Redis、Runtime Service 与核心应用，在关闭或启动失败时清理。

启动本地 Redis、配置已有模型与检索参数后运行：

```bash
conda run --no-capture-output -n insight-agent \
  uvicorn insight_agent.runtime.app:app
```

开发环境如需临时 Redis，只需容器化 Redis；应用与 Qdrant Local Mode 仍直接
运行在 Conda 环境中：

```bash
docker run --rm -p 6379:6379 redis:7-alpine
```

### 资料准备与运行状态检查

受控上传入口默认接受 Markdown、TXT 和 PDF；只有完整配置 `VISION_*` 后才接受
PNG、JPEG 和 WEBP。服务端把文件流式写入 `MATERIAL_UPLOAD_DIR`，执行
`MATERIAL_MAX_BYTES` 限制，用 SHA-256 生成存储文件名，再依次执行摄取、分块、
Embedding、Qdrant Upsert 和 BM25 刷新。

```bash
curl -F 'file=@./notes.md' http://127.0.0.1:8000/v1/materials
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

`material_id` 只是内容/导入追踪标识，不代表资料级检索作用域；本地检索仍查询
共享 Collection。后端没有 URL 摄取接口。Qdrant 继续使用 Local Mode，进程为
`QDRANT_PATH` 持有唯一客户端，避免索引与读取争抢嵌入式存储文件锁。

`/health` 只检查 HTTP 进程存活；`/ready` 检查现有 Redis、Qdrant、SQLite
Checkpoint 和已校验配置，并单独报告可选 Web/Vision 状态，不调用模型或外部
搜索服务。

当前后端仍未实现身份认证、权限控制、租户隔离或限流。由于服务能够接收文件，
不得直接暴露到不受信任的公网；应限制在可信网络，或在前方部署带认证和上传
限制的网关。

创建、查询、订阅和恢复 Run：

```bash
curl -X POST http://127.0.0.1:8000/v1/research/runs \
  -H 'Content-Type: application/json' \
  -d '{"query":"比较 Agent Memory 的主要设计思路"}'

curl http://127.0.0.1:8000/v1/research/runs/<run_id>
curl -N http://127.0.0.1:8000/v1/research/runs/<run_id>/events
curl -X POST http://127.0.0.1:8000/v1/research/runs/<run_id>/resume
```

`AppConfig.runtime` 统一保存 Redis 连接与执行策略，`AppConfig.checkpoint`
保存研究 Checkpoint 路径：

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `CHECKPOINT_PATH` | `.insight_agent/checkpoints.sqlite` | LangGraph 研究 Checkpoint 数据库 |
| `RUNTIME_REDIS_URL` | `redis://localhost:6379/0` | Runtime 记录与事件流 |
| `RUNTIME_MAX_CONCURRENCY` | `2` | 当前进程内的研究并发数 |
| `RUNTIME_RUN_TIMEOUT_SECONDS` | `900` | Run 执行超时，单位秒 |
| `RUNTIME_EVENT_TTL_SECONDS` | `86400` | 事件流保留时间，单位秒 |
| `RUNTIME_INFRA_RETRY_ATTEMPTS` | `3` | 基础设施操作最大尝试次数 |
| `RUNTIME_INFRA_RETRY_BACKOFF_SECONDS` | `0.2` | 基础设施重试退避时间，单位秒 |
| `MATERIAL_UPLOAD_DIR` | `.data/materials` | 服务端受控资料存储目录 |
| `MATERIAL_MAX_BYTES` | `20971520` | 单个上传文件最大字节数 |

本章不实现认证、分布式 Worker、自动恢复、WebSocket 或 Exactly-Once。

## 第十七章新增：Evaluation Harness

第十七章把评测保留在生产 Research Workflow 之外。Runner 读取固定 JSONL
Dataset：带 `retrieval` 标签的 Case 直接调用 Retriever，带 `research` 标签的
Case 直接调用注入的 Workflow Runner，不默认经过 FastAPI、SSE 或 Redis。生产
适配器直接复用既有 Planner 与 `ResearchRoutingWorkflow`，且不传 checkpoint
配置；仓库自带的 smoke 运行则明确使用 Fake Workflow。

以下命令运行完全禁止网络访问的 Offline Mock Test。搜索结果、网页正文均来自录制
Fixture，Vision Case 复用仓库已有本地图片：

```bash
conda run --no-capture-output -n insight-agent \
  python -m evals \
  --dataset eval_data/research_eval_smoke_v1.jsonl \
  --fixture evals/fixtures/smoke_v1.json \
  --mode offline \
  --output eval_results/runs
```

可重复传入 `--case <case-id>` 选择 Case。CLI 还支持 `--dataset-version`、
`--model`、`--prompt-version`、`--index-version`、`--top-k` 与 `--run-id`。
每个 Run 记录 Git Commit、Dataset Version 与 SHA-256、Model、Prompt Version、
Index Version、Retriever Configuration、Mode、Evaluation Type 和稳定的配置指纹。

结果目录如下：

```text
eval_results/runs/<eval-run-id>/
├── run.json
├── bad_cases.json
└── cases/<safe-case-id>/
    ├── result.json
    └── artifacts.json
```

`result.json` 保留逐 Judge 的执行状态与原因、结构错误、各层原始结果、运行指标和
Case 级异常；`artifacts.json` 保留必要的 Workflow 中间产物。单个 Case 失败不会
丢弃后续 Case。汇总质量分母只包含已计算指标和 `completed` Judge Label；
`not_computable`、`timeout`、非法输出、后端错误与运行失败均单独计数。无法从现有
产物观察到的 Token 或调用指标写为 `null`，不会估算。

smoke Dataset 只包含可追溯标签：Gold Retrieval ID 明确属于
`smoke-index-v1`，Web required point 指向固定网页正文；Vision 语义 Gold 缺失，
因此相应维度明确跳过。Offline Runner 在每个 Case 外安装 Socket/DNS Guard，
真实 Web Search、网页服务、外部 LLM Judge、VLM、Embedding 或其他网络模型组件
都不能静默越过 Fixture 边界。

上述结果是 **Mock Test**，不能用于宣称 InsightAgent 的真实研究质量。Live Test
必须单独组装真实 Retriever、Workflow、Judge、VLM 与 Embedding，并标记为
`live_test`；Fixture CLI 会拒绝 `--mode live`，避免把录制结果冒充真实模型结果。

### Bad Case、Baseline 与 Regression

每个完整 Evaluation Run 都会写入版本化的 `bad_cases.json`。确定的语义或结构失败
会分类为 `retrieval_miss`、`rerank_drop`、`evidence_noise`、`evidence_gap`、
`unsupported_claim`、`citation_error` 或 `constraint_violation`；Case 执行失败单独
使用 `runtime_failure`。Judge timeout、backend error、invalid output 与缺少 Gold
Label 会作为评测异常或不可计算项展示，不会被静默算成 Agent 质量失败。

Baseline 只能通过独立命令显式登记，普通评测不会自动更新它：

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --register-baseline eval_results/runs/<eval-run-id> \
  --baseline-output eval_results/baselines/research-smoke-v1.json \
  --baseline-kind mock_only \
  --confirm-baseline
```

`mock_test` 只能登记成 `mock_only` Baseline，用于验证评测框架和固定 Fixture 的
回归行为，不能冒充真实质量基线。除非显式传入 `--overwrite-baseline`，已有 Baseline
不会被覆盖。

下面的命令直接比较已落盘 Candidate，不会重跑 Workflow 或 Judge：

```bash
conda run --no-capture-output -n insight-agent python -m evals \
  --baseline eval_results/baselines/research-smoke-v1.json \
  --candidate eval_results/runs/<candidate-run-id> \
  --policy evals/policies/default_v1.json \
  --report-output eval_results/reports/<candidate-run-id>
```

Comparator 要求 Dataset 版本与内容、Case ID、Gold ID、Fixture、Evaluation
Mode/Test Kind，以及 Gold Retrieval Label 所依赖的 Index Version 一致。Model、
Prompt、Git Commit 与 Reranker 的变化只是被观察的系统改动，不会仅因配置指纹变化
就拒绝比较。缺失指标保持缺失，不按 0 计算；Policy 决定缺失指标与评测基础设施
错误应当忽略、判为回归，还是给出 inconclusive。

命令稳定生成 `regression.json` 和 `regression.md`。退出码为：pass=`0`、
regression=`1`、incompatible/配置错误=`2`、inconclusive=`3`。

至此完成的是第十七章的评测基础设施：固定 Dataset/Fixture、三层 Evaluator、结构化
Artifact、Bad Case、显式 Baseline、Policy 驱动的比较与报告。仓库内置 Mock Test
并未完成真实模型质量验证；后者仍需要单独装配真实组件、运行 `live_test`，并对人工
标注与 Judge 结果进行审查。
