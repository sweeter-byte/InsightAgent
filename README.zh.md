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

复制模板并填写下面三个必填变量:

```bash
cp .env.example .env
```

```text
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

`LLMClient` 在构造时读取这些环境变量;若任一缺失,会抛出携带明确提示的 `RuntimeError`。

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

第二章引入 **Intent Router** 与统一应用门面 `InsightAgent`。CLI 现在通过 `_build_app()` 组装(在一个共享的 `LLMClient` 上组合 `LLMClient` + `ToolRegistry` + `ResearchAgent` + `IntentRouter` + `InsightAgent`)。控制流程为:

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
