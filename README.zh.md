# InsightAgent

一个直接构建在 OpenAI 兼容 Chat Completions API 之上的、极简的手写 **Research Agent(研究型智能体)**。第一章验证了「最小可称为 agent loop」的那条链路:LLM 决策 → Runtime 执行工具 → 工具结果写回 messages → 再次调用 LLM。第二章把这条 loop 包进 `InsightAgent` 门面,并在其前方接入 **Intent Router(意图路由)** 为每个 query 决定执行路径。

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
python -m insight_agent "读取 examples/result.txt,准确率最高的方法是什么?"
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

## 明确未实现（留待后续章节）

以下能力当前不存在，刻意留到后续章节：

- 将 `analyze`/`research` 拆成独立 handler 的 Document / Multimodal Analysis Pipeline 与 Research Workflow
- RAG、Embedding、向量数据库、PDF 解析、Vision
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
