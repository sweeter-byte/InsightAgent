# InsightAgent

一个直接构建在 OpenAI 兼容 Chat Completions API 之上的、极简的手写 **Research Agent(研究型智能体)**。第一章验证了「最小可称为 agent loop」的那条链路:LLM 决策 → Runtime 执行工具 → 工具结果写回 messages → 再次调用 LLM。

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
>> 请读取 examples/result.txt,并告诉我准确率最高的方法。
```

Agent 会自行调用 `read_file`,拿到文件内容作为 tool message,再次询问 LLM,然后打印最终答复。输入 `exit` 或按 Ctrl-D 退出。

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

## 第一章明确未实现

以下能力刻意留到后续章节,当前不存在:

- RAG、Embedding、向量数据库
- Web Search / 联网浏览
- Vision / 多模态输入
- Planner、Intent Router、Multi-Agent 编排
- 长期 Memory、Context Compact
- MCP、Hooks、Permission、Workspace Sandbox
- Path Traversal 防护、文件大小限制、二进制文件支持
- Retry / Timeout / Fallback / Checkpoint
- FastAPI 或其他服务端组件
- LangGraph、LangChain、OpenAI Agents SDK
