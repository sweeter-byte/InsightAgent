# InsightAgent DeepResearch Bench 端到端评测

本适配固定使用 DeepResearch Bench 官方仓库 Commit
`f2735b5c3636759c22f9e936c0232de8bf545b67`。该版本的
`data/prompt_data/query.jsonl` SHA256 为
`62b197a264a027fd4e2e795ff322a82dd6d566f5e350596ccc7572539ec987d1`，
包含 100 个任务，其中中文、英文各 50 个。`prepare` 会下载 Git checkout，
核验 Commit、文件哈希、任务数、ID 唯一性、必需字段和语言分布。官方仓库保存在
`.data/benchmarks/deepresearch_bench/official/`，不会写入知识库或 SciFact 索引。

## 执行边界

Generation 通过生产 `build_application` 组合现有组件，并直接调用
`ResearchCoordinator.start_research(prompt, thread_id=...)`。实际覆盖 Research
Planner、Retrieval Router、Tavily Web Retrieval、Evidence Collection/Grading、
Report Generator、Self Check/Repair 和 LangGraph checkpoint。它不经过 Intent
Router、HTTP、Redis、前端、普通用户知识库或 Vision Retrieval；这些边界会写入
`manifest.json`。评测专用 Coordinator 只向路由器暴露 Web 来源，避免开放域任务
误读现有本地知识库。报告直接取自 `ResearchThreadSnapshot.state.final_output`。

每个 Run 使用 `cases/<task_id>/` 和独立的
`checkpoints/checkpoints.sqlite`。每次任务结束立即原子更新 `execution.json` 与
`state.json`。成功任务默认跳过；中断、超时或进程退出后会使用原 Thread ID 和
LangGraph checkpoint 继续未完成节点；`--retry-failed` 会为已失败任务创建新的
Thread ID。同一 Run ID 的模型、任务、数据版本或执行限制改变时，配置指纹校验会
拒绝续跑。

Generation 进程只打开 `data/prompt_data/query.jsonl`。`criteria.jsonl` 与
`reference.jsonl` 只由独立的官方评分子进程读取；它们不会进入 Agent 上下文。

## 环境准备

项目使用 Conda。先进入已经配置好的环境，脚本不会创建环境、安装依赖或改变
PyTorch/CUDA：

```bash
conda activate insight-agent
scripts/benchmarks/deepresearch.sh prepare
scripts/benchmarks/deepresearch.sh doctor
```

Research Generation 必须配置：

```bash
export LLM_API_KEY='...'
export LLM_BASE_URL='https://.../v1'
export LLM_MODEL='...'
export TAVILY_API_KEY='...'
```

`doctor` 只检查配置、固定数据版本和输出路径，不调用 LLM、Tavily 或 Judge。
Generation 不会静默降级到 Mock 或无网络检索。

官方评分依赖官方 checkout 的 `requirements.txt`。如当前 Conda 环境缺少这些包，
请先检查解析计划，再显式执行 `pip install -e ".[deepresearch]"`；Benchmark 脚本
不会自动安装。官方 Judge 可选
OpenRouter 或 OpenAI：

```bash
export LLM_BACKEND=openrouter             # 或 openai
export OPENROUTER_API_KEY='...'            # openai 时使用 OPENAI_API_KEY
export JINA_API_KEY='...'                  # FACT 网页抓取必需
```

固定官方默认模型为 GPT-5.5（RACE）、GPT-5.6 Luna（文章清洗）和
GPT-5.4-mini（FACT）。显式覆盖 `RACE_MODEL`、`CLEAN_MODEL` 或 `FACT_MODEL`
时，结果会标记为 `custom_judge`，不会冒充官方 RACE/FACT 指标。

## 生成报告

Smoke 固定选择 4 个中英文任务，Pilot 固定选择 16 个任务，Full 使用全部 100 个：

```bash
scripts/benchmarks/deepresearch.sh smoke --run-id drb-smoke-01
scripts/benchmarks/deepresearch.sh pilot --run-id drb-pilot-01
scripts/benchmarks/deepresearch.sh full --run-id drb-full-01
```

这三个命令会调用 Research LLM 和 Tavily，产生费用。默认并发为 1、单任务超时
1800 秒、失败后最多重试 2 次。当前适配器拒绝大于 1 的并发。可用参数包括：

```bash
scripts/benchmarks/deepresearch.sh full --help
scripts/benchmarks/deepresearch.sh full \
  --run-id drb-selected-01 \
  --task-ids 1,8,51,58 \
  --language zh \
  --max-tasks 2 \
  --timeout-seconds 1200 \
  --max-retries 1
```

中断后用完全相同的参数和 Run ID 恢复；重试已记录的失败任务需增加
`--retry-failed`：

```bash
scripts/benchmarks/deepresearch.sh full --run-id drb-full-01
scripts/benchmarks/deepresearch.sh full --run-id drb-full-01 --retry-failed
```

## 导出、评分和报告

导出和汇总完全离线，不会产生 API 请求：

```bash
RUN=eval_results/benchmarks/deepresearch_bench/drb-full-01
scripts/benchmarks/deepresearch.sh export --run-dir "$RUN"
scripts/benchmarks/deepresearch.sh report --run-dir "$RUN"
```

导出文件为 `exports/insightagent.jsonl`，每行仅含原始 `id`、`prompt` 与完整
`article`。`report.raw.md` 和 `report.md` 当前保持一致，不截断正文，不补造 URL。
InsightAgent 的编号引用和参考来源 URL 能被官方 FACT 的提取 Prompt 识别；没有
URL 的来源仍按原样保留。

官方评分与 Generation 完全分离，不会重新运行 Research Agent。以下命令会调用
付费 Judge；FACT 还会调用 Jina：

```bash
scripts/benchmarks/deepresearch.sh score --run-dir "$RUN" --race --confirm-paid
scripts/benchmarks/deepresearch.sh score --run-dir "$RUN" --fact --confirm-paid
scripts/benchmarks/deepresearch.sh score --run-dir "$RUN" --race --fact --confirm-paid
```

只有显式给出 `--confirm-paid` 才会执行评分。重复命令默认利用官方脚本的断点数据；
若要在报告不变时重新评分，加 `--force`。RACE 与 FACT 失败会分别记录，不会触发
报告重生成，也不会把失败样本静默当作 0 分。评分元数据绑定导出 JSONL 的 SHA256；
报告变化后必须显式 `--force` 重评，离线汇总也会拒绝展示过期分数。

## 产物与核验

Run 位于 `eval_results/benchmarks/deepresearch_bench/<run_id>/`，主要文件为：

```text
manifest.json                 配置、代码与数据版本、覆盖边界
state.json                    任务级恢复状态
benchmark.log                 Generation 进度
checkpoints/checkpoints.sqlite
cases/<task_id>/metadata.json
cases/<task_id>/report.raw.md
cases/<task_id>/report.md
cases/<task_id>/execution.json
exports/insightagent.jsonl
scores/evaluation.json        Judge 模型、后端和分阶段状态
scores/evaluation.log
scores/race/
scores/fact/
metrics.json
bad_cases.json
summary.md
```

`execution.json` 中无法从 Provider 获得的 Token 计量为 `null`；不会填 0。
`tool_calls` 由实际 LangGraph Route Decision 数量得到。`metrics.json` 分开记录报告
生成、RACE、FACT 的有效样本与失败状态。`summary.md` 中的“生成成功”只表示产生了
非空报告，不代表官方 Benchmark 评分通过。

复现仍受外部网页变化、Tavily 索引、模型服务实现和非确定性影响。Manifest 记录
代码 Commit、Benchmark Commit/SHA、模型标识、检索配置、时间和执行限制，但无法
冻结第三方搜索结果或未返回的 Provider API 版本。
