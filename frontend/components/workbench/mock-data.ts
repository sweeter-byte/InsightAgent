export type PreviewScenario =
  | "idle"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export type ObservedEvent = {
  id: string;
  label: string;
  note: string;
  tone: "neutral" | "active" | "success" | "danger" | "warning";
};

export type ScenarioView = {
  statusLabel: string;
  statusDetail: string;
  statusTone: "neutral" | "active" | "success" | "danger" | "warning";
  runStatus: string;
  runId: string | null;
  threadId: string | null;
  error: string | null;
  canResume: boolean;
  events: ObservedEvent[];
};

const queued: ObservedEvent = {
  id: "1-0",
  label: "run.queued",
  note: "运行已进入后端队列",
  tone: "neutral",
};

const started: ObservedEvent = {
  id: "2-0",
  label: "run.started",
  note: "Research Runtime 开始执行",
  tone: "active",
};

const planning: ObservedEvent = {
  id: "3-0",
  label: "workflow.progress · planning",
  note: "正在规划并执行研究工作流",
  tone: "active",
};

export const scenarioViews: Record<PreviewScenario, ScenarioView> = {
  idle: {
    statusLabel: "空状态",
    statusDetail: "提交研究问题后，这里会显示真实运行状态。",
    statusTone: "neutral",
    runStatus: "—",
    runId: null,
    threadId: null,
    error: null,
    canResume: false,
    events: [],
  },
  running: {
    statusLabel: "执行中",
    statusDetail: "已收到粗粒度工作流事件，正在等待终态。",
    statusTone: "active",
    runStatus: "running",
    runId: "demo-run",
    threadId: "demo-thread",
    error: null,
    canResume: false,
    events: [queued, started, planning],
  },
  completed: {
    statusLabel: "已完成",
    statusDetail: "后端已返回 final_output 报告文本。",
    statusTone: "success",
    runStatus: "completed",
    runId: "demo-run",
    threadId: "demo-thread",
    error: null,
    canResume: false,
    events: [
      queued,
      started,
      planning,
      {
        id: "4-0",
        label: "workflow.progress · completed",
        note: "工作流已生成最终输出",
        tone: "success",
      },
      {
        id: "5-0",
        label: "run.completed",
        note: "final_output 已保存",
        tone: "success",
      },
    ],
  },
  failed: {
    statusLabel: "失败",
    statusDetail: "运行以失败终止；failed 状态当前不能 Resume。",
    statusTone: "danger",
    runStatus: "failed",
    runId: "demo-run",
    threadId: "demo-thread",
    error: "ResearchExecutionError",
    canResume: false,
    events: [
      queued,
      started,
      {
        id: "3-0",
        label: "run.failed",
        note: "ResearchExecutionError",
        tone: "danger",
      },
    ],
  },
  interrupted: {
    statusLabel: "可恢复",
    statusDetail: "运行已中断，可从保存的 thread checkpoint 恢复。",
    statusTone: "warning",
    runStatus: "interrupted",
    runId: "demo-run",
    threadId: "demo-thread",
    error: null,
    canResume: true,
    events: [
      queued,
      started,
      planning,
      {
        id: "4-0",
        label: "run.interrupted",
        note: "运行被中断，checkpoint 保留",
        tone: "warning",
      },
    ],
  },
};

export const mockMaterials = [
  { name: "chapter18-runtime.md", kind: "MD", chunks: 12, state: "已索引" },
  { name: "evaluation-notes.pdf", kind: "PDF", chunks: 28, state: "已索引" },
] as const;

export const scenarioOptions: Array<{ value: PreviewScenario; label: string }> = [
  { value: "idle", label: "空状态" },
  { value: "running", label: "运行中" },
  { value: "completed", label: "完成" },
  { value: "failed", label: "失败" },
  { value: "interrupted", label: "可恢复" },
];
