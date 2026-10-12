import { ArrowUp, BookOpen, Check, CircleDashed, Sparkles } from "lucide-react";

import type { PreviewScenario, ScenarioView } from "./mock-data";
import { MarkdownReport } from "./markdown-report";
import { SectionLabel } from "./section-label";
import type {
  ActiveResearchRun,
  AsyncPhase,
  SynchronousAnswer,
} from "./workbench-types";
import type { ResearchRunState } from "./run-state";

export type QueryMode = "smart" | "research";

type ResearchWorkspaceProps = {
  mode: QueryMode;
  onModeChange: (mode: QueryMode) => void;
  scenario?: PreviewScenario;
  view?: ScenarioView;
  canSubmit?: boolean;
  question?: string;
  onQuestionChange?: (value: string) => void;
  onSubmit?: () => void;
  submissionPhase?: AsyncPhase;
  submissionError?: string | null;
  answer?: SynchronousAnswer | null;
  activeRun?: ActiveResearchRun | null;
  runState?: ResearchRunState;
  selectionError?: string | null;
};

export function ResearchWorkspace({
  mode,
  onModeChange,
  scenario,
  view,
  canSubmit = false,
  question = "",
  onQuestionChange,
  onSubmit,
  submissionPhase = "idle",
  submissionError,
  answer,
  activeRun,
  runState,
  selectionError,
}: ResearchWorkspaceProps) {
  const isDemo = view !== undefined && scenario !== undefined;
  const presentation = isDemo
    ? view
    : livePresentation(submissionPhase, submissionError, activeRun, runState, selectionError);

  return (
    <main
      className="research-main"
      aria-label="研究工作区"
      data-testid="research-workspace"
    >
      <header className="research-header">
        <div>
          <p className="eyebrow">INSIGHT / 研究工作台</p>
          <h1>把问题变成可核查的研究</h1>
        </div>
        <div className="mock-flag">
          <CircleDashed size={14} /> {isDemo ? "Demo 协议预览" : "真实 HTTP 接口"}
        </div>
      </header>

      <section className="composer" aria-label="查询输入">
        <div className="mode-switch" role="group" aria-label="查询模式">
          <button
            type="button"
            aria-pressed={mode === "smart"}
            disabled={submissionPhase === "submitting"}
            onClick={() => onModeChange("smart")}
          >
            <Sparkles size={15} /> 智能提问
          </button>
          <button
            type="button"
            aria-pressed={mode === "research"}
            disabled={submissionPhase === "submitting"}
            onClick={() => onModeChange("research")}
          >
            <BookOpen size={15} /> 深度研究
          </button>
        </div>
        <label id="composer-title" htmlFor="research-question">研究问题</label>
        <textarea
          id="research-question"
          value={question}
          onChange={(event) => onQuestionChange?.(event.currentTarget.value)}
          rows={4}
        />
        <div className="composer__footer">
          <p>
            {mode === "research"
              ? "显式创建异步 Research Run"
              : "由后端 Intent Router 决定 Direct / Analyze / Research"}
          </p>
          <button
            className="submit-button"
            type="button"
            disabled={!canSubmit || !question.trim()}
            onClick={onSubmit}
          >
            {submissionPhase === "submitting"
              ? mode === "research"
                ? "正在提交 Research Run…"
                : "正在提交智能提问…"
              : mode === "research"
                ? "提交深度研究"
                : "提交智能提问"}
            <ArrowUp size={16} />
          </button>
        </div>
        {submissionError && <p className="composer-error" role="alert">{submissionError}</p>}
      </section>

      <section className={`run-banner run-banner--${presentation.statusTone}`} aria-live="polite">
        <div className="run-banner__signal" aria-hidden="true"><span /></div>
        <div>
          <p>RESEARCH RUN</p>
          <strong>{presentation.statusLabel}</strong>
          <span>{presentation.statusDetail}</span>
        </div>
        <code>{presentation.runId ?? "NO RUN"}</code>
      </section>

      <section className="answer-strip" aria-label="快速回答区域">
        <SectionLabel index="03" title="Direct / Analyze" />
        <div className="answer-strip__body">
          <span className="answer-strip__mark"><Check size={15} /></span>
          {answer ? (
            <p><strong>{answer.text}</strong><br />智能提问已由后端同步完成。</p>
          ) : (
            <p><strong>同步回答区域</strong><br />Direct 或 Analyze 的真实回答将在这里展示。</p>
          )}
          <span>{answer ? `INTENT · ${answer.intent}` : "等待回答"}</span>
        </div>
      </section>

      <section className="research-output" aria-label="研究输出">
        <SectionLabel
          index="04"
          title={isDemo && scenario === "completed" ? "Final Report" : "Research Activity"}
          action={<span className="section-caption">SUBMISSION FACTS</span>}
        />
        {isDemo ? (
          scenario === "completed"
            ? <MockReport />
            : <DemoResearchActivity scenario={scenario} view={view} />
        ) : (
          <LiveResearchActivity activeRun={activeRun} runState={runState} selectionError={selectionError} />
        )}
      </section>
    </main>
  );
}

function livePresentation(
  phase: AsyncPhase,
  error: string | null | undefined,
  activeRun: ActiveResearchRun | null | undefined,
  state: ResearchRunState | undefined,
  selectionError: string | null | undefined,
): ScenarioView {
  if (phase === "submitting") {
    return {
      statusLabel: "正在提交",
      statusDetail: "等待后端确认请求。",
      statusTone: "active",
      runStatus: "submitting",
      runId: null,
      threadId: null,
      error: null,
      canResume: false,
      events: [],
    };
  }
  if (phase === "failed") {
    return {
      statusLabel: "提交失败",
      statusDetail: error ?? "请求未能完成。",
      statusTone: "danger",
      runStatus: "failed",
      runId: null,
      threadId: null,
      error: error ?? null,
      canResume: false,
      events: [],
    };
  }
  if (selectionError) {
    return {
      statusLabel: "运行不存在",
      statusDetail: "保存的运行标识已失效或后端未找到该运行。",
      statusTone: "danger",
      runStatus: "not_found",
      runId: null,
      threadId: null,
      error: selectionError,
      canResume: false,
      events: [],
    };
  }
  if (activeRun) {
    const scopedState = state?.runId === activeRun.runId ? state : undefined;
    const status = scopedState?.observedStatus ?? scopedState?.record?.status ?? activeRun.initialRecord?.status ?? null;
    const statusView = statusPresentation(status);
    return {
      statusLabel: statusView.label,
      statusDetail: connectionLabel(scopedState?.connection ?? "idle"),
      statusTone: statusView.tone,
      runStatus: status ?? "—",
      runId: activeRun.runId,
      threadId: scopedState?.record?.thread_id ?? activeRun.initialRecord?.thread_id ?? null,
      error: scopedState?.error ?? scopedState?.record?.error ?? null,
      canResume: status === "interrupted" || status === "timed_out",
      events: [],
    };
  }
  return {
    statusLabel: "尚未提交",
    statusDetail: "提交后仅展示后端已确认的运行标识。",
    statusTone: "neutral",
    runStatus: "—",
    runId: null,
    threadId: null,
    error: null,
    canResume: false,
    events: [],
  };
}

function LiveResearchActivity({
  activeRun,
  runState,
  selectionError,
}: {
  activeRun?: ActiveResearchRun | null;
  runState?: ResearchRunState;
  selectionError?: string | null;
}) {
  if (selectionError) {
    return <div className="empty-state"><BookOpen size={22} /><h3>运行不存在</h3><p>已清除失效的本地运行标识。</p></div>;
  }
  if (!activeRun) {
    return (
      <div className="empty-state">
        <BookOpen size={22} />
        <h3>尚无研究运行</h3>
        <p>提交深度研究或由智能提问进入 Research 后显示真实 Run ID。</p>
      </div>
    );
  }
  const scopedState = runState?.runId === activeRun.runId ? runState : undefined;
  const status = scopedState?.observedStatus ?? scopedState?.record?.status ?? activeRun.initialRecord?.status ?? null;
  const progress = scopedState?.latestProgress;
  const stage = typeof progress?.stage === "string" ? progress.stage : null;
  const message = typeof progress?.message === "string" ? progress.message : null;
  const finalOutput = scopedState?.record?.final_output;
  const hasFinalOutput =
    status === "completed" &&
    finalOutput !== null &&
    finalOutput !== undefined &&
    finalOutput.trim().length > 0;
  return (
    <div className="activity-sheet">
      <div className="activity-sheet__header">
        <span>{activeRun.question ?? "重新挂载后任务文本不可用"}</span>
        <code>{status ?? "等待状态"}</code>
      </div>
      <span className={`connection-chip connection-chip--${scopedState?.connection ?? "idle"}`}>
        {connectionLabel(scopedState?.connection ?? "idle")}
      </span>
      {stage && <p className="progress-fact"><strong>{stage}</strong>{message && <span>{message}</span>}</p>}
      {hasFinalOutput ? (
        <div>
          <p className="honesty-note">最终报告可用</p>
          <MarkdownReport markdown={finalOutput} />
        </div>
      ) : (
        <p className="honesty-note">
          {status === "completed" && finalOutput === ""
            ? "报告内容为空"
            : status === "failed"
              ? "研究失败，未生成最终报告"
              : "最终报告尚未生成"}
        </p>
      )}
    </div>
  );
}

function statusPresentation(status: ResearchRunState["observedStatus"]): {
  label: string;
  tone: ScenarioView["statusTone"];
} {
  if (status === "queued") return { label: "任务已排队", tone: "neutral" };
  if (status === "running") return { label: "研究运行中", tone: "active" };
  if (status === "completed") return { label: "研究已完成", tone: "success" };
  if (status === "failed") return { label: "研究失败", tone: "danger" };
  if (status === "timed_out") return { label: "研究超时", tone: "warning" };
  if (status === "interrupted") return { label: "研究已中断", tone: "warning" };
  return { label: "等待运行状态", tone: "neutral" };
}

function connectionLabel(connection: ResearchRunState["connection"]): string {
  return {
    idle: "SSE 空闲",
    connecting: "SSE 连接中",
    connected: "SSE 已连接",
    reconnecting: "SSE 正在重连",
    disconnected: "SSE 已断开",
    protocol_error: "SSE 协议错误",
    not_found: "运行不存在",
  }[connection];
}

function DemoResearchActivity({
  scenario,
  view,
}: {
  scenario: PreviewScenario;
  view: ScenarioView;
}) {
  if (scenario === "idle") {
    return (
      <div className="empty-state">
        <BookOpen size={22} />
        <h3>尚无研究运行</h3>
        <p>这是显式 Demo 场景。</p>
      </div>
    );
  }
  return (
    <div className="activity-sheet">
      <div className="activity-sheet__header">
        <span>Demo 中的粗粒度运行事实</span>
        <code>{view.events.length} EVENTS</code>
      </div>
      <p className="honesty-note">后端尚未提供节点级进度，因此不展示百分比、Planner 子任务或 Evidence 节点状态。</p>
      {scenario === "failed" && <p className="inline-error">ResearchExecutionError</p>}
    </div>
  );
}

function MockReport() {
  return (
    <article className="report-paper">
      <div className="report-paper__meta">
        <span>final_output</span>
        <span>Mock 内容</span>
      </div>
      <h2>研究结果</h2>
      <p className="report-paper__lead">研究目标：比较三类查询路径的响应边界与运行特征。</p>
      <h3>任务边界</h3>
      <p>Direct 适合无需检索的即时回答；Analyze 使用共享本地知识集合；Research 创建异步运行，并通过状态与事件流观察执行结果。[1]</p>
      <h3>证据缺口</h3>
      <ul>
        <li>当前事件流不能证明单个研究节点的完成比例。</li>
        <li>资料检索尚不能按 material_id 隔离。</li>
      </ul>
      <h3>参考来源</h3>
      <p>[1] Chapter 18 运行时说明</p>
    </article>
  );
}
