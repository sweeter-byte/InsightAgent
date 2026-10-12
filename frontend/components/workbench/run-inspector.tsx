import { ExternalLink, RotateCcw, TerminalSquare } from "lucide-react";

import type { ObservedEvent, ScenarioView } from "./mock-data";
import { SectionLabel } from "./section-label";
import type { ResearchRunState } from "./run-state";
import type { ActiveResearchRun } from "./workbench-types";
import type { SseEvent } from "@/lib/protocol/types";

export function RunInspector({
  view,
  activeRun,
  runState,
  error,
  selectionError,
}: {
  view?: ScenarioView;
  activeRun?: ActiveResearchRun | null;
  runState?: ResearchRunState;
  error?: string | null;
  selectionError?: string | null;
}) {
  const scopedRunState = activeRun && runState?.runId === activeRun.runId ? runState : undefined;
  const liveView: ScenarioView = view ?? createLiveView(activeRun, scopedRunState, error, selectionError);
  const connection = view ? null : connectionLabel(scopedRunState?.connection ?? "idle");
  return (
    <aside className="rail rail--right" aria-label="运行检查器">
      <section className="rail-section">
        <SectionLabel index="05" title="Run Inspector" />
        <div className="status-card">
          <div className="status-card__top">
            <span className={`status-dot status-dot--${liveView.statusTone}`} />
            <strong>{liveView.statusLabel}</strong>
            <code>{liveView.runStatus}</code>
          </div>
          <dl>
            <div><dt>RUN ID</dt><dd>{liveView.runId ?? "—"}</dd></div>
            <div><dt>THREAD ID</dt><dd>{liveView.threadId ?? "—"}</dd></div>
            {!view && <div><dt>CONNECTION</dt><dd>{connection}</dd></div>}
            <div><dt>ERROR</dt><dd>{liveView.error ?? "—"}</dd></div>
          </dl>
        </div>
      </section>

      <section className="rail-section rail-section--grow">
        <SectionLabel
          index="06"
          title="Observed Events"
          action={<TerminalSquare size={15} aria-hidden="true" />}
        />
        {liveView.events.length === 0 ? (
          <p className="inspector-empty">尚未收到后端事件。</p>
        ) : (
          <ol className="event-list">
            {liveView.events.map((event) => (
              <li key={event.id}>
                <span className={`event-node event-node--${event.tone}`} />
                <div>
                  <code>{event.id}</code>
                  <strong>{event.label}</strong>
                  <p>{event.note}</p>
                </div>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section className="rail-section source-panel">
        <SectionLabel index="07" title="Citations / Sources" />
        {view && liveView.runStatus === "completed" ? (
          <a href="#report-source-1">
            <span>[1]</span>
            <div><strong>Chapter 18 运行时说明</strong><small>本地资料 · Mock</small></div>
            <ExternalLink size={14} aria-hidden="true" />
          </a>
        ) : (
          <p className="inspector-empty">报告完成后展示可引用来源。</p>
        )}
      </section>

      <section className="resume-panel">
        <button type="button" disabled={!liveView.canResume}>
          <RotateCcw size={15} /> 恢复运行
        </button>
        <p>{liveView.canResume ? "未接入 Resume；本阶段不会发送 Resume 请求。" : "本阶段未接入 Resume 通信。"}</p>
      </section>
    </aside>
  );
}

function createLiveView(
  activeRun: ActiveResearchRun | null | undefined,
  state: ResearchRunState | undefined,
  submissionError: string | null | undefined,
  selectionError: string | null | undefined,
): ScenarioView {
  if (selectionError) {
    return {
      statusLabel: "运行不存在",
      statusDetail: selectionError,
      statusTone: "danger",
      runStatus: "not_found",
      runId: null,
      threadId: null,
      error: selectionError,
      canResume: false,
      events: [],
    };
  }
  const scopedState = activeRun && state?.runId === activeRun.runId ? state : undefined;
  const status = scopedState?.observedStatus ?? scopedState?.record?.status ?? activeRun?.initialRecord?.status ?? null;
  const presentation = statusPresentation(status);
  return {
    statusLabel: activeRun ? presentation.label : submissionError ? "提交失败" : "尚未提交",
    statusDetail: activeRun ? connectionLabel(scopedState?.connection ?? "idle") : "尚无已确认运行。",
    statusTone: activeRun ? presentation.tone : submissionError ? "danger" : "neutral",
    runStatus: status ?? "—",
    runId: activeRun?.runId ?? null,
    threadId: scopedState?.record?.thread_id ?? activeRun?.initialRecord?.thread_id ?? null,
    error: scopedState?.error ?? scopedState?.record?.error ?? submissionError ?? null,
    canResume: status === "interrupted" || status === "timed_out",
    events: scopedState?.events.map(observedEvent) ?? [],
  };
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

function observedEvent(event: SseEvent): ObservedEvent {
  const stage = typeof event.data.stage === "string" ? event.data.stage : null;
  const label = event.event === "workflow.progress" && stage
    ? `${event.event} · ${stage}`
    : event.event;
  return {
    id: event.id,
    label,
    note: eventNote(event, stage),
    tone: eventTone(event.event),
  };
}

function eventNote(event: SseEvent, stage: string | null): string {
  if (typeof event.data.message === "string") return event.data.message;
  if (typeof event.data.error === "string") return event.data.error;
  if (event.event === "workflow.progress" && stage) return `工作流阶段：${stage}`;
  if (event.event === "run.started" && typeof event.data.resume === "boolean") {
    return event.data.resume ? "从后端检查点恢复执行" : "后端开始执行";
  }
  if (event.event === "run.completed" && event.data.final_output != null) {
    return "后端事件包含最终输出";
  }
  if (event.event.startsWith("run.")) return "收到后端运行事件";
  return "收到后端事件";
}

function eventTone(event: string): ObservedEvent["tone"] {
  if (event === "run.completed") return "success";
  if (event === "run.failed") return "danger";
  if (event === "run.timed_out" || event === "run.interrupted") return "warning";
  if (event === "run.started" || event === "workflow.progress") return "active";
  return "neutral";
}
