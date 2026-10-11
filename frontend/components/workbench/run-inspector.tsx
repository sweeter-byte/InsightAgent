import { ExternalLink, RotateCcw, TerminalSquare } from "lucide-react";

import type { ScenarioView } from "./mock-data";
import { SectionLabel } from "./section-label";
import type { ConfirmedRun } from "./workbench-types";

export function RunInspector({
  view,
  run,
  error,
}: {
  view?: ScenarioView;
  run?: ConfirmedRun | null;
  error?: string | null;
}) {
  const liveView: ScenarioView = view ?? {
    statusLabel: run ? "已提交" : error ? "提交失败" : "尚未提交",
    statusDetail: run ? "仅展示提交响应。" : "尚无已确认运行。",
    statusTone: run ? "success" : error ? "danger" : "neutral",
    runStatus: run?.status ?? "—",
    runId: run?.runId ?? null,
    threadId: run?.threadId ?? null,
    error: error ?? null,
    canResume: false,
    events: [],
  };
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
          <p className="inspector-empty">状态观察尚未接入。</p>
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
        <button type="button" disabled={!view || !liveView.canResume}>
          <RotateCcw size={15} /> 恢复运行
        </button>
        <p>{view && liveView.canResume ? "本阶段不会发送 Resume 请求，仅展示可用操作。" : "本阶段未接入 Resume 通信。"}</p>
      </section>
    </aside>
  );
}
