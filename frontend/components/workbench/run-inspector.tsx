import { ExternalLink, RotateCcw, TerminalSquare } from "lucide-react";

import type { ScenarioView } from "./mock-data";
import { SectionLabel } from "./section-label";

export function RunInspector({ view }: { view: ScenarioView }) {
  return (
    <aside className="rail rail--right" aria-label="运行检查器">
      <section className="rail-section">
        <SectionLabel index="05" title="Run Inspector" />
        <div className="status-card">
          <div className="status-card__top">
            <span className={`status-dot status-dot--${view.statusTone}`} />
            <strong>{view.statusLabel}</strong>
            <code>{view.runStatus}</code>
          </div>
          <dl>
            <div><dt>RUN ID</dt><dd>{view.runId ?? "—"}</dd></div>
            <div><dt>THREAD ID</dt><dd>{view.threadId ?? "—"}</dd></div>
            <div><dt>ERROR</dt><dd>{view.error ?? "—"}</dd></div>
          </dl>
        </div>
      </section>

      <section className="rail-section rail-section--grow">
        <SectionLabel
          index="06"
          title="Observed Events"
          action={<TerminalSquare size={15} aria-hidden="true" />}
        />
        {view.events.length === 0 ? (
          <p className="inspector-empty">事件流尚未建立。</p>
        ) : (
          <ol className="event-list">
            {view.events.map((event) => (
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
        {view.runStatus === "completed" ? (
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
        <button type="button" disabled={!view.canResume}>
          <RotateCcw size={15} /> 恢复运行
        </button>
        <p>{view.canResume ? "本阶段不会发送 Resume 请求，仅展示可用操作。" : "仅 interrupted / timed_out 状态可恢复。"}</p>
      </section>
    </aside>
  );
}
