import { ArrowUp, BookOpen, Check, CircleDashed, Sparkles } from "lucide-react";
import type { Dispatch, SetStateAction } from "react";

import type { PreviewScenario, ScenarioView } from "./mock-data";
import { SectionLabel } from "./section-label";

type QueryMode = "smart" | "research";

export function ResearchWorkspace({
  mode,
  onModeChange,
  scenario,
  view,
}: {
  mode: QueryMode;
  onModeChange: Dispatch<SetStateAction<QueryMode>>;
  scenario: PreviewScenario;
  view: ScenarioView;
}) {
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
        <div className="mock-flag"><CircleDashed size={14} /> Mock 协议预览</div>
      </header>

      <section className="composer" aria-labelledby="composer-title">
        <div className="mode-switch" role="group" aria-label="查询模式">
          <button
            type="button"
            aria-pressed={mode === "smart"}
            onClick={() => onModeChange("smart")}
          >
            <Sparkles size={15} /> 智能提问
          </button>
          <button
            type="button"
            aria-pressed={mode === "research"}
            onClick={() => onModeChange("research")}
          >
            <BookOpen size={15} /> 深度研究
          </button>
        </div>
        <label id="composer-title" htmlFor="research-question">研究问题</label>
        <textarea
          id="research-question"
          defaultValue="比较 Direct、Analyze 与多步 Research 三种路径，并说明它们各自适用的任务边界。"
          rows={4}
        />
        <div className="composer__footer">
          <p>{mode === "research" ? "异步 Research Run · HTTP + SSE" : "由后端 Intent Router 决定 Direct / Analyze"}</p>
          <button className="submit-button" type="button" disabled>
            下一阶段接入 <ArrowUp size={16} />
          </button>
        </div>
      </section>

      <section className={`run-banner run-banner--${view.statusTone}`} aria-live="polite">
        <div className="run-banner__signal" aria-hidden="true"><span /></div>
        <div>
          <p>RESEARCH RUN</p>
          <strong>{view.statusLabel}</strong>
          <span>{view.statusDetail}</span>
        </div>
        <code>{view.runId ?? "NO RUN"}</code>
      </section>

      <section className="answer-strip" aria-label="快速回答区域">
        <SectionLabel index="03" title="Direct / Analyze" />
        <div className="answer-strip__body">
          <span className="answer-strip__mark"><Check size={15} /></span>
          <p><strong>同步回答区域</strong><br />`/v1/query` 的 Direct 或 Analyze 文本将在这里展示。</p>
          <span>等待接入</span>
        </div>
      </section>

      <section className="research-output" aria-label="研究输出">
        <SectionLabel
          index="04"
          title={scenario === "completed" ? "Final Report" : "Research Activity"}
          action={<span className="section-caption">MARKDOWN / UTF-8</span>}
        />
        {scenario === "completed" ? <MockReport /> : <ResearchActivity scenario={scenario} view={view} />}
      </section>
    </main>
  );
}

function ResearchActivity({ scenario, view }: { scenario: PreviewScenario; view: ScenarioView }) {
  if (scenario === "idle") {
    return (
      <div className="empty-state">
        <BookOpen size={22} />
        <h3>尚无研究运行</h3>
        <p>提交深度研究后，真实事件与最终报告会在此出现。</p>
      </div>
    );
  }

  return (
    <div className="activity-sheet">
      <div className="activity-sheet__header">
        <span>当前仅展示可证实的运行事实</span>
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
