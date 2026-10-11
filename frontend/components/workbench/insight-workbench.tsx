"use client";

import { Files, Gauge } from "lucide-react";
import { useState } from "react";

import { MaterialsWorkspace } from "./materials-workspace";
import {
  scenarioOptions,
  scenarioViews,
  type PreviewScenario,
} from "./mock-data";
import { ResearchWorkspace } from "./research-workspace";
import { RunInspector } from "./run-inspector";

export function InsightWorkbench({
  initialScenario = "running",
}: {
  initialScenario?: PreviewScenario;
}) {
  const [scenario, setScenario] = useState<PreviewScenario>(initialScenario);
  const [mode, setMode] = useState<"smart" | "research">("research");
  const [mobilePanel, setMobilePanel] = useState<"materials" | "run" | null>(null);
  const view = scenarioViews[scenario];

  return (
    <div className="workbench-frame">
      <div className="preview-bar">
        <div>
          <span>PHASE 01</span>
          <strong>界面状态预览</strong>
        </div>
        <div className="preview-switch" role="group" aria-label="Mock 运行状态">
          {scenarioOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={scenario === option.value}
              onClick={() => setScenario(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mobile-context">
        <p>资料与运行信息在窄屏下移至研究区之后。</p>
        <div>
          <button
            type="button"
            aria-label="打开资料区"
            aria-expanded={mobilePanel === "materials"}
            onClick={() => setMobilePanel(mobilePanel === "materials" ? null : "materials")}
          ><Files size={15} /> 资料</button>
          <button
            type="button"
            aria-label="打开运行区"
            aria-expanded={mobilePanel === "run"}
            onClick={() => setMobilePanel(mobilePanel === "run" ? null : "run")}
          ><Gauge size={15} /> 运行</button>
        </div>
        {mobilePanel && (
          <div className="mobile-context__panel">
            {mobilePanel === "materials"
              ? "2 条会话内 Mock 资料 · 当前为共享知识库"
              : `${view.statusLabel} · ${view.events.length} 个已观测事件`}
          </div>
        )}
      </div>

      <div className="workbench-grid" data-testid="workbench-shell">
        <MaterialsWorkspace />
        <ResearchWorkspace
          mode={mode}
          onModeChange={setMode}
          scenario={scenario}
          view={view}
        />
        <RunInspector view={view} />
      </div>
    </div>
  );
}
