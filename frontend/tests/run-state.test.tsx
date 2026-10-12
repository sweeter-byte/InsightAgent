import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import { ResearchWorkspace } from "@/components/workbench/research-workspace";
import { RunInspector } from "@/components/workbench/run-inspector";
import type { ResearchRunState } from "@/components/workbench/run-state";
import type { ActiveResearchRun } from "@/components/workbench/workbench-types";
import type { RunRecord, SseEvent } from "@/lib/protocol/types";

const activeRun: ActiveResearchRun = {
  runId: "live-run",
  source: "explicit",
  question: "What happened in this run?",
};

function runRecord(
  status: RunRecord["status"],
  overrides: Partial<RunRecord> = {},
): RunRecord {
  return {
    run_id: "live-run",
    thread_id: "live-thread",
    status,
    created_at: "2026-10-12T00:00:00Z",
    updated_at: "2026-10-12T00:01:00Z",
    error: null,
    final_output: null,
    ...overrides,
  };
}

function liveState(
  status: RunRecord["status"],
  options: {
    connection?: ResearchRunState["connection"];
    events?: SseEvent[];
    latestProgress?: SseEvent["data"] | null;
    record?: RunRecord;
    error?: string | null;
  } = {},
): ResearchRunState {
  const events = options.events ?? [];
  return {
    runId: "live-run",
    generation: 1,
    record: options.record ?? runRecord(status),
    observedStatus: status,
    connection: options.connection ?? "connected",
    events,
    seenEventIds: new Set(events.map((event) => event.id)),
    lastEventId: events.at(-1)?.id ?? null,
    latestProgress: options.latestProgress ?? null,
    error: options.error ?? null,
  };
}

function renderLive(state: ResearchRunState, selection = activeRun) {
  return render(
    <>
      <ResearchWorkspace
        mode="research"
        onModeChange={() => undefined}
        activeRun={selection}
        runState={state}
      />
      <RunInspector activeRun={selection} runState={state} />
    </>,
  );
}

describe("run state presentation", () => {
  it("shows only observed coarse events while running", () => {
    render(<InsightWorkbench demoScenario="running" />);

    expect(screen.getAllByText("执行中").length).toBeGreaterThan(0);
    expect(screen.getByText("run.queued")).toBeInTheDocument();
    expect(screen.getByText("workflow.progress · planning")).toBeInTheDocument();
    expect(screen.getByText(/后端尚未提供节点级进度/)).toBeInTheDocument();
    expect(screen.queryByText(/证据评分中/)).not.toBeInTheDocument();
  });

  it("reveals the Markdown report and sources when completed", async () => {
    const user = userEvent.setup();
    render(<InsightWorkbench demoScenario="running" />);

    await user.click(screen.getByRole("button", { name: "完成" }));
    expect(screen.getAllByText("已完成").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "研究结果" })).toBeInTheDocument();
    expect(screen.getByText("[1] Chapter 18 运行时说明")).toBeInTheDocument();
  });

  it("shows error details without enabling resume for failed runs", () => {
    render(<InsightWorkbench demoScenario="failed" />);

    expect(screen.getAllByText("失败").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ResearchExecutionError").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "恢复运行" })).toBeDisabled();
  });

  it("enables the static resume affordance only for recoverable states", () => {
    render(<InsightWorkbench demoScenario="interrupted" />);

    expect(screen.getAllByText("可恢复").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "恢复运行" })).toBeEnabled();
    expect(screen.getByText(/本阶段不会发送 Resume 请求/)).toBeInTheDocument();
  });

  it("renders authoritative terminal output as escaped plain text and real events", () => {
    const events: SseEvent[] = [
      { id: "1-0", event: "run.queued", data: {} },
      { id: "2-0", event: "run.started", data: { resume: false } },
      {
        id: "3-0",
        event: "workflow.progress",
        data: { stage: "planning", message: "Building a search plan" },
      },
      { id: "4-0", event: "run.completed", data: { final_output: "present" } },
    ];
    renderLive(liveState("completed", {
      events,
      latestProgress: events[2]!.data,
      record: runRecord("completed", {
        final_output: "# Research Report\n\nVerified result.",
      }),
    }));

    expect(screen.getAllByText("研究已完成").length).toBeGreaterThan(0);
    expect(screen.getAllByText("SSE 已连接").length).toBeGreaterThan(0);
    expect(screen.getByText("workflow.progress · planning")).toBeInTheDocument();
    expect(screen.getAllByText("Building a search plan").length).toBeGreaterThan(0);
    expect(screen.getByText("最终报告可用")).toBeInTheDocument();
    expect(screen.getByText(/# Research Report/).tagName).toBe("PRE");
    expect(screen.queryByRole("heading", { name: "Research Report" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Chapter 18/)).not.toBeInTheDocument();
  });

  it("does not mistake a workflow stage named completed for terminal run status", () => {
    const progress = { id: "3-0", event: "workflow.progress", data: { stage: "completed" } };
    renderLive(liveState("running", {
      events: [progress],
      latestProgress: progress.data,
      record: runRecord("running"),
    }));

    expect(screen.getAllByText("研究运行中").length).toBeGreaterThan(0);
    expect(screen.getByText("workflow.progress · completed")).toBeInTheDocument();
    expect(screen.queryByText("研究已完成")).not.toBeInTheDocument();
    expect(screen.queryByText("最终报告可用")).not.toBeInTheDocument();
  });

  it.each([
    ["failed", "研究失败", false, "Worker exploded"],
    ["timed_out", "研究超时", true, null],
    ["interrupted", "研究已中断", true, null],
  ] as const)("presents %s honestly without wiring Resume HTTP", async (status, label, enabled, error) => {
    const user = userEvent.setup();
    const request = vi.spyOn(globalThis, "fetch");
    renderLive(liveState(status, {
      record: runRecord(status, { error }),
      error,
    }));

    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    if (error) expect(screen.getAllByText(error).length).toBeGreaterThan(0);
    const button = screen.getByRole("button", { name: "恢复运行" });
    expect(button).toHaveProperty("disabled", !enabled);
    if (enabled) await user.click(button);
    expect(request).not.toHaveBeenCalled();
    expect(screen.getByText(/未接入 Resume/)).toBeInTheDocument();
    request.mockRestore();
  });

  it("renders unknown events conservatively and never invents progress", () => {
    renderLive(liveState("running", {
      events: [{ id: "odd-1", event: "vendor.unknown", data: { percent: 92 } }],
    }));

    expect(screen.getByText("vendor.unknown")).toBeInTheDocument();
    expect(screen.getByText("收到后端事件")).toBeInTheDocument();
    expect(screen.queryByText(/92%|Planner Task|Evidence Grader|Retriever/)).not.toBeInTheDocument();
  });

  it("explains that restored selections cannot recover the submitted question", () => {
    renderLive(liveState("running"), {
      runId: "live-run",
      source: "restored",
      question: null,
    });

    expect(screen.getByText("重新挂载后任务文本不可用")).toBeInTheDocument();
  });

  it("never attributes stale observed state to a newly selected run", () => {
    renderLive(liveState("completed", {
      events: [{ id: "old-event", event: "run.completed", data: {} }],
      record: runRecord("completed", { final_output: "OLD REPORT" }),
    }), {
      runId: "new-run",
      source: "smart",
      question: "New question",
    });

    expect(screen.getAllByText("new-run").length).toBeGreaterThan(0);
    expect(screen.queryByText("OLD REPORT")).not.toBeInTheDocument();
    expect(screen.queryByText("old-event")).not.toBeInTheDocument();
    expect(screen.getByText("THREAD ID").nextElementSibling).toHaveTextContent("—");
    expect(screen.getByText("CONNECTION").nextElementSibling).toHaveTextContent("SSE 空闲");
    expect(screen.queryByText("研究已完成")).not.toBeInTheDocument();
  });
});
