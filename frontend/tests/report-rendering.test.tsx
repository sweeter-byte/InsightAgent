import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResearchWorkspace } from "@/components/workbench/research-workspace";
import { RunInspector } from "@/components/workbench/run-inspector";
import type { ResearchRunState } from "@/components/workbench/run-state";
import type { ActiveResearchRun } from "@/components/workbench/workbench-types";
import type { RunRecord } from "@/lib/protocol/types";

const selection: ActiveResearchRun = {
  runId: "report-run",
  source: "explicit",
  question: "Summarize the indexed material",
};

function record(finalOutput: string | null, status: RunRecord["status"] = "completed"): RunRecord {
  return {
    run_id: "report-run",
    thread_id: "report-thread",
    status,
    created_at: "2026-10-12T00:00:00Z",
    updated_at: "2026-10-12T00:01:00Z",
    error: status === "failed" ? "ResearchExecutionError" : null,
    final_output: finalOutput,
  };
}

function state(finalOutput: string | null, status: RunRecord["status"] = "completed"): ResearchRunState {
  return {
    runId: "report-run",
    generation: 1,
    record: record(finalOutput, status),
    observedStatus: status,
    connection: "connected",
    events: [],
    seenEventIds: new Set(),
    lastEventId: null,
    latestProgress: null,
    error: status === "failed" ? "ResearchExecutionError" : null,
  };
}

function renderReport(finalOutput: string | null, status: RunRecord["status"] = "completed") {
  const runState = state(finalOutput, status);
  return render(
    <>
      <ResearchWorkspace
        mode="research"
        onModeChange={() => undefined}
        activeRun={selection}
        runState={runState}
      />
      <RunInspector activeRun={selection} runState={runState} />
    </>,
  );
}

describe("authoritative Markdown report", () => {
  it("renders GFM structure, citations, safe links, and navigable sources", () => {
    renderReport(`# 研究结果

> 结论摘要

正文保留引用编号。[1]

- 第一项
- 第二项

| 路径 | 特点 |
| --- | --- |
| Research | 异步 |

\`\`\`ts
const result = "verified";
\`\`\`

## 参考来源

[1] Chapter 18 运行时说明，https://example.test/chapter-18
[2] 本地资料，段落 4`);

    const report = screen.getByRole("article", { name: "最终研究报告" });
    expect(within(report).getByRole("heading", { name: "研究结果" })).toBeInTheDocument();
    expect(within(report).getByText("结论摘要")).toBeInTheDocument();
    expect(within(report).getByText("正文保留引用编号。[1]")).toBeInTheDocument();
    expect(within(report).getAllByRole("listitem")).toHaveLength(2);
    expect(within(report).getByRole("table")).toBeInTheDocument();
    expect(within(report).getByText('const result = "verified";')).toBeInTheDocument();

    const externalLink = within(report).getByRole("link", {
      name: "https://example.test/chapter-18",
    });
    expect(externalLink).toHaveAttribute("target", "_blank");
    expect(externalLink).toHaveAttribute("rel", "noopener noreferrer");

    const sources = screen.getByRole("region", { name: "引用来源" });
    expect(within(sources).getByText("[1]")).toBeInTheDocument();
    expect(within(sources).getByText("Chapter 18 运行时说明")).toBeInTheDocument();
    expect(within(sources).getByText("[2]")).toBeInTheDocument();
    expect(within(sources).getByText("本地资料")).toBeInTheDocument();
    expect(within(sources).getAllByRole("link")[0]).toHaveAttribute(
      "href",
      "#report-references",
    );
    expect(within(report).getByRole("heading", { name: "参考来源" })).toHaveAttribute(
      "id",
      "report-references",
    );
  });

  it("does not mount arbitrary HTML or unsafe link targets", () => {
    const { container } = renderReport(`# 安全报告

<script>alert("bad")</script>
<img src="x" onerror="alert('bad')" />

[危险链接](javascript:alert('bad'))`);

    expect(container.querySelector("script")).not.toBeInTheDocument();
    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(container.querySelector('a[href^="javascript:"]')).not.toBeInTheDocument();
  });

  it("shows explicit empty, pending, and failed report states", () => {
    const { rerender } = render(
      <ResearchWorkspace
        mode="research"
        onModeChange={() => undefined}
        activeRun={selection}
        runState={state("")}
      />,
    );
    expect(screen.getByText("报告内容为空")).toBeInTheDocument();

    rerender(
      <ResearchWorkspace
        mode="research"
        onModeChange={() => undefined}
        activeRun={selection}
        runState={state(null, "running")}
      />,
    );
    expect(screen.getByText("最终报告尚未生成")).toBeInTheDocument();

    rerender(
      <ResearchWorkspace
        mode="research"
        onModeChange={() => undefined}
        activeRun={selection}
        runState={state(null, "failed")}
      />,
    );
    expect(screen.getByText("研究失败，未生成最终报告")).toBeInTheDocument();
  });

  it("clears the previous report and sources when the selected Run changes", () => {
    const oldState = state("# OLD REPORT\n\n## 参考来源\n\n[1] OLD SOURCE");
    const { rerender } = render(
      <>
        <ResearchWorkspace mode="research" onModeChange={() => undefined} activeRun={selection} runState={oldState} />
        <RunInspector activeRun={selection} runState={oldState} />
      </>,
    );
    expect(screen.getByRole("heading", { name: "OLD REPORT" })).toBeInTheDocument();
    expect(screen.getByText("OLD SOURCE")).toBeInTheDocument();

    const nextSelection: ActiveResearchRun = {
      runId: "next-run",
      source: "explicit",
      question: "Next question",
    };
    rerender(
      <>
        <ResearchWorkspace mode="research" onModeChange={() => undefined} activeRun={nextSelection} runState={oldState} />
        <RunInspector activeRun={nextSelection} runState={oldState} />
      </>,
    );

    expect(screen.queryByRole("heading", { name: "OLD REPORT" })).not.toBeInTheDocument();
    expect(screen.queryByText("OLD SOURCE")).not.toBeInTheDocument();
    expect(screen.getByText("最终报告尚未生成")).toBeInTheDocument();
  });
});
