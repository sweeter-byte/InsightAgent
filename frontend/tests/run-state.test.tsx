import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";

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
});
