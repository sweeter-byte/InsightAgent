import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";

describe("InsightWorkbench", () => {
  it("renders the three research regions with the real HTTP boundary", () => {
    render(<InsightWorkbench />);

    expect(screen.getByText("InsightAgent")).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "资料工作区" })).toBeInTheDocument();
    expect(screen.getByRole("main", { name: "研究工作区" })).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "运行检查器" })).toBeInTheDocument();
    expect(screen.getByText("共享知识库")).toBeInTheDocument();
    expect(screen.getByText("真实 HTTP 接口")).toBeInTheDocument();
    expect(screen.queryByText("Demo 状态预览")).not.toBeInTheDocument();
  });

  it("switches between smart query and deep research modes with buttons", async () => {
    const user = userEvent.setup();
    render(<InsightWorkbench />);

    const smartMode = screen.getByRole("button", { name: "智能提问" });
    const researchMode = screen.getByRole("button", { name: "深度研究" });
    expect(researchMode).toHaveAttribute("aria-pressed", "true");

    await user.click(smartMode);
    expect(smartMode).toHaveAttribute("aria-pressed", "true");
    expect(researchMode).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps upload interaction session-only and does not claim persistence", () => {
    render(<InsightWorkbench />);

    expect(screen.getByLabelText("选择研究资料"));
    expect(screen.getByText("资料记录仅保存在当前浏览器会话，刷新页面后会清空。"));
    expect(screen.queryByText("chapter18-runtime.md")).not.toBeInTheDocument();
  });

  it("renders mock scenarios only when demo mode is explicit", () => {
    render(<InsightWorkbench demoScenario="completed" />);

    expect(screen.getByText("Demo 状态预览")).toBeInTheDocument();
    expect(screen.getByText("Mock 内容")).toBeInTheDocument();
  });
});
