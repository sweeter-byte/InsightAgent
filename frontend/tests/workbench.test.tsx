import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";

describe("InsightWorkbench", () => {
  it("renders the three research regions and the phase-one boundary", () => {
    render(<InsightWorkbench />);

    expect(screen.getByText("InsightAgent")).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "资料工作区" })).toBeInTheDocument();
    expect(screen.getByRole("main", { name: "研究工作区" })).toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: "运行检查器" })).toBeInTheDocument();
    expect(screen.getByText("共享知识库")).toBeInTheDocument();
    expect(screen.getByText("Mock 协议预览")).toBeInTheDocument();
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
    expect(screen.getByText("仅为会话内 Mock 记录，不代表持久化资料列表。"));
  });
});
