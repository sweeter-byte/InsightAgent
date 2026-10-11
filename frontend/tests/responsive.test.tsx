import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";

describe("responsive workspace structure", () => {
  it("keeps research main first and exposes compact mobile rail summaries", () => {
    render(<InsightWorkbench />);

    const shell = screen.getByTestId("workbench-shell");
    expect(shell).toHaveClass("workbench-grid");
    expect(screen.getByTestId("research-workspace")).toHaveClass("research-main");
    expect(screen.getByText("资料与运行信息在窄屏下移至研究区之后。"))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: "打开资料区" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "打开运行区" })).toBeInTheDocument();
  });

  it("opens the compact materials context without exposing a fake API list", async () => {
    const user = userEvent.setup();
    render(<InsightWorkbench />);

    await user.click(screen.getByRole("button", { name: "打开资料区" }));
    expect(screen.getByText("会话资料仅在本页保留 · 当前为共享知识库"))
      .toBeInTheDocument();
  });
});
