import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import { RESEARCH_RUN_STORAGE_KEY } from "@/components/workbench/run-selection";
import {
  completedRunFixture,
  interruptedRunFixture,
  queuedRunFixture,
} from "@/lib/protocol/fixtures";

import { mockServer } from "./mocks/server";

const runEndpoint = "http://localhost:3000/v1/research/runs/demo-run";
const resumeEndpoint = `${runEndpoint}/resume`;
const eventsEndpoint = `${runEndpoint}/events`;

function eventStream(frame: string) {
  return new HttpResponse(frame, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function renderInterruptedRun() {
  sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "demo-run");
  const user = userEvent.setup();
  render(<InsightWorkbench />);
  expect((await screen.findAllByText("研究已中断")).length).toBeGreaterThan(0);
  return user;
}

describe("Resume integration", () => {
  beforeEach(() => {
    sessionStorage.clear();
    mockServer.use(
      http.get(runEndpoint, () => HttpResponse.json(interruptedRunFixture)),
    );
  });

  it("resumes the same Run once, disables while pending, and restarts observation", async () => {
    let resumeCalls = 0;
    let eventCalls = 0;
    let getCalls = 0;
    mockServer.use(
      http.get(runEndpoint, () => {
        getCalls += 1;
        return HttpResponse.json(getCalls === 1 ? interruptedRunFixture : completedRunFixture);
      }),
      http.post(resumeEndpoint, async () => {
        resumeCalls += 1;
        await delay(100);
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
      http.get(eventsEndpoint, () => {
        eventCalls += 1;
        return eventStream("id: resumed-complete\nevent: run.completed\ndata: {}\n\n");
      }),
    );
    const user = await renderInterruptedRun();
    const button = screen.getByRole("button", { name: "恢复运行" });

    const firstClick = user.click(button);
    expect(await screen.findByRole("button", { name: "正在恢复…" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "正在恢复…" }));
    await firstClick;

    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("demo-run").length).toBeGreaterThan(0);
    expect(await screen.findByRole("heading", { name: "Research Report" })).toBeInTheDocument();
    expect(resumeCalls).toBe(1);
    expect(eventCalls).toBe(1);
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("demo-run");
  });

  it("clears a missing Run after Resume returns 404", async () => {
    mockServer.use(
      http.post(resumeEndpoint, () =>
        HttpResponse.json({ detail: "missing" }, { status: 404 }),
      ),
    );
    const user = await renderInterruptedRun();

    await user.click(screen.getByRole("button", { name: "恢复运行" }));

    expect((await screen.findAllByText("运行不存在")).length).toBeGreaterThan(0);
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBeNull();
  });

  it.each([
    [409, "当前状态已不允许恢复，请重新检查运行状态。"],
    [0, "无法连接 InsightAgent 后端，请检查服务后重试。"],
  ] as const)("shows a scoped error for Resume failure %s", async (status, message) => {
    mockServer.use(
      http.post(resumeEndpoint, () =>
        status === 0
          ? HttpResponse.error()
          : HttpResponse.json({ detail: "resume rejected" }, { status }),
      ),
    );
    const user = await renderInterruptedRun();

    await user.click(screen.getByRole("button", { name: "恢复运行" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(screen.getByRole("button", { name: "恢复运行" })).toBeEnabled();
  });

  it("never enables Resume for a completed Run", async () => {
    sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "demo-run");
    mockServer.use(
      http.get(runEndpoint, () => HttpResponse.json(completedRunFixture)),
    );
    render(<InsightWorkbench />);

    await screen.findAllByText("研究已完成");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "恢复运行" })).toBeDisabled(),
    );
  });
});
