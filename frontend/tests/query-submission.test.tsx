import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import { RESEARCH_RUN_STORAGE_KEY } from "@/components/workbench/run-selection";
import { api } from "@/lib/api-client";
import {
  completedRunFixture,
  analyzeQueryResponseFixture,
  directQueryResponseFixture,
  queuedRunFixture,
  researchQueryResponseFixture,
} from "@/lib/protocol/fixtures";

import { mockServer } from "./mocks/server";

const queryEndpoint = "http://localhost:3000/v1/query";
const runEndpoint = "http://localhost:3000/v1/research/runs";
const selectedRunEndpoint = `${runEndpoint}/demo-run`;
const eventsEndpoint = `${selectedRunEndpoint}/events`;

function eventStream(...frames: string[]) {
  const encoder = new TextEncoder();
  return new HttpResponse(
    new ReadableStream({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

async function renderReadyWorkbench() {
  const user = userEvent.setup();
  const result = render(<InsightWorkbench />);
  await screen.findByText("后端已就绪");
  return { user, ...result };
}

async function replaceQuestion(user: ReturnType<typeof userEvent.setup>, value: string) {
  const input = screen.getByLabelText("研究问题");
  await user.clear(input);
  await user.type(input, value);
  return input;
}

describe("query submission", () => {
  beforeEach(() => sessionStorage.clear());

  it.each([
    ["direct", directQueryResponseFixture, "direct answer"],
    ["analyze", analyzeQueryResponseFixture, "analyze answer"],
  ])("renders a completed %s smart-query answer", async (_intent, response, answer) => {
    let body: unknown;
    mockServer.use(
      http.post(queryEndpoint, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(response);
      }),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "智能提问" }));
    await replaceQuestion(user, "  explain the material  ");

    await user.click(screen.getByRole("button", { name: "提交智能提问" }));

    expect(await screen.findByText(answer)).toBeInTheDocument();
    expect(screen.getByText(`INTENT · ${response.intent}`)).toBeInTheDocument();
    expect(body).toEqual({ query: "explain the material" });
    expect(screen.queryByText("任务已提交")).not.toBeInTheDocument();
  });

  it("fetches the full run after smart Research returns only a run_id", async () => {
    let getCalls = 0;
    let eventCalls = 0;
    mockServer.use(
      http.post(queryEndpoint, () =>
        HttpResponse.json(researchQueryResponseFixture, { status: 202 }),
      ),
      http.get(selectedRunEndpoint, () => {
        getCalls += 1;
        return HttpResponse.json({ ...queuedRunFixture, status: getCalls > 1 ? "completed" : "running" });
      }),
      http.get(eventsEndpoint, () => {
        eventCalls += 1;
        return eventStream("id: smart-complete\nevent: run.completed\ndata: {}\n\n");
      }),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "智能提问" }));

    await user.click(screen.getByRole("button", { name: "提交智能提问" }));

    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("demo-run").length).toBeGreaterThan(0);
    expect(screen.getByText("THREAD ID").nextElementSibling).toHaveTextContent("demo-thread");
    expect(getCalls).toBeGreaterThanOrEqual(2);
    expect(eventCalls).toBe(1);
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("demo-run");
  });

  it("uses the explicit run endpoint and preserves returned run metadata", async () => {
    let body: unknown;
    let queryCalls = 0;
    let eventCalls = 0;
    mockServer.use(
      http.post(queryEndpoint, () => {
        queryCalls += 1;
        return HttpResponse.json(researchQueryResponseFixture, { status: 202 });
      }),
      http.post(runEndpoint, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
      http.get(eventsEndpoint, () => {
        eventCalls += 1;
        return new Promise<Response>(() => undefined);
      }),
    );
    const { user } = await renderReadyWorkbench();
    await replaceQuestion(user, "compare systems");

    await user.click(screen.getByRole("button", { name: "提交深度研究" }));

    expect((await screen.findAllByText("任务已排队")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("demo-run").length).toBeGreaterThan(0);
    expect(screen.getAllByText("demo-thread").length).toBeGreaterThan(0);
    expect(screen.getAllByText("queued").length).toBeGreaterThan(0);
    expect(body).toEqual({ query: "compare systems" });
    expect(queryCalls).toBe(0);
    await waitFor(() => expect(eventCalls).toBe(1));
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("demo-run");
  });

  it.each(["direct", "analyze"] as const)(
    "clears a persisted Research selection after a %s answer",
    async (intent) => {
      sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "stale-run");
      mockServer.use(
        http.get(`${runEndpoint}/stale-run`, () =>
          HttpResponse.json({ detail: "missing" }, { status: 404 }),
        ),
        http.post(queryEndpoint, () =>
          HttpResponse.json(
            intent === "direct" ? directQueryResponseFixture : analyzeQueryResponseFixture,
          ),
        ),
      );
      const { user } = await renderReadyWorkbench();
      await user.click(screen.getByRole("button", { name: "智能提问" }));
      await user.click(screen.getByRole("button", { name: "提交智能提问" }));

      await screen.findByText(`${intent} answer`);
      expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBeNull();
    },
  );

  it("restores one persisted run after mount and resumes GET plus SSE observation", async () => {
    sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "demo-run");
    let getCalls = 0;
    let eventCalls = 0;
    mockServer.use(
      http.get(selectedRunEndpoint, () => {
        getCalls += 1;
        return HttpResponse.json(getCalls === 1 ? queuedRunFixture : completedRunFixture);
      }),
      http.get(eventsEndpoint, () => {
        eventCalls += 1;
        return eventStream("id: restored-complete\nevent: run.completed\ndata: {}\n\n");
      }),
    );

    render(<InsightWorkbench />);

    expect(await screen.findByText("重新挂载后任务文本不可用")).toBeInTheDocument();
    expect((await screen.findAllByText("研究已完成")).length).toBeGreaterThan(0);
    expect(getCalls).toBeGreaterThanOrEqual(2);
    expect(eventCalls).toBe(1);
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBe("demo-run");
  });

  it("clears a missing restored run while retaining a visible not-found state", async () => {
    sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "missing-run");
    mockServer.use(
      http.get(`${runEndpoint}/missing-run`, () =>
        HttpResponse.json({ detail: "missing" }, { status: 404 }),
      ),
    );

    render(<InsightWorkbench />);

    expect((await screen.findAllByText("运行不存在")).length).toBeGreaterThan(0);
    expect(sessionStorage.getItem(RESEARCH_RUN_STORAGE_KEY)).toBeNull();
  });

  it("clears a stale not-found presentation when a replacement submission starts", async () => {
    sessionStorage.setItem(RESEARCH_RUN_STORAGE_KEY, "missing-run");
    mockServer.use(
      http.get(`${runEndpoint}/missing-run`, () =>
        HttpResponse.json({ detail: "missing" }, { status: 404 }),
      ),
      http.post(runEndpoint, async () => {
        await delay(100);
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
      http.get(eventsEndpoint, () => new Promise<Response>(() => undefined)),
    );
    const { user } = await renderReadyWorkbench();
    await screen.findAllByText("运行不存在");

    void user.click(screen.getByRole("button", { name: "提交深度研究" }));
    await screen.findByText("正在提交 Research Run…");

    expect(screen.queryByText("运行不存在")).not.toBeInTheDocument();
  });

  it("shows a protocol error for malformed SSE without changing the run to failed", async () => {
    mockServer.use(
      http.post(runEndpoint, () => HttpResponse.json(queuedRunFixture, { status: 202 })),
      http.get(eventsEndpoint, () => eventStream("event: run.started\ndata: {}\n\n")),
      http.get(selectedRunEndpoint, () => HttpResponse.json(completedRunFixture)),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "提交深度研究" }));

    expect((await screen.findAllByText("SSE 协议错误")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("研究已完成").length).toBeGreaterThan(0);
    expect(screen.queryByText("研究失败")).not.toBeInTheDocument();
  });

  it("calibrates a network-disconnected stream with terminal GET and does not reconnect", async () => {
    let eventCalls = 0;
    mockServer.use(
      http.post(runEndpoint, () => HttpResponse.json(queuedRunFixture, { status: 202 })),
      http.get(eventsEndpoint, () => {
        eventCalls += 1;
        return HttpResponse.error();
      }),
      http.get(selectedRunEndpoint, () => HttpResponse.json(completedRunFixture)),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "提交深度研究" }));

    expect((await screen.findAllByText("SSE 已断开")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("研究已完成").length).toBeGreaterThan(0);
    await new Promise((resolve) => setTimeout(resolve, 550));
    expect(eventCalls).toBe(1);
  });

  it.each([
    [502, "unable to route query", "上游模型未能完成本次请求，请重试。"],
    [503, "research runtime unavailable", "研究运行服务暂不可用，请稍后重试。"],
  ])("maps query HTTP %s and preserves the question", async (status, detail, message) => {
    mockServer.use(
      http.post(queryEndpoint, () => HttpResponse.json({ detail }, { status })),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "智能提问" }));
    const input = await replaceQuestion(user, "keep this question");

    await user.click(screen.getByRole("button", { name: "提交智能提问" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(input).toHaveValue("keep this question");
  });

  it("distinguishes network and protocol failures", async () => {
    let networkFailure = true;
    mockServer.use(
      http.post(queryEndpoint, () =>
        networkFailure
          ? HttpResponse.error()
          : HttpResponse.json({ request_id: "incomplete" }),
      ),
    );
    const { user } = await renderReadyWorkbench();
    await user.click(screen.getByRole("button", { name: "智能提问" }));
    const button = screen.getByRole("button", { name: "提交智能提问" });

    await user.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "无法连接 InsightAgent 后端，请检查服务后重试。",
    );

    networkFailure = false;
    await user.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "后端返回了无法识别的响应，请检查前后端版本。",
    );
  });

  it("disables submission and prevents duplicate requests while pending", async () => {
    let calls = 0;
    mockServer.use(
      http.post(runEndpoint, async () => {
        calls += 1;
        await delay(100);
        return HttpResponse.json(queuedRunFixture, { status: 202 });
      }),
    );
    const { user } = await renderReadyWorkbench();
    const button = screen.getByRole("button", { name: "提交深度研究" });

    const first = user.click(button);
    await screen.findByText("正在提交 Research Run…");
    expect(button).toBeDisabled();
    await user.click(button);
    await first;
    await screen.findAllByText("研究已完成");

    expect(calls).toBe(1);
  });

  it("aborts a pending submission when unmounted", async () => {
    let aborted = false;
    const createRun = vi.spyOn(api, "createResearchRun").mockImplementation(
      (_query, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            aborted = true;
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { user, unmount } = await renderReadyWorkbench();
    void user.click(screen.getByRole("button", { name: "提交深度研究" }));
    await screen.findByText("正在提交 Research Run…");

    unmount();

    await waitFor(() => expect(aborted).toBe(true));
    createRun.mockRestore();
  });

  it("does not fabricate SSE events, citations, or a completed report", async () => {
    render(<InsightWorkbench />);
    await screen.findByText("后端已就绪");

    expect(screen.queryByText("run.started")).not.toBeInTheDocument();
    expect(screen.queryByText("[1] Chapter 18 运行时说明")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "研究结果" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Mock 内容/)).not.toBeInTheDocument();
  });
});
