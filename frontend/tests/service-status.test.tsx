import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import { api } from "@/lib/api-client";
import {
  healthResponseFixture,
  notReadyResponseFixture,
  readinessResponseFixture,
} from "@/lib/protocol/fixtures";

import { mockServer } from "./mocks/server";

const appBase = "http://localhost:3000";

describe("backend service status", () => {
  it("shows a live process and ready required services without treating optional providers as failures", async () => {
    render(<InsightWorkbench />);

    expect(await screen.findByText("后端已就绪")).toBeInTheDocument();
    expect(screen.getByText("后端连接正常")).toBeInTheDocument();
    expect(screen.getByText("vision")).toBeInTheDocument();
    expect(screen.getAllByText("未配置").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "提交深度研究" })).toBeEnabled();
  });

  it("parses a 503 readiness body and blocks doomed mutations", async () => {
    mockServer.use(
      http.get(`${appBase}/ready`, () =>
        HttpResponse.json(notReadyResponseFixture, { status: 503 }),
      ),
    );
    render(<InsightWorkbench />);

    expect(await screen.findByText("后端未就绪")).toBeInTheDocument();
    expect(screen.getByText("后端连接正常")).toBeInTheDocument();
    expect(screen.getByText("redis unavailable")).toBeInTheDocument();
    expect(screen.getByLabelText("选择研究资料")).toBeDisabled();
    expect(screen.getByRole("button", { name: "提交深度研究" })).toBeDisabled();
  });

  it("does not disguise a network failure as readiness", async () => {
    mockServer.use(
      http.get(`${appBase}/health`, () => HttpResponse.error()),
      http.get(`${appBase}/ready`, () => HttpResponse.error()),
    );
    render(<InsightWorkbench />);

    expect(await screen.findByText("后端不可达")).toBeInTheDocument();
    expect(screen.queryByText("后端已就绪")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "提交深度研究" })).toBeDisabled();
  });

  it("allows a failed probe to be retried", async () => {
    let available = false;
    mockServer.use(
      http.get(`${appBase}/health`, () =>
        available ? HttpResponse.json(healthResponseFixture) : HttpResponse.error(),
      ),
      http.get(`${appBase}/ready`, () =>
        available
          ? HttpResponse.json(readinessResponseFixture)
          : HttpResponse.error(),
      ),
    );
    const user = userEvent.setup();
    render(<InsightWorkbench />);
    expect(await screen.findByText("后端不可达")).toBeInTheDocument();

    available = true;
    await user.click(screen.getByRole("button", { name: "重新检查" }));

    expect(await screen.findByText("后端已就绪")).toBeInTheDocument();
  });

  it("aborts both probes when the workbench unmounts", async () => {
    const aborted = new Set<string>();
    const health = vi.spyOn(api, "health").mockImplementation(
      (signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            aborted.add("health");
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const readiness = vi.spyOn(api, "readiness").mockImplementation(
      (signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            aborted.add("readiness");
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { unmount } = render(<InsightWorkbench />);
    expect(await screen.findByText("正在检查后端")).toBeInTheDocument();

    unmount();

    await vi.waitFor(() => expect(aborted).toEqual(new Set(["health", "readiness"])));
    health.mockRestore();
    readiness.mockRestore();
  });
});
