import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import { InsightWorkbench } from "@/components/workbench/insight-workbench";
import { api } from "@/lib/api-client";
import { materialImportResultFixture } from "@/lib/protocol/fixtures";

import { mockServer } from "./mocks/server";

const endpoint = "http://localhost:3000/v1/materials";

async function renderReadyWorkbench() {
  const user = userEvent.setup();
  const result = render(<InsightWorkbench />);
  await screen.findByText("后端已就绪");
  return { user, ...result };
}

describe("material upload", () => {
  it("uploads one file and records a first import only in session state", async () => {
    let keys: string[] = [];
    mockServer.use(
      http.post(endpoint, async ({ request }) => {
        keys = [...(await request.formData()).keys()];
        return HttpResponse.json(materialImportResultFixture, { status: 201 });
      }),
    );
    const { user } = await renderReadyWorkbench();

    await user.upload(
      screen.getByLabelText("选择研究资料"),
      new File(["# fixed"], "fixed.md", { type: "text/markdown" }),
    );

    expect(await screen.findByText("已导入共享知识库")).toBeInTheDocument();
    expect(screen.getByText("fixed.md")).toBeInTheDocument();
    expect(screen.getByText(/^2 chunks/)).toBeInTheDocument();
    expect(screen.getByText("首次索引")).toBeInTheDocument();
    expect(keys).toEqual(["file"]);
    expect(screen.getByText(/刷新页面后会清空/)).toBeInTheDocument();
  });

  it("uses distinct copy for a deduplicated 200 response", async () => {
    mockServer.use(
      http.post(endpoint, () =>
        HttpResponse.json(
          { ...materialImportResultFixture, deduplicated: true },
          { status: 200 },
        ),
      ),
    );
    const { user } = await renderReadyWorkbench();

    await user.upload(
      screen.getByLabelText("选择研究资料"),
      new File(["same"], "same.md", { type: "text/markdown" }),
    );

    expect(await screen.findByText("已存在，完成去重导入")).toBeInTheDocument();
    expect(screen.getByText("去重索引")).toBeInTheDocument();
  });

  it.each([
    [413, "material upload too large", "文件超过后端允许的上传大小。"],
    [422, "invalid material upload", "资料格式无效、内容为空或无法解析。"],
    [422, "image ingestion unavailable; configure VISION_*", "图片解析不可用，请先配置 Vision Provider。"],
    [503, "knowledge service unavailable", "知识服务暂不可用，请稍后重试。"],
  ])("maps upload HTTP %s to a safe explanation", async (status, detail, message) => {
    mockServer.use(
      http.post(endpoint, () => HttpResponse.json({ detail }, { status })),
    );
    const { user } = await renderReadyWorkbench();

    await user.upload(
      screen.getByLabelText("选择研究资料"),
      new File(["bad"], "bad.md", { type: "text/markdown" }),
    );

    expect(await screen.findByText(message)).toBeInTheDocument();
    expect(screen.queryByText("bad.md")).not.toBeInTheDocument();
    expect(screen.getByLabelText("选择研究资料")).toBeEnabled();
  });

  it("distinguishes network and protocol failures", async () => {
    let networkFailure = true;
    mockServer.use(
      http.post(endpoint, () =>
        networkFailure
          ? HttpResponse.error()
          : HttpResponse.json({ material_id: "incomplete" }, { status: 201 }),
      ),
    );
    const { user } = await renderReadyWorkbench();
    const input = screen.getByLabelText("选择研究资料");

    await user.upload(input, new File(["one"], "one.md"));
    expect(
      await screen.findByText("无法连接 InsightAgent 后端，请检查服务后重试。"),
    ).toBeInTheDocument();

    networkFailure = false;
    await user.upload(input, new File(["two"], "two.md"));
    expect(
      await screen.findByText("后端返回了无法识别的响应，请检查前后端版本。"),
    ).toBeInTheDocument();
  });

  it("disables the input and prevents duplicate submission while processing", async () => {
    let requests = 0;
    mockServer.use(
      http.post(endpoint, async () => {
        requests += 1;
        await delay(100);
        return HttpResponse.json(materialImportResultFixture, { status: 201 });
      }),
    );
    const { user } = await renderReadyWorkbench();
    const input = screen.getByLabelText("选择研究资料");
    const file = new File(["fixed"], "fixed.md");

    const first = user.upload(input, file);
    await screen.findByText("正在上传并建立索引…");
    expect(input).toBeDisabled();
    await user.upload(input, file);
    await first;
    await screen.findByText("已导入共享知识库");

    expect(requests).toBe(1);
  });

  it("aborts an in-flight upload when the workbench unmounts", async () => {
    let aborted = false;
    const upload = vi.spyOn(api, "uploadMaterial").mockImplementation(
      (_file, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => {
            aborted = true;
            reject(new DOMException("Aborted", "AbortError"));
          });
        }),
    );
    const { user, unmount } = await renderReadyWorkbench();
    void user.upload(
      screen.getByLabelText("选择研究资料"),
      new File(["pending"], "pending.md"),
    );
    await screen.findByText("正在上传并建立索引…");

    unmount();

    await waitFor(() => expect(aborted).toBe(true));
    upload.mockRestore();
  });
});
