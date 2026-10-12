import { describe, expect, it, vi } from "vitest";

import {
  clearSelectedRunId,
  readSelectedRunId,
  RESEARCH_RUN_STORAGE_KEY,
  storeSelectedRunId,
} from "@/components/workbench/run-selection";

describe("research run session selection", () => {
  it("stores only the selected run id", () => {
    const storage = { setItem: vi.fn() } as unknown as Storage;

    storeSelectedRunId("run-1", storage);

    expect(storage.setItem).toHaveBeenCalledWith(
      RESEARCH_RUN_STORAGE_KEY,
      "run-1",
    );
    expect(storage.setItem).toHaveBeenCalledTimes(1);
  });

  it.each([
    [" run-1 ", "run-1"],
    ["   ", null],
    ["", null],
    [null, null],
  ])("trims a stored selection %#", (stored, expected) => {
    const storage = {
      getItem: vi.fn(() => stored),
    } as unknown as Storage;

    expect(readSelectedRunId(storage)).toBe(expected);
  });

  it("clears the selected run id", () => {
    const storage = { removeItem: vi.fn() } as unknown as Storage;

    clearSelectedRunId(storage);

    expect(storage.removeItem).toHaveBeenCalledWith(RESEARCH_RUN_STORAGE_KEY);
  });

  it("silently degrades when storage is blocked", () => {
    const blocked = {
      getItem: vi.fn(() => {
        throw new DOMException("blocked", "SecurityError");
      }),
      setItem: vi.fn(() => {
        throw new DOMException("blocked", "SecurityError");
      }),
      removeItem: vi.fn(() => {
        throw new DOMException("blocked", "SecurityError");
      }),
    } as unknown as Storage;

    expect(readSelectedRunId(blocked)).toBeNull();
    expect(() => storeSelectedRunId("run-1", blocked)).not.toThrow();
    expect(() => clearSelectedRunId(blocked)).not.toThrow();
  });
});
