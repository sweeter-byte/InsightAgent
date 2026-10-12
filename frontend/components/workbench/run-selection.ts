export const RESEARCH_RUN_STORAGE_KEY =
  "insightagent.current-research-run.v1";

export function readSelectedRunId(
  storage: Pick<Storage, "getItem">,
): string | null {
  try {
    const runId = storage.getItem(RESEARCH_RUN_STORAGE_KEY)?.trim();
    return runId || null;
  } catch {
    return null;
  }
}

export function storeSelectedRunId(
  runId: string,
  storage: Pick<Storage, "setItem">,
): void {
  try {
    storage.setItem(RESEARCH_RUN_STORAGE_KEY, runId);
  } catch {
    // Storage can be unavailable in privacy modes or blocked browser contexts.
  }
}

export function clearSelectedRunId(
  storage: Pick<Storage, "removeItem">,
): void {
  try {
    storage.removeItem(RESEARCH_RUN_STORAGE_KEY);
  } catch {
    // Losing persistence must not interrupt the active research session.
  }
}
