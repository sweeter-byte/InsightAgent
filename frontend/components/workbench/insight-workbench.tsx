"use client";

import { Files, Gauge } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiHttpError,
  ApiProtocolError,
  api,
  describeApiError,
} from "@/lib/api-client";
import type { ComponentReadiness } from "@/lib/protocol/types";

import { MaterialsWorkspace } from "./materials-workspace";
import {
  scenarioOptions,
  scenarioViews,
  type PreviewScenario,
} from "./mock-data";
import { ResearchWorkspace } from "./research-workspace";
import { RunInspector } from "./run-inspector";
import {
  clearSelectedRunId,
  readSelectedRunId,
  storeSelectedRunId,
} from "./run-selection";
import { useResearchRun } from "./use-research-run";
import type {
  ActiveResearchRun,
  AsyncPhase,
  ConnectionState,
  SessionMaterial,
  SynchronousAnswer,
} from "./workbench-types";

const initialConnection: ConnectionState = {
  phase: "checking",
  healthAlive: false,
  readiness: null,
  error: null,
};

function browserSessionStorage(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function InsightWorkbench({
  demoScenario,
}: {
  demoScenario?: PreviewScenario;
}) {
  const initialDemoScenario = demoScenario;
  const [scenario, setScenario] = useState<PreviewScenario>(
    initialDemoScenario ?? "idle",
  );
  const [mode, setMode] = useState<"smart" | "research">("research");
  const [question, setQuestion] = useState(
    "比较 Direct、Analyze 与多步 Research 三种路径，并说明它们各自适用的任务边界。",
  );
  const [mobilePanel, setMobilePanel] = useState<"materials" | "run" | null>(null);
  const [connection, setConnection] = useState<ConnectionState>(initialConnection);
  const [materials, setMaterials] = useState<SessionMaterial[]>([]);
  const [uploadPhase, setUploadPhase] = useState<AsyncPhase>("idle");
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [submissionPhase, setSubmissionPhase] = useState<AsyncPhase>("idle");
  const [submissionError, setSubmissionError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<SynchronousAnswer | null>(null);
  const [activeRun, setActiveRun] = useState<ActiveResearchRun | null>(null);
  const [missingRunId, setMissingRunId] = useState<string | null>(null);
  const [resumePhase, setResumePhase] = useState<AsyncPhase>("idle");
  const [resumeError, setResumeError] = useState<string | null>(null);
  const connectionController = useRef<AbortController | null>(null);
  const connectionSequence = useRef(0);
  const uploadController = useRef<AbortController | null>(null);
  const uploadSequence = useRef(0);
  const uploadInFlight = useRef(false);
  const submissionController = useRef<AbortController | null>(null);
  const submissionSequence = useRef(0);
  const submissionInFlight = useRef(false);
  const resumeController = useRef<AbortController | null>(null);
  const resumeSequence = useRef(0);
  const resumeInFlight = useRef(false);
  const view = scenarioViews[scenario];
  const isDemo = initialDemoScenario !== undefined;
  const canMutate = connection.phase === "ready";
  const handleRunNotFound = useCallback((runId: string) => {
    setActiveRun((current) => {
      if (current?.runId !== runId) return current;
      const storage = browserSessionStorage();
      if (storage) clearSelectedRunId(storage);
      setMissingRunId(runId);
      return null;
    });
  }, []);
  const runState = useResearchRun(isDemo ? null : activeRun, {
    onNotFound: handleRunNotFound,
  });

  const checkConnection = useCallback(() => {
    setConnection(initialConnection);
    connectionController.current?.abort();
    const controller = new AbortController();
    connectionController.current = controller;
    const sequence = ++connectionSequence.current;
    void readConnectionState(controller.signal).then((next) => {
      if (!controller.signal.aborted && sequence === connectionSequence.current) {
        setConnection(next);
      }
    });
  }, []);

  useEffect(() => {
    if (isDemo) return;
    connectionController.current?.abort();
    const controller = new AbortController();
    connectionController.current = controller;
    const sequence = ++connectionSequence.current;
    void readConnectionState(controller.signal).then((next) => {
      if (!controller.signal.aborted && sequence === connectionSequence.current) {
        setConnection(next);
      }
    });
    return () => {
      connectionSequence.current += 1;
      connectionController.current?.abort();
    };
  }, [isDemo]);

  useEffect(() => {
    if (isDemo) return;
    const storage = browserSessionStorage();
    if (!storage) return;
    const runId = readSelectedRunId(storage);
    if (!runId) return;
    const restore = window.setTimeout(() => {
      setActiveRun({ runId, source: "restored", question: null });
    }, 0);
    return () => window.clearTimeout(restore);
  }, [isDemo]);

  useEffect(
    () => () => {
      uploadSequence.current += 1;
      uploadController.current?.abort();
      submissionSequence.current += 1;
      submissionController.current?.abort();
      resumeSequence.current += 1;
      resumeController.current?.abort();
    },
    [],
  );

  const uploadMaterial = useCallback(async (file: File) => {
    if (uploadInFlight.current) return;
    uploadInFlight.current = true;
    uploadController.current?.abort();
    const controller = new AbortController();
    uploadController.current = controller;
    const sequence = ++uploadSequence.current;
    setUploadPhase("submitting");
    setUploadMessage(null);
    try {
      const result = await api.uploadMaterial(file, controller.signal);
      if (controller.signal.aborted || sequence !== uploadSequence.current) return;
      setMaterials((current) => [
        ...current,
        { fileName: file.name, ...result },
      ]);
      setUploadPhase("succeeded");
      setUploadMessage(
        result.deduplicated
          ? "已存在，完成去重导入"
          : "已导入共享知识库",
      );
    } catch (error) {
      if (controller.signal.aborted || sequence !== uploadSequence.current) return;
      setUploadPhase("failed");
      setUploadMessage(describeApiError(error, "upload"));
    } finally {
      if (sequence === uploadSequence.current) uploadInFlight.current = false;
    }
  }, []);

  const submitQuestion = useCallback(async () => {
    const normalizedQuestion = question.trim();
    if (!normalizedQuestion || submissionInFlight.current || !canMutate) return;
    submissionInFlight.current = true;
    submissionController.current?.abort();
    const controller = new AbortController();
    submissionController.current = controller;
    const sequence = ++submissionSequence.current;
    setSubmissionPhase("submitting");
    setSubmissionError(null);
    setMissingRunId(null);
    setResumePhase("idle");
    setResumeError(null);
    try {
      if (mode === "smart") {
        const result = await api.query(normalizedQuestion, controller.signal);
        if (controller.signal.aborted || sequence !== submissionSequence.current) return;
        if (result.intent === "research") {
          if (result.status !== "queued" || !result.run_id) {
            throw new ApiProtocolError();
          }
          const selection: ActiveResearchRun = {
            runId: result.run_id,
            source: "smart",
            question: normalizedQuestion,
          };
          setMissingRunId(null);
          setActiveRun(selection);
          const storage = browserSessionStorage();
          if (storage) storeSelectedRunId(selection.runId, storage);
          setAnswer(null);
        } else {
          if (result.status !== "completed" || result.answer === null) {
            throw new ApiProtocolError();
          }
          setAnswer({ intent: result.intent, text: result.answer });
          setActiveRun(null);
          setMissingRunId(null);
          const storage = browserSessionStorage();
          if (storage) clearSelectedRunId(storage);
        }
      } else {
        const result = await api.createResearchRun(
          normalizedQuestion,
          controller.signal,
        );
        if (controller.signal.aborted || sequence !== submissionSequence.current) return;
        const selection: ActiveResearchRun = {
          runId: result.run_id,
          source: "explicit",
          question: normalizedQuestion,
          initialRecord: result,
        };
        setMissingRunId(null);
        setActiveRun(selection);
        const storage = browserSessionStorage();
        if (storage) storeSelectedRunId(selection.runId, storage);
        setAnswer(null);
      }
      setSubmissionPhase("succeeded");
    } catch (error) {
      if (controller.signal.aborted || sequence !== submissionSequence.current) return;
      setSubmissionPhase("failed");
      setSubmissionError(describeApiError(error, "query"));
    } finally {
      if (sequence === submissionSequence.current) submissionInFlight.current = false;
    }
  }, [canMutate, mode, question]);

  const resumeRun = useCallback(async () => {
    if (!activeRun || resumeInFlight.current) return;
    const scopedState = runState.runId === activeRun.runId ? runState : null;
    const status = scopedState?.observedStatus ?? scopedState?.record?.status ?? null;
    if (status !== "interrupted" && status !== "timed_out") return;

    resumeInFlight.current = true;
    resumeController.current?.abort();
    const controller = new AbortController();
    resumeController.current = controller;
    const sequence = ++resumeSequence.current;
    const runId = activeRun.runId;
    setResumePhase("submitting");
    setResumeError(null);
    try {
      const resumed = await api.resumeResearchRun(runId, controller.signal);
      if (controller.signal.aborted || sequence !== resumeSequence.current) return;
      if (resumed.run_id !== runId) throw new ApiProtocolError();
      setActiveRun((current) =>
        current?.runId === runId ? { ...current, initialRecord: resumed } : current,
      );
      setResumePhase("succeeded");
    } catch (error) {
      if (controller.signal.aborted || sequence !== resumeSequence.current) return;
      setResumePhase("failed");
      setResumeError(describeApiError(error, "resume"));
      if (error instanceof ApiHttpError && error.status === 404) {
        handleRunNotFound(runId);
      }
    } finally {
      if (sequence === resumeSequence.current) resumeInFlight.current = false;
    }
  }, [activeRun, handleRunNotFound, runState]);

  return (
    <div className="workbench-frame">
      {isDemo ? (
        <div className="preview-bar">
          <div>
            <span>DEMO</span>
            <strong>Demo 状态预览</strong>
          </div>
          <div className="preview-switch" role="group" aria-label="Mock 运行状态">
            {scenarioOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                aria-pressed={scenario === option.value}
                onClick={() => setScenario(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <ServiceStatus connection={connection} onRetry={checkConnection} />
      )}

      <div className="mobile-context">
        <p>资料与运行信息在窄屏下移至研究区之后。</p>
        <div>
          <button
            type="button"
            aria-label="打开资料区"
            aria-expanded={mobilePanel === "materials"}
            onClick={() => setMobilePanel(mobilePanel === "materials" ? null : "materials")}
          ><Files size={15} /> 资料</button>
          <button
            type="button"
            aria-label="打开运行区"
            aria-expanded={mobilePanel === "run"}
            onClick={() => setMobilePanel(mobilePanel === "run" ? null : "run")}
          ><Gauge size={15} /> 运行</button>
        </div>
        {mobilePanel && (
          <div className="mobile-context__panel">
            {mobilePanel === "materials"
              ? "会话资料仅在本页保留 · 当前为共享知识库"
              : isDemo
                ? `${view.statusLabel} · ${view.events.length} 个 Demo 事件`
                : activeRun
                  ? activeRun.runId === runState.runId
                    ? `${runState.observedStatus ?? runState.record?.status ?? "等待状态"} · ${runState.events.length} 个事件`
                    : "等待状态 · 0 个事件"
                  : missingRunId
                    ? "运行不存在"
                    : "尚未提交 Research Run"}
          </div>
        )}
      </div>

      <div className="workbench-grid" data-testid="workbench-shell">
        <MaterialsWorkspace
          disabled={(!canMutate && !isDemo) || uploadPhase === "submitting"}
          materials={materials}
          uploadPhase={uploadPhase}
          uploadMessage={uploadMessage}
          onSelectFile={uploadMaterial}
        />
        <ResearchWorkspace
          mode={mode}
          onModeChange={setMode}
          scenario={isDemo ? scenario : undefined}
          view={isDemo ? view : undefined}
          canSubmit={canMutate && submissionPhase !== "submitting"}
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={submitQuestion}
          submissionPhase={submissionPhase}
          submissionError={submissionError}
          answer={answer}
          activeRun={activeRun}
          runState={runState}
          selectionError={missingRunId ? "运行不存在" : null}
        />
        <RunInspector
          view={isDemo ? view : undefined}
          activeRun={isDemo ? undefined : activeRun}
          runState={isDemo ? undefined : runState}
          selectionError={isDemo || !missingRunId ? undefined : "运行不存在"}
          error={isDemo ? undefined : submissionError}
          onResume={isDemo ? undefined : resumeRun}
          resumePhase={isDemo ? undefined : resumePhase}
          resumeError={isDemo ? undefined : resumeError}
        />
      </div>
    </div>
  );
}

function ServiceStatus({
  connection,
  onRetry,
}: {
  connection: ConnectionState;
  onRetry: () => void;
}) {
  const title = {
    checking: "正在检查后端",
    ready: "后端已就绪",
    not_ready: "后端未就绪",
    unreachable: "后端不可达",
    protocol_error: "就绪检查失败",
  }[connection.phase];
  const components = connection.readiness
    ? [
        ...Object.entries(connection.readiness.required).map(([name, value]) => ({
          name,
          value,
          kind: "required" as const,
        })),
        ...Object.entries(connection.readiness.optional).map(([name, value]) => ({
          name,
          value,
          kind: "optional" as const,
        })),
      ]
    : [];

  return (
    <section className={`service-status service-status--${connection.phase}`} aria-live="polite">
      <div className="service-status__summary">
        <span>HTTP API</span>
        <strong>{title}</strong>
        <small>{connection.healthAlive ? "后端连接正常" : connection.error ?? "正在连接…"}</small>
      </div>
      <div className="service-components" aria-label="后端组件状态">
        {components.map(({ name, value, kind }) => (
          <span className="service-component" key={`${kind}-${name}`}>
            <code>{name}</code>
            <b>{readinessLabel(value)}</b>
            {value.detail && <small>{value.detail}</small>}
          </span>
        ))}
      </div>
      <button type="button" onClick={onRetry} disabled={connection.phase === "checking"}>
        重新检查
      </button>
    </section>
  );
}

function readinessLabel(value: ComponentReadiness): string {
  return {
    ready: "就绪",
    unavailable: "不可用",
    available: "可用",
    unconfigured: "未配置",
  }[value.status];
}

async function readConnectionState(signal: AbortSignal): Promise<ConnectionState> {
  const [healthResult, readinessResult] = await Promise.allSettled([
    api.health(signal),
    api.readiness(signal),
  ]);
  if (healthResult.status === "rejected") {
    return {
      phase: "unreachable",
      healthAlive: false,
      readiness: null,
      error: describeApiError(healthResult.reason, "connection"),
    };
  }
  if (readinessResult.status === "rejected") {
    return {
      phase: "protocol_error",
      healthAlive: true,
      readiness: null,
      error: describeApiError(readinessResult.reason, "connection"),
    };
  }
  return {
    phase: readinessResult.value.status === "ready" ? "ready" : "not_ready",
    healthAlive: true,
    readiness: readinessResult.value,
    error: null,
  };
}
