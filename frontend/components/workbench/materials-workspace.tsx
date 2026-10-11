import { Database, FileText, Plus, Upload } from "lucide-react";

import { SectionLabel } from "./section-label";
import type { AsyncPhase, SessionMaterial } from "./workbench-types";

export function MaterialsWorkspace({
  disabled = false,
  materials = [],
  uploadPhase = "idle",
  uploadMessage,
  onSelectFile,
}: {
  disabled?: boolean;
  materials?: SessionMaterial[];
  uploadPhase?: AsyncPhase;
  uploadMessage?: string | null;
  onSelectFile?: (file: File) => void;
}) {
  return (
    <aside className="rail rail--left" aria-label="资料工作区">
      <header className="brand">
        <div className="brand__mark" aria-hidden="true">IA</div>
        <div>
          <p className="brand__name">InsightAgent</p>
          <p className="brand__edition">Research Workbench · Local</p>
        </div>
      </header>

      <section className="rail-section">
        <SectionLabel index="01" title="Materials Workspace" />
        <label className="upload-zone" htmlFor="material-upload">
          <input
            id="material-upload"
            className="sr-only"
            type="file"
            disabled={disabled}
            aria-label="选择研究资料"
            accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.webp"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) onSelectFile?.(file);
            }}
          />
          <span className="upload-zone__icon"><Upload size={18} /></span>
          <strong>选择研究资料</strong>
          <small>Markdown · Text · PDF · Image*</small>
          <span className="upload-zone__phase">
            {uploadPhase === "submitting"
              ? "正在上传并建立索引…"
              : "真实上传 · 会话内记录"}
          </span>
        </label>
        {uploadMessage && (
          <p
            className={uploadPhase === "failed" ? "upload-feedback upload-feedback--error" : "upload-feedback"}
            role="status"
          >
            {uploadMessage}
          </p>
        )}
      </section>

      <section className="rail-section rail-section--grow">
        <SectionLabel
          index="02"
          title="Session Materials"
          action={
            <label
              className="icon-button"
              htmlFor="material-upload"
              aria-label="添加资料"
              aria-disabled={disabled}
            >
              <Plus size={15} />
            </label>
          }
        />
        <div className="material-list">
          {materials.map((material, index) => (
            <article className="material-row" key={`${material.material_id}-${index}`}>
              <span className="material-row__icon" aria-hidden="true"><FileText size={16} /></span>
              <div className="material-row__body">
                <strong>{material.fileName}</strong>
                <span>{material.chunk_count} chunks · {material.status}</span>
              </div>
              <span className="material-row__state">
                {material.deduplicated ? "去重索引" : "首次索引"}
              </span>
            </article>
          ))}
        </div>
        {materials.length === 0 && (
          <p className="mock-note">当前会话尚无已确认上传的资料。</p>
        )}
        <p className="mock-note">资料记录仅保存在当前浏览器会话，刷新页面后会清空。</p>
      </section>

      <section className="knowledge-note" aria-label="共享知识库说明">
        <Database size={17} aria-hidden="true" />
        <div>
          <strong>共享知识库</strong>
          <p>当前资料进入共享知识库，无法按 material_id 隔离。</p>
        </div>
      </section>
    </aside>
  );
}
