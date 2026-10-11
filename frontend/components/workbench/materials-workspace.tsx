import { Database, FileText, Plus, Upload } from "lucide-react";

import { mockMaterials } from "./mock-data";
import { SectionLabel } from "./section-label";

export function MaterialsWorkspace() {
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
            aria-label="选择研究资料"
            accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.webp"
          />
          <span className="upload-zone__icon"><Upload size={18} /></span>
          <strong>选择研究资料</strong>
          <small>Markdown · Text · PDF · Image*</small>
          <span className="upload-zone__phase">静态入口 · API 待接入</span>
        </label>
      </section>

      <section className="rail-section rail-section--grow">
        <SectionLabel
          index="02"
          title="Session Materials"
          action={<button className="icon-button" type="button" aria-label="添加资料"><Plus size={15} /></button>}
        />
        <div className="material-list">
          {mockMaterials.map((material) => (
            <article className="material-row" key={material.name}>
              <span className="material-row__icon" aria-hidden="true"><FileText size={16} /></span>
              <div className="material-row__body">
                <strong>{material.name}</strong>
                <span>{material.kind} · {material.chunks} chunks</span>
              </div>
              <span className="material-row__state">{material.state}</span>
            </article>
          ))}
        </div>
        <p className="mock-note">仅为会话内 Mock 记录，不代表持久化资料列表。</p>
      </section>

      <section className="knowledge-note" aria-label="共享知识库说明">
        <Database size={17} aria-hidden="true" />
        <div>
          <strong>共享知识库</strong>
          <p>当前检索面向同一集合，无法按 material_id 隔离。</p>
        </div>
      </section>
    </aside>
  );
}
