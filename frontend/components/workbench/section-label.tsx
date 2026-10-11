import type { ReactNode } from "react";

export function SectionLabel({
  index,
  title,
  action,
}: {
  index: string;
  title: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-label">
      <div className="section-label__title">
        <span aria-hidden="true">{index}</span>
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}
