import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";

function nodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  return "";
}

function isExternalLink(href: string | undefined): boolean {
  return href?.startsWith("https://") === true || href?.startsWith("http://") === true;
}

export function MarkdownReport({ markdown }: { markdown: string }) {
  return (
    <article className="report-paper report-markdown" aria-label="最终研究报告">
      <div className="report-paper__meta" aria-hidden="true">
        <span>final_output</span>
        <span>authoritative markdown</span>
      </div>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeSanitize]}
        components={{
          a({ href, children, ...props }) {
            const external = isExternalLink(href);
            return (
              <a
                {...props}
                href={href}
                {...(external
                  ? { target: "_blank", rel: "noopener noreferrer" }
                  : {})}
              >
                {children}
              </a>
            );
          },
          h2({ children, ...props }) {
            const isReferences = nodeText(children).trim() === "参考来源";
            return (
              <h2 {...props} id={isReferences ? "report-references" : undefined}>
                {children}
              </h2>
            );
          },
          table({ children, ...props }) {
            return (
              <div className="report-table-scroll">
                <table {...props}>{children}</table>
              </div>
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </article>
  );
}
