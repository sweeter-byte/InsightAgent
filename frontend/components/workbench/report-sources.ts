export type ReportSource = {
  number: number;
  label: string;
  detail: string | null;
  url: string | null;
};

const referencesHeading = /^#{1,6}\s+参考来源\s*$/;
const markdownHeading = /^#{1,6}\s+/;
const sourceLine = /^\s*\[(\d+)]\s+(.+?)\s*$/;
const externalUrl = /https?:\/\/[^\s，,)）]+/i;

export function extractReportSources(markdown: string): ReportSource[] {
  const lines = markdown.split(/\r?\n/);
  const headingIndex = lines.findIndex((line) => referencesHeading.test(line.trim()));
  if (headingIndex === -1) return [];

  const sources: ReportSource[] = [];
  const seenNumbers = new Set<number>();
  for (const line of lines.slice(headingIndex + 1)) {
    if (markdownHeading.test(line.trim())) break;
    const match = sourceLine.exec(line);
    if (!match) continue;

    const number = Number(match[1]);
    const body = match[2]?.trim();
    if (!Number.isSafeInteger(number) || !body || seenNumbers.has(number)) continue;

    const separatorIndex = body.search(/[，,]/);
    const label = (separatorIndex === -1 ? body : body.slice(0, separatorIndex)).trim();
    if (!label) continue;
    const detail = separatorIndex === -1 ? null : body.slice(separatorIndex + 1).trim() || null;
    const url = body.match(externalUrl)?.[0] ?? null;
    sources.push({ number, label, detail, url });
    seenNumbers.add(number);
  }
  return sources;
}
