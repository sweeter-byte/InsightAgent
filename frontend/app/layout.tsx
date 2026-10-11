import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "InsightAgent Research Workbench",
  description: "Local research workspace for InsightAgent",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
