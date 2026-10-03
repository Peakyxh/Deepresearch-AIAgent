import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepResearch Workspace",
  description: "Human-in-the-loop research workspace",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
