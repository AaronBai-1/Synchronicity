import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Synchronicity",
  description: "Badminton match analytics - stroke-level tactical insights from match video",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <Link href="/" className="brand">
            Synchronicity
          </Link>
          <span className="tag">badminton analytics — Phase 0 scaffold</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
