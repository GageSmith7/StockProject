import type { Metadata } from "next";
import { Space_Mono, DM_Sans } from "next/font/google";
import "./globals.css";

const spaceMono = Space_Mono({
  weight:   ["400", "700"],
  subsets:  ["latin"],
  variable: "--font-space-mono",
  display:  "swap",
});

const dmSans = DM_Sans({
  subsets:  ["latin"],
  variable: "--font-dm-sans",
  display:  "swap",
});

export const metadata: Metadata = {
  title: "Portfolio Simulator",
  description: "Build a portfolio, analyze historical performance, project future outcomes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceMono.variable} ${dmSans.variable}`}>
      <body className="min-h-screen antialiased">
        {/* Nav */}
        <header
          className="sticky top-0 z-50 border-b"
          style={{
            background: "rgba(8, 13, 24, 0.92)",
            borderColor: "var(--border)",
            backdropFilter: "blur(12px)",
          }}
        >
          <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span
                className="text-base tracking-widest uppercase"
                style={{ color: "var(--accent)" }}
              >
                PortfolioSim
              </span>
              <span
                className="text-xs hidden sm:block tracking-wider"
                style={{ color: "var(--text-muted)" }}
              >
                / ANALYZE / SIMULATE / PREDICT
              </span>
            </div>
            <span
              className="text-xs tracking-wide"
              style={{ color: "var(--text-muted)" }}
            >
              NOT FINANCIAL ADVICE
            </span>
          </div>
        </header>

        <main className="max-w-screen-2xl mx-auto px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
