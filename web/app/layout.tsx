import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "SixFingers",
  description: "Check whether an image looks fully AI-generated.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site">
          <div className="wrap">
            <Link href="/" className="mark">SixFingers<span>.</span></Link>
            <nav aria-label="Main">
              <Link href="/scorecard">Scorecard</Link>
              <Link href="/about">About</Link>
              <Link href="/privacy">Privacy</Link>
              <Link href="/terms">Terms</Link>
            </nav>
          </div>
        </header>
        <main>{children}</main>
        <footer className="site">
          <div className="wrap">
            <span>Built by <a href="https://avikwok.dev">Avidan Kwok</a></span>
            <a href="https://github.com/avi-kwok/SixFingers">Source</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
