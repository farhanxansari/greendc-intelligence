import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Nav } from "@/components/Nav";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "GreenDC Intelligence",
  description: "AI-driven data-centre energy analytics",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-50 text-slate-900 antialiased`}>
        <div className="flex min-h-screen">
          <Nav />
          <main className="flex-1 overflow-x-hidden p-6 lg:p-8">{children}</main>
        </div>
      </body>
    </html>
  );
}