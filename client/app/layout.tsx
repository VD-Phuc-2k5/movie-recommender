import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "MovieLens Spark Recommender",
  description: "MovieLens 32M Spark ALS pipeline dashboard",
};

const screens = [
  ["/datasets", "Datasets"],
  ["/processing", "Processing"],
  ["/training", "Training"],
  ["/results", "Results"],
  ["/prediction-recommendation", "Prediction & Recommendation"],
] as const;

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="vi">
      <body>
        <div className="min-h-screen bg-zinc-50">
          <header className="border-b border-zinc-200 bg-white">
            <div className="mx-auto flex max-w-6xl flex-col gap-4 px-5 py-5 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-xs font-semibold tracking-[0.16em] text-indigo-600 uppercase">rs-movie</p>
                <p className="mt-1 text-sm font-medium text-zinc-800">MovieLens 32M · Spark ALS</p>
              </div>
              <nav aria-label="Application screens" className="flex flex-wrap gap-x-5 gap-y-2 text-sm font-medium text-zinc-600">
                {screens.map(([href, label]) => (
                  <Link className="transition hover:text-indigo-600" href={href} key={href}>
                    {label}
                  </Link>
                ))}
              </nav>
            </div>
          </header>
          <main className="mx-auto w-full max-w-6xl px-5 py-10">{children}</main>
        </div>
      </body>
    </html>
  );
}
