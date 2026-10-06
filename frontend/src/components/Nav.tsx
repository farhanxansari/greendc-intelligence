"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/forecast", label: "Forecast" },
  { href: "/anomalies", label: "Anomalies" },
  { href: "/pue", label: "PUE & Efficiency" },
  { href: "/recommendations", label: "Recommendations" },
];

export function Nav() {
  const path = usePathname();
  return (
    <aside className="hidden w-60 shrink-0 border-r border-slate-200 bg-white md:block">
      <div className="px-5 py-6">
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <span className="text-sm font-semibold text-slate-900">GreenDC Intelligence</span>
        </div>
        <p className="mt-1 text-xs text-slate-500">Data-centre energy analytics</p>
      </div>
      <nav className="px-3">
        {LINKS.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              className={`mb-1 block rounded-lg px-3 py-2 text-sm ${
                active ? "bg-emerald-50 font-medium text-emerald-700" : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              }`}
            >
              {l.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}