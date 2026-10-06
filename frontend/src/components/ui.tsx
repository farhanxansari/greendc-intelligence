import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function Card({ title, subtitle, action, children, className = "" }: {
  title?: string; subtitle?: ReactNode; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || action) && (
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

export function KpiCard({ label, value, unit, hint }: { label: string; value: string; unit?: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-900">
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-slate-500">{unit}</span>}
      </p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

const TONES: Record<string, string> = {
  high: "bg-red-50 text-red-700 ring-red-200",
  medium: "bg-amber-50 text-amber-700 ring-amber-200",
  low: "bg-slate-100 text-slate-600 ring-slate-200",
  operational: "bg-sky-50 text-sky-700 ring-sky-200",
  sustained_change: "bg-violet-50 text-violet-700 ring-violet-200",
  data_quality: "bg-orange-50 text-orange-700 ring-orange-200",
};

export function Badge({ tone, children }: { tone: string; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone] ?? TONES.low}`}>
      {children}
    </span>
  );
}

export function LoadState({ loading, error, height = "h-64" }: { loading: boolean; error: string | null; height?: string }) {
  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        Could not load data: {error}
        <br />
        <span className="text-xs">Is the API running on port 8000?</span>
      </div>
    );
  }
  if (loading) return <div className={`${height} animate-pulse rounded-lg bg-slate-100`} />;
  return null;
}

export function Segmented<T extends string | number>({ options, value, onChange }: {
  options: { label: string; value: T }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-slate-200 bg-white p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-xs font-medium ${
            o.value === value ? "bg-emerald-600 text-white" : "text-slate-600 hover:bg-slate-50"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}