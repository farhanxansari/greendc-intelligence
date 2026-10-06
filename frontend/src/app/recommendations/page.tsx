"use client";

import { useState } from "react";
import { Badge, Card, KpiCard, LoadState, PageHeader, Segmented } from "@/components/ui";
import type { Recommendations } from "@/lib/api";
import { fmtDate, fmtNum, humanize } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const PRESETS = [
  { label: "Latest", value: "" },
  { label: "1 Jun 2024 (HVAC step-change)", value: "2024-06-01" },
  { label: "15 Sep 2020 (meter faults)", value: "2020-09-15" },
];

const fmtEvidence = (v: unknown) =>
  typeof v === "number" ? fmtNum(v, Number.isInteger(v) ? 0 : 2) : typeof v === "boolean" ? (v ? "yes" : "no") : String(v);

export default function RecommendationsPage() {
  const [asOf, setAsOf] = useState("");
  const recs = useApi<Recommendations>("/api/recommendations", { as_of: asOf || undefined });

  const list = recs.data?.recommendations ?? [];
  const total = list.reduce((s, r) => s + (r.est_savings_mwh ?? 0), 0);
  const count = (sev: string) => list.filter((r) => r.severity === sev).length;

  return (
    <>
      <PageHeader
        title="Recommendations"
        subtitle={recs.data ? `Evidence-backed actions as of ${fmtDate(recs.data.as_of)}. Replay any date to see what the system would have advised.` : "Loading..."}
      />

      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Segmented options={PRESETS} value={PRESETS.some((p) => p.value === asOf) ? asOf : "custom"} onChange={setAsOf} />
        <label className="flex items-center gap-2 text-xs text-slate-600">
          or pick a date
          <input
            type="date"
            min="2019-02-01"
            max="2025-08-28"
            value={asOf}
            onChange={(e) => setAsOf(e.target.value)}
            className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs"
          />
        </label>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="High" value={String(count("high"))} />
        <KpiCard label="Medium" value={String(count("medium"))} />
        <KpiCard label="Low" value={String(count("low"))} />
        <KpiCard label="Est. savings" value={fmtNum(total, 1)} unit="MWh" hint="where estimable" />
      </div>

      <LoadState loading={recs.loading} error={recs.error} height="h-48" />

      <div className="space-y-4">
        {list.map((r) => (
          <Card key={r.rule_id + r.title}>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={r.severity}>{r.severity}</Badge>
              <Badge tone="low">{humanize(r.category)}</Badge>
              {r.est_savings_mwh != null && (
                <span className="ml-auto text-sm font-medium text-emerald-700">
                  Est. saving {fmtNum(r.est_savings_mwh, 1)} MWh
                </span>
              )}
            </div>
            <h3 className="mt-3 text-base font-semibold text-slate-900">{r.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">{r.action}</p>
            <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 rounded-lg bg-slate-50 p-3 text-xs sm:grid-cols-3 lg:grid-cols-4">
              {Object.entries(r.evidence).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-slate-500">{humanize(k)}</dt>
                  <dd className="font-medium text-slate-800">{fmtEvidence(v)}</dd>
                </div>
              ))}
            </dl>
            {r.period_start && (
              <p className="mt-2 text-xs text-slate-400">
                Evidence window: {fmtDate(r.period_start)} to {fmtDate(r.period_end)}
              </p>
            )}
          </Card>
        ))}
        {recs.data && list.length === 0 && (
          <Card><p className="text-sm text-slate-500">No recommendations for this date: all rules within thresholds.</p></Card>
        )}
      </div>
    </>
  );
}