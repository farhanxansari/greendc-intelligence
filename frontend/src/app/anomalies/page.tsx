
"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Card, KpiCard, LoadState, PageHeader, Segmented } from "@/components/ui";
import type { Anomalies, Confidence, Evaluation } from "@/lib/api";
import { fmtDateTime, fmtNum, humanize } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const YEARS = ["2019", "2020", "2021", "2022", "2023", "2024", "2025"];
const CATEGORIES = [
  { label: "All", value: "all" },
  { label: "Operational", value: "operational" },
  { label: "Sustained change", value: "sustained_change" },
  { label: "Data quality", value: "data_quality" },
];
const MAX_ROWS = 200;

export default function AnomaliesPage() {
  const [year, setYear] = useState("2025");
  const [conf, setConf] = useState<Confidence>("medium");
  const [cat, setCat] = useState("all");

  const an = useApi<Anomalies>("/api/anomalies", {
    start: `${year}-01-01`, end: `${year}-12-31`, min_confidence: conf,
    category: cat === "all" ? undefined : cat, limit: 5000,
  });
  const evaluation = useApi<Evaluation>("/api/evaluation");

  const monthly = useMemo(() => {
    const rows = new Map<string, { month: string; operational: number; sustained_change: number; data_quality: number }>();
    for (const e of an.data?.events ?? []) {
      const key = e.start.slice(0, 7);
      const row = rows.get(key) ?? { month: key, operational: 0, sustained_change: 0, data_quality: 0 };
      row[e.category] += 1;
      rows.set(key, row);
    }
    return [...rows.values()].sort((a, b) => a.month.localeCompare(b.month));
  }, [an.data]);

  const by = an.data?.by_category ?? {};

  return (
    <>
      <PageHeader
        title="Anomalies"
        subtitle="Four complementary detectors: overhead residual, load change, isolation forest, meter fault"
        action={<Segmented options={YEARS.map((y) => ({ label: y, value: y }))} value={year} onChange={setYear} />}
      />

      <div className="mb-4 flex flex-wrap gap-3">
        <Segmented options={CATEGORIES} value={cat} onChange={setCat} />
        <Segmented
          options={[
            { label: "All confidence", value: "low" as Confidence },
            { label: "Medium+", value: "medium" as Confidence },
            { label: "High only", value: "high" as Confidence },
          ]}
          value={conf}
          onChange={setConf}
        />
      </div>

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="Events" value={fmtNum(an.data?.count)} hint={`${year}, current filters`} />
        <KpiCard label="Operational" value={fmtNum(by.operational ?? 0)} hint="short excursions: act" />
        <KpiCard label="Sustained change" value={fmtNum(by.sustained_change ?? 0)} hint=">= 72 h: review" />
        <KpiCard label="Data quality" value={fmtNum(by.data_quality ?? 0)} hint="meter faults: maintain" />
      </div>

      <Card className="mb-6" title="Events per month" subtitle="By category">
        <LoadState loading={an.loading} error={an.error} height="h-56" />
        {an.data && (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} width={40} />
              <Tooltip />
              <Legend />
              <Bar dataKey="operational" name="Operational" stackId="a" fill="#0284c7" />
              <Bar dataKey="sustained_change" name="Sustained change" stackId="a" fill="#7c3aed" />
              <Bar dataKey="data_quality" name="Data quality" stackId="a" fill="#ea580c" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card
        className="mb-6"
        title="Event log"
        subtitle={an.data && an.data.count > MAX_ROWS ? `Showing latest ${MAX_ROWS} of ${an.data.count}` : "Newest first"}
      >
        <LoadState loading={an.loading} error={an.error} height="h-40" />
        {an.data && (
          <div className="max-h-[520px] overflow-auto">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-white text-xs uppercase text-slate-500">
                <tr>
                  <th className="pb-2 font-medium">Start</th>
                  <th className="pb-2 text-right font-medium">Hours</th>
                  <th className="pb-2 font-medium">Detector</th>
                  <th className="pb-2 font-medium">Type</th>
                  <th className="pb-2 font-medium">Category</th>
                  <th className="pb-2 font-medium">Confidence</th>
                  <th className="pb-2 text-right font-medium">Impact kWh</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {an.data.events.slice(0, MAX_ROWS).map((e) => (
                  <tr key={`${e.start}-${e.detector}-${e.kind}`}>
                    <td className="py-2 whitespace-nowrap text-slate-700">{fmtDateTime(e.start)}</td>
                    <td className="py-2 text-right text-slate-700">{e.duration_h}</td>
                    <td className="py-2 text-slate-700">{humanize(e.detector)}</td>
                    <td className="py-2 text-slate-700">{humanize(e.kind)}</td>
                    <td className="py-2"><Badge tone={e.category}>{humanize(e.category)}</Badge></td>
                    <td className="py-2"><Badge tone={e.confidence}>{e.confidence}</Badge></td>
                    <td className="py-2 text-right text-slate-700">{fmtNum(e.impact_kwh)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {an.data.events.length === 0 && <p className="mt-2 text-sm text-slate-500">No events for these filters.</p>}
          </div>
        )}
      </Card>

      <Card title="Detector evaluation" subtitle="Synthetic faults injected into 2025 test data, 30 trials per fault type">
        <LoadState loading={evaluation.loading} error={evaluation.error} height="h-32" />
        {evaluation.data?.anomaly_recall && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="pb-2 font-medium">Injected fault</th>
                <th className="pb-2 font-medium">Detector</th>
                <th className="pb-2 text-right font-medium">Recall</th>
                <th className="pb-2 text-right font-medium">95% CI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {evaluation.data.anomaly_recall.map((r) => (
                <tr key={r.kind}>
                  <td className="py-2 text-slate-800">{humanize(r.kind)}</td>
                  <td className="py-2 text-slate-600">{humanize(r.expected_detector)}</td>
                  <td className="py-2 text-right font-medium text-emerald-700">{fmtNum(r["recall_%"], 1)}%</td>
                  <td className="py-2 text-right text-slate-600">{fmtNum(r["ci95_low_%"], 1)} - {fmtNum(r["ci95_high_%"], 1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}