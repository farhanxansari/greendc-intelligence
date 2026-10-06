"use client";

import Link from "next/link";
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Badge, Card, KpiCard, LoadState, PageHeader } from "@/components/ui";
import type { Anomalies, ForecastNext, Meta, Metrics, PueSummary, Recommendations } from "@/lib/api";
import { fmtDate, fmtDateTime, fmtHour, fmtNum, fmtShort, humanize } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const fmt = (v: unknown, d = 2) => (typeof v === "number" ? v.toFixed(d) : String(v ?? "-"));

export default function OverviewPage() {
  const meta = useApi<Meta>("/api/meta");
  const pue = useApi<PueSummary>("/api/pue/summary");
  const metrics = useApi<Metrics>("/api/metrics", { freq: "D" });
  const forecast = useApi<ForecastNext>("/api/forecast/next");
  const recs = useApi<Recommendations>("/api/recommendations");
  const anomalies = useApi<Anomalies>("/api/anomalies", {
    min_confidence: "high", category: "operational", limit: 6,
  });

  const k7 = pue.data?.kpis.last_7d;
  const k30 = pue.data?.kpis.last_30d;

  const daily = (metrics.data?.points ?? []).map((p) => ({
    label: fmtShort(p.ts),
    facility: p.facility_kw == null ? null : p.facility_kw / 1000,
    it: p.it_power_kw == null ? null : p.it_power_kw / 1000,
    pue: p.pue_calc,
  }));

  const fc = [
    ...(forecast.data?.history ?? []).map((p) => ({
      label: fmtHour(p.ts), actual: p.facility_kw, forecast: null as number | null, persistence: null as number | null,
    })),
    ...(forecast.data?.points ?? []).map((p) => ({
      label: fmtHour(p.ts), actual: null as number | null, forecast: p.forecast_kw, persistence: p.persistence_kw,
    })),
  ];

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle={meta.data
          ? <>NLR ESIF HPC data centre, data through {fmtDateTime(meta.data.data_end)} (facility local time)</>
          : "Loading..."}
      />

      {/* KPIs */}
      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-5">
        <KpiCard label="PUE (7 days)" value={fmt(k7?.pue, 3)} hint="energy-weighted" />
        <KpiCard label="PUE (30 days)" value={fmt(k30?.pue, 3)} hint="energy-weighted" />
        <KpiCard label="IT load" value={fmt(k7?.it_mw, 2)} unit="MW" hint="7-day mean" />
        <KpiCard label="Overhead" value={fmtNum(k7?.overhead_kw)} unit="kW" hint="7-day mean" />
        <KpiCard label="Facility energy" value={fmtNum(k30?.facility_mwh)} unit="MWh" hint="last 30 days" />
      </div>

      {/* Charts */}
      <div className="mb-6 grid gap-6 xl:grid-cols-2">
        <Card title="Facility and IT power" subtitle="Daily mean, last 90 days">
          <LoadState loading={metrics.loading} error={metrics.error} />
          {metrics.data && (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11 }} width={72} tickFormatter={(v: number) => `${v} MW`} />
                <Tooltip formatter={(v) => fmt(v, 2)} />
                <Legend />
                <Line dataKey="facility" name="Facility" stroke="#059669" dot={false} strokeWidth={2} />
                <Line dataKey="it" name="IT" stroke="#0284c7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="PUE" subtitle="Daily energy-weighted PUE, last 90 days">
          <LoadState loading={metrics.loading} error={metrics.error} />
          {metrics.data && (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]}
                       tickFormatter={(v: number) => v.toFixed(3)} width={60} />
                <Tooltip formatter={(v) => fmt(v, 4)} />
                <Line dataKey="pue" name="PUE" stroke="#7c3aed" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <Card
        className="mb-6"
        title="Next 24 hours: facility power forecast"
        subtitle={forecast.data?.weather_assumption}
        action={<Link href="/forecast" className="text-xs font-medium text-emerald-700 hover:underline">Forecast details</Link>}
      >
        <LoadState loading={forecast.loading} error={forecast.error} />
        {forecast.data && (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={fc}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={40} />
              <YAxis tick={{ fontSize: 11 }} unit=" kW" width={70} domain={["auto", "auto"]} />
              <Tooltip formatter={(v) => fmtNum(typeof v === "number" ? v : null)} />
              <Legend />
              <Line dataKey="actual" name="Actual (last 72 h)" stroke="#334155" dot={false} strokeWidth={2} />
              <Line dataKey="forecast" name="XGBoost forecast" stroke="#059669" dot={false} strokeWidth={2} />
              <Line dataKey="persistence" name="Persistence baseline" stroke="#94a3b8" dot={false} strokeDasharray="5 4" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>

      {/* Recommendations + anomalies */}
      <div className="grid gap-6 xl:grid-cols-2">
        <Card
          title="Top recommendations"
          subtitle={recs.data ? `As of ${fmtDate(recs.data.as_of)}` : undefined}
          action={<Link href="/recommendations" className="text-xs font-medium text-emerald-700 hover:underline">View all</Link>}
        >
          <LoadState loading={recs.loading} error={recs.error} height="h-40" />
          <ul className="space-y-4">
            {recs.data?.recommendations.slice(0, 3).map((r) => (
              <li key={r.rule_id + r.title} className="border-b border-slate-100 pb-4 last:border-0 last:pb-0">
                <div className="flex items-center gap-2">
                  <Badge tone={r.severity}>{r.severity}</Badge>
                  <span className="text-sm font-medium text-slate-900">{r.title}</span>
                </div>
                <p className="mt-1 line-clamp-2 text-sm text-slate-600">{r.action}</p>
                {r.est_savings_mwh != null && (
                  <p className="mt-1 text-xs font-medium text-emerald-700">Estimated saving: {fmtNum(r.est_savings_mwh, 1)} MWh</p>
                )}
              </li>
            ))}
            {recs.data && recs.data.recommendations.length === 0 && (
              <li className="text-sm text-slate-500">No recommendations at this time.</li>
            )}
          </ul>
        </Card>

        <Card
          title="Recent high-confidence anomalies"
          subtitle="Operational events, last 12 months"
          action={<Link href="/anomalies" className="text-xs font-medium text-emerald-700 hover:underline">View all</Link>}
        >
          <LoadState loading={anomalies.loading} error={anomalies.error} height="h-40" />
          {anomalies.data && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-500">
                  <tr>
                    <th className="pb-2 font-medium">Start</th>
                    <th className="pb-2 font-medium">Type</th>
                    <th className="pb-2 text-right font-medium">Hours</th>
                    <th className="pb-2 text-right font-medium">Impact kWh</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {anomalies.data.events.map((e) => (
                    <tr key={e.start + e.detector}>
                      <td className="py-2 text-slate-700">{fmtDateTime(e.start)}</td>
                      <td className="py-2 text-slate-700">{humanize(e.kind)}</td>
                      <td className="py-2 text-right text-slate-700">{e.duration_h}</td>
                      <td className="py-2 text-right text-slate-700">{fmtNum(e.impact_kwh)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {anomalies.data.events.length === 0 && <p className="text-sm text-slate-500">No events.</p>}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}