"use client";

import { useMemo } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card, KpiCard, LoadState, PageHeader } from "@/components/ui";
import type { ComponentsResp, PueSummary, TempBands } from "@/lib/api";
import { fmtMonth, fmtNum } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const pct = (pue: number | null | undefined) => (pue == null ? null : (pue - 1) * 100);
const COMPONENTS = [
  { key: "hvac_kw", name: "HVAC", color: "#f59e0b" },
  { key: "pump_kw", name: "Pumps", color: "#059669" },
  { key: "cooling_kw", name: "Cooling", color: "#0284c7" },
  { key: "plug_and_light_kw", name: "Plug & light", color: "#dc2626" },
  { key: "filter_pump_kw", name: "Filter pump", color: "#94a3b8" },
];

export default function PuePage() {
  const summary = useApi<PueSummary>("/api/pue/summary");
  const temps = useApi<TempBands>("/api/pue/temperature", { start: "2025-01-01" });
  const comps = useApi<ComponentsResp>("/api/pue/components", { start: "2019-01-01", freq: "MS" });

  const yearly = useMemo(() => (summary.data?.yearly ?? []).map((r) => ({
    year: r.ts.slice(0, 4),
    all: pct(r.pue),
    clean: pct(r.pue_excl_meter_faults),
  })), [summary.data]);

  const monthly = useMemo(() => (summary.data?.monthly ?? [])
    .filter((r) => r.ts >= "2019")
    .map((r) => ({ label: fmtMonth(r.ts), pue: r.pue })), [summary.data]);

  const growth = useMemo(() => {
    const y = summary.data?.yearly ?? [];
    const a = y.find((r) => r.ts.startsWith("2020"));
    const b = y.find((r) => r.ts.startsWith("2024"));
    if (!a || !b) return null;
    return { it: b.it_mwh / a.it_mwh, oh: b.overhead_mwh / a.overhead_mwh };
  }, [summary.data]);

  const compData = useMemo(() => (comps.data?.points ?? []).map((p) => ({ label: fmtMonth(p.ts), ...p })), [comps.data]);

  return (
    <>
      <PageHeader title="PUE & Efficiency"
                  subtitle="All PUE values are energy-weighted (total facility energy / IT energy), per The Green Grid" />

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard label="PUE (30 days)" value={summary.data ? summary.data.kpis.last_30d.pue.toFixed(3) : "-"} />
        <KpiCard label="Overhead (30 days)" value={summary.data ? `${pct(summary.data.kpis.last_30d.pue)!.toFixed(1)}%` : "-"} hint="of IT energy" />
        <KpiCard label="IT energy growth" value={growth ? `${growth.it.toFixed(1)}x` : "-"} hint="2020 to 2024" />
        <KpiCard label="Overhead energy growth" value={growth ? `${growth.oh.toFixed(1)}x` : "-"} hint="2020 to 2024" />
      </div>

      <div className="mb-6 grid gap-6 xl:grid-cols-2">
        <Card title="Overhead per year (PUE - 1)"
              subtitle="Grey bars exclude hours flagged as meter faults; 2020-21 were flattered by faulty meters">
          <LoadState loading={summary.loading} error={summary.error} />
          {summary.data && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={yearly}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} unit="%" width={45} />
                <Tooltip formatter={(v) => (typeof v === "number" ? `${v.toFixed(2)}%` : "-")} />
                <Legend />
                <Bar dataKey="all" name="All hours" fill="#059669" />
                <Bar dataKey="clean" name="Excl. meter faults" fill="#94a3b8" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Monthly PUE since 2019" subtitle="Energy-weighted per month">
          <LoadState loading={summary.loading} error={summary.error} />
          {summary.data && (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={monthly}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={30} />
                <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} width={50}
                       tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : "-")} />
                <Line dataKey="pue" name="PUE" stroke="#7c3aed" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <Card className="mb-6" title="Overhead components, monthly mean"
            subtitle="The April 2024 HVAC step-change is the main driver of the PUE rise">
        <LoadState loading={comps.loading} error={comps.error} />
        {comps.data && (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={compData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={30} />
              <YAxis tick={{ fontSize: 11 }} unit=" kW" width={60} />
              <Tooltip formatter={(v) => (typeof v === "number" ? `${v.toFixed(1)} kW` : "-")} />
              <Legend />
              {COMPONENTS.map((c) => (
                <Area key={c.key} type="monotone" dataKey={c.key} name={c.name} stackId="1"
                      stroke={c.color} fill={c.color} fillOpacity={0.7} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card title="Overhead vs outdoor temperature (2025)"
            subtitle="Single facility configuration, so the effect is not confounded by the 2024 change">
        <LoadState loading={temps.loading} error={temps.error} />
        {temps.data && (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={temps.data.bands}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="temp_band_c" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} unit=" kW" domain={["auto", "auto"]} width={60} />
              <Tooltip
                formatter={(v, name) => (typeof v === "number" ? [`${fmtNum(v, 1)} kW`, name] : ["-", name])}
                labelFormatter={(l) => `Outdoor ${l} C`}
              />
              <Line dataKey="overhead_kw_mean" name="Mean overhead" stroke="#059669" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>
    </>
  );
}