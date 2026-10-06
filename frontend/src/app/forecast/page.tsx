"use client";

import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Card, LoadState, PageHeader, Segmented } from "@/components/ui";
import type { ForecastTest } from "@/lib/api";
import { fmtHour, fmtNum, humanize } from "@/lib/format";
import { useApi } from "@/lib/useApi";

const MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08"].map((m) => ({
  label: new Date(`2025-${m}-15`).toLocaleDateString("en-GB", { month: "short" }),
  value: `2025-${m}`,
}));
const BASELINES = ["persistence_24h", "persistence_1h", "seasonal_naive_168h"];

export default function ForecastPage() {
  const [horizon, setHorizon] = useState<number>(24);
  const [month, setMonth] = useState("2025-08");
  const test = useApi<ForecastTest>("/api/forecast/test", { horizon });
  const persistKey = `persistence_${horizon}h`;

  // best model is chosen on VALIDATION (never on test), mirroring the training script
  const bestModel = useMemo(() => {
    const val = (test.data?.metrics ?? []).filter((m) => m.split === "val" && !BASELINES.includes(m.model));
    return [...val].sort((a, b) => a.MAE_kW - b.MAE_kW)[0]?.model ?? "xgboost";
  }, [test.data]);

  const testRows = useMemo(
    () => (test.data?.metrics ?? []).filter((m) => m.split === "test").sort((a, b) => a.MAE_kW - b.MAE_kW),
    [test.data],
  );

  const series = useMemo(
    () => (test.data?.points ?? [])
      .filter((p) => String(p.ts).startsWith(month))
      .map((p) => ({
        label: fmtHour(String(p.ts)),
        actual: p.actual as number | null,
        persistence: p[persistKey] as number | null,
        model: p[bestModel] as number | null,
      })),
    [test.data, month, persistKey, bestModel],
  );

  const skill = testRows
    .filter((m) => !m.model.startsWith("persistence"))
    .map((m) => ({ model: humanize(m.model), skill: m["skill_vs_persistence_%"] }));

  return (
    <>
      <PageHeader
        title="Forecast"
        subtitle="Facility power forecasts on the held-out 2025 test period (models never saw this data)"
        action={
          <Segmented
            options={[{ label: "Day-ahead (24 h)", value: 24 }, { label: "Hour-ahead (1 h)", value: 1 }]}
            value={horizon}
            onChange={setHorizon}
          />
        }
      />

      <Card
        className="mb-6"
        title={`Actual vs ${humanize(bestModel)} vs persistence`}
        subtitle="Hourly facility power, test period. Best model selected on the 2024 validation period."
        action={<Segmented options={MONTHS} value={month} onChange={setMonth} />}
      >
        <LoadState loading={test.loading} error={test.error} height="h-80" />
        {test.data && (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={60} />
              <YAxis tick={{ fontSize: 11 }} width={72} domain={["auto", "auto"]}
                     tickFormatter={(v: number) => `${fmtNum(v)} kW`} />
              <Tooltip formatter={(v) => (typeof v === "number" ? `${fmtNum(v)} kW` : "-")} />
              <Legend />
              <Line dataKey="actual" name="Actual" stroke="#334155" dot={false} strokeWidth={1.5} />
              <Line dataKey="model" name={humanize(bestModel)} stroke="#059669" dot={false} strokeWidth={1.5} />
              <Line dataKey="persistence" name="Persistence" stroke="#94a3b8" dot={false} strokeDasharray="4 3" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Test-period metrics (2025)" subtitle="Sorted by MAE. Skill = % MAE reduction vs persistence.">
          <LoadState loading={test.loading} error={test.error} height="h-40" />
          {test.data && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-slate-500">
                  <tr>
                    <th className="pb-2 font-medium">Model</th>
                    <th className="pb-2 text-right font-medium">MAE kW</th>
                    <th className="pb-2 text-right font-medium">RMSE kW</th>
                    <th className="pb-2 text-right font-medium">MAPE %</th>
                    <th className="pb-2 text-right font-medium">Skill %</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {testRows.map((m) => (
                    <tr key={m.model} className={m.model === bestModel ? "bg-emerald-50/60" : ""}>
                      <td className="py-2 font-medium text-slate-800">
                        {humanize(m.model)}{m.model === bestModel && <span className="ml-2 text-xs text-emerald-700">selected</span>}
                      </td>
                      <td className="py-2 text-right">{fmtNum(m.MAE_kW, 1)}</td>
                      <td className="py-2 text-right">{fmtNum(m.RMSE_kW, 1)}</td>
                      <td className="py-2 text-right">{fmtNum(m["MAPE_%"], 2)}</td>
                      <td className={`py-2 text-right font-medium ${m["skill_vs_persistence_%"] >= 0 ? "text-emerald-700" : "text-red-600"}`}>
                        {fmtNum(m["skill_vs_persistence_%"], 1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title="Skill vs persistence (test)" subtitle="Positive = better than 'same as yesterday'">
          <LoadState loading={test.loading} error={test.error} height="h-40" />
          {test.data && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={skill} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} unit="%" />
                <YAxis type="category" dataKey="model" tick={{ fontSize: 11 }} width={110} />
                <Tooltip formatter={(v) => (typeof v === "number" ? `${v.toFixed(1)}%` : "-")} />
                <Bar dataKey="skill" name="Skill">
                  {skill.map((s) => <Cell key={s.model} fill={s.skill >= 0 ? "#059669" : "#dc2626"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <Card className="mt-6" title="Method notes">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Models predict the <b>ratio change vs persistence</b>, which makes them robust to the IT regime shifts (1.2 to 3.7 to 2.6 MW).</li>
          <li>Split: train Jan 2019 to Jun 2024, validate Jul to Dec 2024, test Jan to Aug 2025. Models are refit on train+validation before testing.</li>
          <li>Day-ahead gains are bounded because HPC load is driven by job submissions, which are unknowable a day ahead.</li>
        </ul>
      </Card>
    </>
  );
}