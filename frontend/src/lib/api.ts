export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Params = Record<string, string | number | undefined>;

export async function apiGet<T>(path: string, params?: Params): Promise<T> {
  const url = new URL(path, API_URL);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  return (await res.json()) as T;
}

export type Meta = {
  data_start: string; data_end: string; timezone: string; hours: number; source: string;
};

export type Kpi = { pue: number; it_mw: number; overhead_kw: number; facility_mwh: number };

export type PeriodRow = {
  ts: string; it_mwh: number; facility_mwh: number; overhead_mwh: number;
  pue: number; coverage_pct: number; valid_hours: number;
};

export type PueSummary = {
  kpis: { last_7d: Kpi; last_30d: Kpi };
  yearly: (PeriodRow & { pue_excl_meter_faults: number | null })[];
  monthly: PeriodRow[];
};

export type MetricPoint = {
  ts: string;
  it_power_kw: number | null; overhead_kw: number | null; facility_kw: number | null;
  pue_calc: number | null; outdoor_temp_c: number | null;
  cooling_kw: number | null; hvac_kw: number | null; pump_kw: number | null;
  plug_and_light_kw: number | null;
};

export type Metrics = { freq: string; start: string; end: string; points: MetricPoint[] };

export type ForecastNext = {
  generated_from: string; horizon_h: number; weather_assumption: string;
  points: { ts: string; forecast_kw: number | null; persistence_kw: number | null }[];
  history: { ts: string; facility_kw: number | null }[];
};

export type Confidence = "low" | "medium" | "high";
export type Category = "operational" | "sustained_change" | "data_quality";

export type AnomalyEvent = {
  start: string; end: string; duration_h: number; peak_score: number | null;
  impact_kwh: number; kind: string; detector: string; max_agreement: number;
  confidence: Confidence; category: Category;
};

export type Anomalies = { count: number; by_category: Record<string, number>; events: AnomalyEvent[] };

export type Recommendation = {
  rule_id: string; title: string; severity: "high" | "medium" | "low"; category: string;
  action: string; evidence: Record<string, unknown>; est_savings_mwh: number | null;
  period_start: string; period_end: string;
};

export type Recommendations = { as_of: string; recommendations: Recommendation[] };

export type ForecastMetric = {
  model: string; split: string; MAE_kW: number; RMSE_kW: number;
  "MAPE_%": number; R2: number; "skill_vs_persistence_%": number;
};

export type ForecastTest = {
  horizon: number;
  points: Record<string, number | string | null>[];
  metrics: ForecastMetric[];
};

export type RecallRow = {
  kind: string; expected_detector: string; detected: number; trials: number;
  "recall_%": number; "ci95_low_%": number; "ci95_high_%": number; median_delay_h: number;
};

export type Evaluation = {
  forecast_h24: ForecastMetric[] | null;
  forecast_h1: ForecastMetric[] | null;
  overhead_model: { split: string; MAE_model_kW: number; MAE_ref_kW: number;
                    "skill_vs_ref_%": number; "MAPE_model_%": number }[] | null;
  anomaly_recall: RecallRow[] | null;
};

export type TempBands = {
  start: string; end: string | null;
  bands: { temp_band_c: string; pue: number; hours: number; overhead_kw_mean: number }[];
};

export type ComponentsResp = {
  freq: string;
  points: { ts: string; [k: string]: number | string | null }[];
};