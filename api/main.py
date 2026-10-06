"""GreenDC Intelligence API (FastAPI).

Thin layer over the greendc package: all analytics live in src/greendc. This file
loads precomputed pipeline outputs, runs light computations, and serves JSON.

Run (repo root):  uvicorn api.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""
from dataclasses import asdict
from functools import lru_cache
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from greendc.config import HOURLY_FILE, MODELS_DIR, PRED_DIR, RESULTS_DIR, TZ, ts
from greendc.features.forecast_features import make_forecast_frame
from greendc.pue.metrics import (component_breakdown, exclude_meter_faults, period_pue,
                                 pue_by_temp_band)
from greendc.recommend.engine import load_events
from greendc.recommend.engine import run as run_recommendations

app = FastAPI(
    title="GreenDC Intelligence API",
    version="0.1.0",
    description="Energy analytics for the NLR ESIF HPC data centre: metrics, PUE, "
                "forecasts, anomalies and recommendations.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

CONF_RANK = {"low": 0, "medium": 1, "high": 2}
METRIC_COLS = ["it_power_kw", "overhead_kw", "facility_kw", "pue_calc", "outdoor_temp_c",
               "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]
WEATHER_ASSUMPTION = ("Outdoor temperature for the next 24 h is taken from the same hour "
                      "yesterday (naive weather forecast).")


# ---------------------------------------------------------------- serialisation
def to_records(df: pd.DataFrame, index_name: str | None = "ts") -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (ISO timestamps, NaN/inf -> null)."""
    out = df.copy()
    if index_name is None:
        out = out.reset_index(drop=True)
    else:
        idx = out.index
        labels = ([t.isoformat() for t in idx] if isinstance(idx, pd.DatetimeIndex)
                  else [str(v) for v in idx])
        out = out.reset_index(drop=True)
        out.insert(0, index_name, labels)
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = [t.isoformat() if pd.notna(t) else None for t in out[c]]
    out = out.replace([np.inf, -np.inf], np.nan).astype(object)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


# ---------------------------------------------------------------- cached loaders
@lru_cache(maxsize=1)
def hourly() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_FILE)


@lru_cache(maxsize=1)
def anomaly_events() -> pd.DataFrame:
    return load_events()


@lru_cache(maxsize=1)
def forecast_bundle() -> dict:
    return joblib.load(MODELS_DIR / "forecast_best_h24.joblib")


@lru_cache(maxsize=32)
def cached_recommendations(as_of: str | None) -> dict:
    recs, as_of_ts = run_recommendations(as_of)
    return {"as_of": as_of_ts.isoformat(), "recommendations": [asdict(r) for r in recs]}


def window(start: str | None, end: str | None, default_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) in facility local time. end is an inclusive date."""
    df = hourly()
    e = df.index.max() + pd.Timedelta(hours=1) if end is None else ts(end) + pd.Timedelta(days=1)
    s = e - pd.Timedelta(days=default_days) if start is None else ts(start)
    if s >= e:
        raise HTTPException(400, "start must be before end")
    return s, e


def read_result(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found - run the pipeline first")
    return pd.read_csv(path)

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict:
    df = hourly()
    return {
        "data_start": df.index.min().isoformat(),
        "data_end": df.index.max().isoformat(),
        "timezone": TZ,
        "hours": int(len(df)),
        "source": "NLR ESIF HPC data centre PUE dataset (Clark & Strelka 2025, DOI 10.7799/3015212)",
    }


@app.get("/api/metrics")
def metrics(start: str | None = None, end: str | None = None,
            freq: Literal["h", "D", "W", "MS"] | None = None) -> dict:
    s, e = window(start, end, 90)
    d = hourly()
    d = d[(d.index >= s) & (d.index < e)]
    if d.empty:
        raise HTTPException(404, "no data in range")
    span_days = (e - s).days
    freq = freq or ("h" if span_days <= 14 else "D" if span_days <= 730 else "W")
    if freq == "h":
        out = d[METRIC_COLS].copy()
    else:
        out = d[METRIC_COLS].resample(freq).mean()
        out["pue_calc"] = period_pue(d, freq)["pue"].reindex(out.index)  # energy-weighted
    return {"freq": freq, "start": s.isoformat(), "end": e.isoformat(),
            "points": to_records(out.round(3))}


@app.get("/api/pue/summary")
def pue_summary() -> dict:
    df = hourly()
    yearly = period_pue(df, "YS")
    clean, _ = exclude_meter_faults(df)
    yearly["pue_excl_meter_faults"] = period_pue(clean, "YS")["pue"].reindex(yearly.index)
    monthly = period_pue(df, "MS")

    def kpi(days: int) -> dict:
        d = df[df.index > df.index.max() - pd.Timedelta(days=days)]
        d = d.dropna(subset=["it_power_kw", "facility_kw"])
        return {"pue": round(float(d["facility_kw"].sum() / d["it_power_kw"].sum()), 4),
                "it_mw": round(float(d["it_power_kw"].mean() / 1000), 3),
                "overhead_kw": round(float(d["overhead_kw"].mean()), 1),
                "facility_mwh": round(float(d["facility_kw"].sum() / 1000), 1)}

    return {"kpis": {"last_7d": kpi(7), "last_30d": kpi(30)},
            "yearly": to_records(yearly.round(4)),
            "monthly": to_records(monthly.round(4))}


@app.get("/api/pue/temperature")
def pue_temperature(start: str = "2025-01-01", end: str | None = None) -> dict:
    bands = pue_by_temp_band(hourly(), start=start, end=end)
    return {"start": start, "end": end, "bands": to_records(bands.round(4), "temp_band_c")}


@app.get("/api/pue/components")
def pue_components(start: str | None = None, end: str | None = None,
                   freq: Literal["D", "W", "MS"] = "MS") -> dict:
    s, e = window(start, end, 730)
    d = hourly()
    d = d[(d.index >= s) & (d.index < e)]
    return {"freq": freq, "points": to_records(component_breakdown(d, freq).round(2))}


@app.get("/api/forecast/test")
def forecast_test(horizon: int = Query(24), start: str | None = None,
                  end: str | None = None) -> dict:
    if horizon not in (1, 24):
        raise HTTPException(400, "horizon must be 1 or 24")
    path = PRED_DIR / f"forecast_test_predictions_h{horizon}.parquet"
    if not path.exists():
        raise HTTPException(404, "test predictions not found - run the pipeline first")
    preds = pd.read_parquet(path)
    if start or end:
        s, e = window(start, end, 30)
        preds = preds[(preds.index >= s) & (preds.index < e)]
    metrics_table = read_result(f"forecast_metrics_h{horizon}.csv")
    return {"horizon": horizon, "points": to_records(preds.round(1)),
            "metrics": to_records(metrics_table.round(4), None)}


@app.get("/api/forecast/next")
def forecast_next() -> dict:
    df, bundle = hourly(), forecast_bundle()
    h = bundle["horizon"]
    last = df.index.max()
    future = pd.date_range(last + pd.Timedelta(hours=1), periods=h, freq="h")
    ext = df.reindex(df.index.append(future))
    ext.loc[future, "outdoor_temp_c"] = df["outdoor_temp_c"].reindex(
        future - pd.Timedelta(hours=24)).to_numpy()
    frame = make_forecast_frame(ext, h).loc[future]
    ratio = bundle["model"].predict(frame[bundle["features"]])
    out = pd.DataFrame({"forecast_kw": frame["base"].to_numpy() * (1 + ratio),
                        "persistence_kw": frame["base"].to_numpy()}, index=future)
    history = df.loc[df.index > last - pd.Timedelta(hours=72), ["facility_kw"]]
    return {"generated_from": last.isoformat(), "horizon_h": h,
            "weather_assumption": WEATHER_ASSUMPTION,
            "points": to_records(out.round(1)), "history": to_records(history.round(1))}


@app.get("/api/anomalies")
def anomalies(start: str | None = None, end: str | None = None,
              min_confidence: Literal["low", "medium", "high"] = "medium",
              category: str | None = Query(None, description="comma-separated"),
              detector: str | None = Query(None, description="comma-separated"),
              limit: int = Query(500, ge=1, le=5000)) -> dict:
    ev = anomaly_events()
    s, e = window(start, end, 365)
    sel = ev[(ev["end"] >= s) & (ev["start"] < e)]
    sel = sel[sel["confidence"].map(CONF_RANK) >= CONF_RANK[min_confidence]]
    if category:
        sel = sel[sel["category"].isin(category.split(","))]
    if detector:
        sel = sel[sel["detector"].isin(detector.split(","))]
    by_category = {k: int(v) for k, v in sel.groupby("category").size().items()}
    sel = sel.sort_values("start", ascending=False).head(limit)
    return {"count": int(len(sel)), "by_category": by_category,
            "events": to_records(sel.round({"impact_kwh": 1, "peak_score": 2}), None)}


@app.get("/api/recommendations")
def recommendations(as_of: str | None = None) -> dict:
    try:
        return cached_recommendations(as_of)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/evaluation")
def evaluation() -> dict:
    files = {"forecast_h24": "forecast_metrics_h24.csv",
             "forecast_h1": "forecast_metrics_h1.csv",
             "overhead_model": "overhead_model_metrics.csv",
             "anomaly_recall": "anomaly_recall_summary.csv"}
    out = {}
    for key, name in files.items():
        path = RESULTS_DIR / name
        out[key] = to_records(pd.read_csv(path).round(4), None) if path.exists() else None
    return out