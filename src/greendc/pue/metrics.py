"""PUE and efficiency metrics (Objective 4).

- Period PUE is energy-weighted: sum(facility kWh) / sum(IT kWh), as recommended
  by The Green Grid - NOT the mean of hourly ratios.
- "Avoidable overhead": daily PUE vs the facility's own best-achieved PUE
  (10th percentile, trailing 90 days), converted to excess MWh.
- Yearly PUE is also reported excluding hours flagged as meter faults by the
  anomaly pipeline (run anomaly.detect first for that table).
Run: python -m greendc.pue.metrics
"""
import pandas as pd

from greendc.config import HOURLY_FILE, MODEL_START, PRED_DIR, RESULTS_DIR, ts

COMPONENTS = ["cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]
FILTER_PUMP_KW = 2.67


def load() -> pd.DataFrame:
    return pd.read_parquet(HOURLY_FILE)


def period_pue(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Energy-weighted PUE per period. Hourly mean kW == kWh for that hour."""
    valid = df.dropna(subset=["it_power_kw", "facility_kw"])
    g = valid.resample(freq)
    out = pd.DataFrame({
        "it_mwh": g["it_power_kw"].sum() / 1000,
        "facility_mwh": g["facility_kw"].sum() / 1000,
        "valid_hours": g["it_power_kw"].count(),
    })
    out["overhead_mwh"] = out["facility_mwh"] - out["it_mwh"]
    out["pue"] = out["facility_mwh"] / out["it_mwh"]
    expected = df.resample(freq).size().reindex(out.index)
    out["coverage_pct"] = 100 * out["valid_hours"] / expected
    return out[out["valid_hours"] > 0]


def component_breakdown(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Mean kW and % share of each overhead component."""
    comp = df[COMPONENTS].assign(filter_pump_kw=FILTER_PUMP_KW).resample(freq).mean()
    share = comp.div(comp.sum(axis=1), axis=0).mul(100).add_suffix("_share_pct")
    return comp.join(share).dropna(how="all")


def pue_by_temp_band(df: pd.DataFrame, start: str = MODEL_START, end: str | None = None,
                     step: int = 5) -> pd.DataFrame:
    """Energy-weighted PUE per outdoor-temperature band within a period."""
    d = df[df.index >= ts(start)]
    if end:
        d = d[d.index < ts(end)]
    d = d.dropna(subset=["outdoor_temp_c", "it_power_kw", "facility_kw"])
    bands = pd.cut(d["outdoor_temp_c"], range(-25, 45, step))
    g = d.groupby(bands, observed=True)
    out = pd.DataFrame({
        "pue": g["facility_kw"].sum() / g["it_power_kw"].sum(),
        "hours": g.size(),
        "overhead_kw_mean": g["overhead_kw"].mean(),
    })
    return out[out["hours"] >= 24]


def efficiency_gap(daily: pd.DataFrame, window: str = "90D", q: float = 0.10) -> pd.DataFrame:
    """Daily PUE vs the facility's own best-achieved PUE -> avoidable overhead MWh."""
    out = daily.copy()
    out["best_pue"] = out["pue"].rolling(window, min_periods=30).quantile(q)
    out["pue_gap"] = (out["pue"] - out["best_pue"]).clip(lower=0)
    out["avoidable_mwh"] = out["pue_gap"] * out["it_mwh"]
    return out


def exclude_meter_faults(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Blank IT/facility on hours the anomaly pipeline flagged as meter faults."""
    path = PRED_DIR / "anomaly_hourly.parquet"
    if not path.exists():
        return df, pd.Series(False, index=df.index)
    mf = pd.read_parquet(path)["meter_fault_flag"].reindex(df.index, fill_value=False).astype(bool)
    clean = df.copy()
    clean.loc[mf.to_numpy(), ["it_power_kw", "facility_kw"]] = float("nan")
    return clean, mf


def main() -> None:
    df = load()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    daily = period_pue(df, "D")
    monthly = period_pue(df, "MS")
    yearly = period_pue(df, "YS")

    clean, mf = exclude_meter_faults(df)
    yearly_clean = period_pue(clean, "YS")
    pue_compare = pd.DataFrame({
        "pue_all_hours": yearly["pue"].to_numpy(),
        "pue_excl_meter_faults": yearly_clean["pue"].reindex(yearly.index).to_numpy(),
        "hours_excluded": mf.groupby(mf.index.year).sum().reindex(yearly.index.year).fillna(0).to_numpy(),
    }, index=yearly.index.year)
    pue_compare.index.name = "year"

    gap = efficiency_gap(daily)
    gap_monthly = gap["avoidable_mwh"].resample("MS").sum().to_frame()
    gap_monthly["avoidable_pct_of_overhead"] = (
        100 * gap_monthly["avoidable_mwh"] / monthly["overhead_mwh"].reindex(gap_monthly.index))

    temp_all = pue_by_temp_band(df)                       # 2019+, regime-confounded
    temp_2025 = pue_by_temp_band(df, start="2025-01-01")  # single stable regime

    outputs = {
        "pue_daily.csv": gap, "pue_monthly.csv": monthly, "pue_yearly.csv": yearly,
        "pue_yearly_excl_meter_faults.csv": pue_compare,
        "overhead_components_monthly.csv": component_breakdown(df),
        "pue_by_temp_2019plus.csv": temp_all, "pue_by_temp_2025.csv": temp_2025,
        "avoidable_overhead_monthly.csv": gap_monthly,
    }
    for name, frame in outputs.items():
        frame.to_csv(RESULTS_DIR / name)

    pd.set_option("display.width", 140)
    print("Yearly energy-weighted PUE:")
    print(yearly.assign(year=yearly.index.year).set_index("year").round(3))
    print("\nYearly PUE with vs without meter-fault hours (only 2019+ is checked):")
    print(pue_compare.round(4))
    print("\nPUE by outdoor temperature, 2025 only (single regime):")
    print(temp_2025.round(4))
    recent = gap_monthly[gap_monthly.index >= ts("2024-01-01")]
    print("\nAvoidable overhead vs best-achieved PUE (2024+):")
    print(recent.round(2))
    print(f"\nSaved {len(outputs)} tables -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()