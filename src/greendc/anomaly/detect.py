"""Anomaly detection (Objective 3): four complementary detectors -> one events table.

1. overhead_residual : actual vs model-expected overhead
                       model hours:    (robust z > 4 AND |dev| > 5 %) OR hard rule
                       fallback hours: hard rule only (weather missing -> crude reference)
                       hard rule:      |dev| > 20 % AND |dev| > 15 kW
                       (stays sensitive after structural changes; kW floor keeps it
                        actionable when total overhead is small, e.g. 2019-2022)
2. load_change       : facility power vs trailing 7-day median
                       flag if (robust z > 4 AND |dev| > 25 %) OR |dev| > 40 %
3. isolation_forest  : multivariate outliers on scale-free features
4. meter_fault       : cooling/HVAC/pump meter stuck >= 12 h (on a meter that normally varies),
                       or overhead < 50 % of its 30-day median for >= 24 h
Confidence: high = >=2 detectors agree; medium = lasts >=3 h; low = otherwise.
Category:   data_quality (meter faults) | sustained_change (>= 72 h) | operational (alerts)
Requires: python -m greendc.models.overhead (run first)
Run:      python -m greendc.anomaly.detect
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from greendc.config import HOURLY_FILE, MODEL_START, PRED_DIR, RESULTS_DIR, SEED, TRAIN_END, ts
from greendc.pue.metrics import COMPONENTS

Z_THRESH = 4.0
RESID_MIN_PCT = 5.0            # statistical rule (model hours only)
RESID_HARD_PCT = 20.0          # hard rule: relative deviation ...
RESID_HARD_MIN_KW = 15.0       # ... AND absolute deviation an operator would act on
LOAD_MIN_PCT = 25.0            # HPC load routinely moves +/-10 % week to week
LOAD_HARD_PCT = 40.0           # hard rule for load
STUCK_HOURS = 12
STUCK_COMPONENTS = ["cooling_kw", "hvac_kw", "pump_kw"]   # plug/light excluded: small, coarse-stepped
STUCK_MIN_NORMAL_STD = 0.5     # kW: only call a meter "stuck" if it normally varies
UNDERREPORT_HOURS = 24
IF_CONTAMINATION = 0.005
EPS = 1e-6
SUSTAINED_HOURS = 72           # longer events = configuration/capacity change, not an alert


# ---------------------------------------------------------------- helpers
def robust_z(x: pd.Series, window: str = "30D", min_periods: int = 168) -> pd.Series:
    """Robust z-score vs the trailing window (excludes the current point; NaNs skipped)."""
    past = x.shift(1)
    med = past.rolling(window, min_periods=min_periods).median()
    mad = (past - med).abs().rolling(window, min_periods=min_periods).median()
    return (x - med) / (1.4826 * mad + EPS)


def trailing_rel(s: pd.Series, window: str = "7D") -> pd.Series:
    return s / s.shift(1).rolling(window, min_periods=96).median() - 1


def run_length(flag: pd.Series) -> pd.Series:
    """Length of the consecutive-True run each row belongs to (0 where False)."""
    grp = (flag != flag.shift()).cumsum()
    return flag.groupby(grp).transform("size").where(flag, 0)


def detector_frame(flag, score, kind, impact, index) -> pd.DataFrame:
    return pd.DataFrame({"flag": pd.Series(flag, index=index).fillna(False).astype(bool),
                         "score": score, "kind": kind, "impact_kwh": impact}, index=index)


# ---------------------------------------------------------------- detectors
def detect_overhead_residual(exp: pd.DataFrame) -> pd.DataFrame:
    resid_kw = exp["actual"] - exp["expected"]
    resid_pct = 100 * resid_kw / exp["expected"]
    if "fallback" in exp:
        fallback = exp["fallback"].astype("boolean").fillna(False).astype(bool)
    else:
        fallback = pd.Series(False, index=exp.index)

    # z-score uses model-based residuals only, so crude fallback hours don't inflate the spread
    z = robust_z(resid_pct.where(~fallback))
    statistical = ~fallback & (z.abs() > Z_THRESH) & (resid_pct.abs() > RESID_MIN_PCT)
    hard = (resid_pct.abs() > RESID_HARD_PCT) & (resid_kw.abs() > RESID_HARD_MIN_KW)
    flag = statistical | hard

    kind = np.where(resid_pct > 0, "overhead_excess", "overhead_deficit")
    impact = resid_kw.where(flag, 0.0)
    return detector_frame(flag, z.abs(), kind, impact, exp.index)


def detect_load_change(df: pd.DataFrame) -> pd.DataFrame:
    fac = df["facility_kw"]
    ref = fac.shift(1).rolling("7D", min_periods=96).median()
    rel_pct = 100 * (fac / ref - 1)
    z = robust_z(rel_pct)
    flag = ((z.abs() > Z_THRESH) & (rel_pct.abs() > LOAD_MIN_PCT)) | (rel_pct.abs() > LOAD_HARD_PCT)
    kind = np.where(rel_pct < 0, "load_drop", "load_spike")
    return detector_frame(flag, z.abs(), kind, (fac - ref).where(flag, 0.0), df.index)


def detect_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "facility_rel": trailing_rel(df["facility_kw"]),
        "overhead_rel": trailing_rel(df["overhead_kw"]),
        "pue_dev": df["pue_calc"] - df["pue_calc"].shift(1).rolling("7D", min_periods=96).median(),
        "cooling_share_dev": df["cooling_share"]
        - df["cooling_share"].shift(1).rolling("7D", min_periods=96).median(),
        "temp_c": df["outdoor_temp_c"],
    }).replace([np.inf, -np.inf], np.nan).dropna()
    iso = IsolationForest(n_estimators=300, contamination=IF_CONTAMINATION, random_state=SEED)
    iso.fit(X[X.index < ts(TRAIN_END)])
    score = pd.Series(-iso.score_samples(X), index=X.index).reindex(df.index)
    flag = pd.Series(iso.predict(X) == -1, index=X.index).reindex(df.index, fill_value=False)
    return detector_frame(flag, score, "multivariate_outlier", 0.0, df.index)


def detect_meter_faults(df: pd.DataFrame) -> pd.DataFrame:
    # Under-reporting first (takes label priority): overhead < 50 % of its 30-day median for >= 24 h
    oh = df["overhead_kw"]
    low = oh < 0.5 * oh.shift(1).rolling("30D", min_periods=168).median()
    under = run_length(low) >= UNDERREPORT_HOURS
    flag = under.copy()
    kind = pd.Series(np.where(under, "overhead_underreport", ""), index=df.index, dtype=object)

    # Stuck meter: identical hourly values for STUCK_HOURS on a meter that normally varies.
    # max - min is exact; rolling std accumulates float error on flat segments.
    for c in STUCK_COMPONENTS:
        normally_varies = df[c].shift(1).rolling("30D", min_periods=168).std() > STUCK_MIN_NORMAL_STD
        roll = df[c].rolling(STUCK_HOURS)
        flat = (roll.max() - roll.min()).lt(EPS)
        end = flat & df[c].notna() & normally_varies
        whole = end.astype(int).rolling(STUCK_HOURS, min_periods=1).max() \
                   .shift(-(STUCK_HOURS - 1)).fillna(0).astype(bool)
        kind = kind.mask(whole & (kind == ""), f"stuck_meter:{c}")
        flag |= whole
    return detector_frame(flag, flag.astype(float), kind, 0.0, df.index)


# ---------------------------------------------------------------- events
def to_events(name: str, det: pd.DataFrame) -> pd.DataFrame:
    flag = det["flag"]
    if not flag.any():
        return pd.DataFrame()
    event_id = (flag != flag.shift()).cumsum()[flag]
    d = det[flag].assign(event=event_id).rename_axis("ts").reset_index()
    ev = d.groupby("event").agg(
        start=("ts", "min"), end=("ts", "max"), duration_h=("ts", "size"),
        peak_score=("score", "max"), impact_kwh=("impact_kwh", "sum"),
        kind=("kind", lambda s: s.mode().iat[0]))
    return ev.assign(detector=name).reset_index(drop=True)


def build_events(detectors: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine detector outputs into an hourly flag table and a scored events table."""
    hourly = pd.DataFrame({f"{k}_flag": v["flag"] for k, v in detectors.items()})
    hourly["n_detectors"] = hourly.sum(axis=1)

    events = pd.concat([to_events(k, v) for k, v in detectors.items()], ignore_index=True)
    events["max_agreement"] = [int(hourly.loc[s:e, "n_detectors"].max())
                               for s, e in zip(events["start"], events["end"])]
    events["confidence"] = np.select(
        [events["max_agreement"] >= 2, events["duration_h"] >= 3], ["high", "medium"], "low")
    events["category"] = np.select(
        [events["detector"] == "meter_fault",
         (events["duration_h"] >= SUSTAINED_HOURS)
         & events["detector"].isin(["load_change", "overhead_residual"])],
        ["data_quality", "sustained_change"], "operational")
    events = events.sort_values("start").reset_index(drop=True)
    return hourly, events


def main() -> None:
    df = pd.read_parquet(HOURLY_FILE)
    df = df[df.index >= ts(MODEL_START)]
    exp = pd.read_parquet(PRED_DIR / "overhead_expected.parquet").reindex(df.index)

    detectors = {
        "overhead_residual": detect_overhead_residual(exp),
        "load_change": detect_load_change(df),
        "isolation_forest": detect_isolation_forest(df),
        "meter_fault": detect_meter_faults(df),
    }
    hourly, events = build_events(detectors)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(RESULTS_DIR / "anomaly_events.csv", index=False)
    hourly.to_parquet(PRED_DIR / "anomaly_hourly.parquet")

    pd.set_option("display.width", 160)
    print("% of hours flagged per detector:")
    print((100 * hourly.drop(columns="n_detectors").mean()).round(2))

    print("\nEvents per detector per year:")
    print(pd.crosstab(events["detector"], events["start"].dt.year))

    print("\nEvents by kind and confidence:")
    print(pd.crosstab(events["kind"], events["confidence"]))

    cat = events.groupby("category").agg(events=("start", "size"), hours=("duration_h", "sum"))
    cat["pct_of_hours"] = (100 * cat["hours"] / len(hourly)).round(2)
    print("\nEvents by category:")
    print(cat)

    top = events[events.detector == "overhead_residual"] \
        .assign(abs_impact=lambda e: e["impact_kwh"].abs()).nlargest(10, "abs_impact")
    cols = ["start", "duration_h", "kind", "impact_kwh", "peak_score", "confidence", "category"]
    print("\nTop 10 overhead-residual events by energy impact:")
    print(top[cols].round({"impact_kwh": 1, "peak_score": 1}).to_string(index=False))

    mf = events[(events.detector == "meter_fault") & events["start"].dt.year.isin([2020, 2021])]
    print("\nMeter-fault events 2020-21 (should include the low-PUE plateaus):")
    print(mf[["start", "end", "duration_h", "kind"]].head(20).to_string(index=False))

    print(f"\nSaved {len(events):,} events -> {RESULTS_DIR / 'anomaly_events.csv'}")


if __name__ == "__main__":
    main()