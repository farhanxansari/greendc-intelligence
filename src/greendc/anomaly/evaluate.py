"""Evaluate anomaly detectors by injecting labelled synthetic faults into 2025 (test) data.

Per trial: inject one fault of each type at random clean, well-separated times,
recompute derived columns + expected overhead, run all detectors, score.
Run: python -m greendc.anomaly.evaluate   (after models.overhead and anomaly.detect)
"""
import joblib
import numpy as np
import pandas as pd

from greendc.anomaly import detect as D
from greendc.config import HOURLY_FILE, MODEL_START, MODELS_DIR, RESULTS_DIR, SEED, VAL_END, ts
from greendc.models.overhead import make_overhead_frame, predict_expected
from greendc.pue.metrics import COMPONENTS, FILTER_PUMP_KW

N_TRIALS = 30
MIN_SEPARATION_H = 24 * 10
GRACE_H = 6                     # detection may land up to 6 h after the fault ends
WINDOW_H = 80                   # max fault length (72 h) + grace, must be clean
KINDS = ["overhead_excess", "outage", "stuck_meter", "meter_dropout"]
EXPECTED = {"overhead_excess": "overhead_residual", "outage": "load_change",
            "stuck_meter": "meter_fault", "meter_dropout": "meter_fault"}
BUNDLE = joblib.load(MODELS_DIR / "overhead_model.joblib")


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    df["overhead_kw"] = df[COMPONENTS].sum(axis=1, min_count=4) + FILTER_PUMP_KW
    df["facility_kw"] = df["it_power_kw"] + df["overhead_kw"]
    df["pue_calc"] = df["facility_kw"] / df["it_power_kw"]
    df["cooling_share"] = (df["cooling_kw"] + df["hvac_kw"] + df["pump_kw"]
                           + FILTER_PUMP_KW) / df["facility_kw"]
    return df


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    exp = predict_expected(make_overhead_frame(df), BUNDLE)
    dets = {
        "overhead_residual": D.detect_overhead_residual(exp),
        "load_change": D.detect_load_change(df),
        "isolation_forest": D.detect_isolation_forest(df),
        "meter_fault": D.detect_meter_faults(df),
    }
    return pd.DataFrame({k: v["flag"] for k, v in dets.items()})


def inject(df: pd.DataFrame, kind: str, pos: int, rng) -> tuple[int, float, str]:
    """Modify df in place; return (duration_h, magnitude, detail)."""
    if kind == "overhead_excess":
        dur, mag, detail = int(rng.integers(6, 25)), float(rng.uniform(0.15, 0.40)), "cooling_kw"
        w = df.index[pos:pos + dur]
        df.loc[w, "cooling_kw"] += mag * df.loc[w, "overhead_kw"]
    elif kind == "outage":
        dur, mag, detail = int(rng.integers(4, 25)), float(rng.uniform(0.3, 0.6)), "it_power_kw"
        w = df.index[pos:pos + dur]
        df.loc[w, "it_power_kw"] *= mag
    elif kind == "stuck_meter":
        dur, mag = int(rng.integers(18, 49)), 0.0
        detail = str(rng.choice(["cooling_kw", "pump_kw"]))
        w = df.index[pos:pos + dur]
        df.loc[w, detail] = df[detail].iloc[pos]
    else:  # meter_dropout
        dur, mag, detail = int(rng.integers(36, 73)), 0.0, "hvac_kw"
        w = df.index[pos:pos + dur]
        df.loc[w, "hvac_kw"] = 0.0
    return dur, mag, detail


def choose_starts(cands: np.ndarray, k: int, rng) -> list[int]:
    chosen: list[int] = []
    for p in rng.permutation(cands):
        if all(abs(int(p) - c) >= MIN_SEPARATION_H for c in chosen):
            chosen.append(int(p))
        if len(chosen) == k:
            break
    return chosen


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson confidence interval for a proportion, in %."""
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def main() -> None:
    clean = pd.read_parquet(HOURLY_FILE)
    clean = clean[clean.index >= ts(MODEL_START)]
    print("Running detectors on clean data ...")
    clean_flags = run_all(clean)

    bad = clean["is_gap"].to_numpy() | clean_flags.any(axis=1).to_numpy()
    test_pos = np.where(clean.index >= ts(VAL_END))[0]
    cands = np.array([p for p in test_pos
                      if p + WINDOW_H < len(clean) and not bad[p:p + WINDOW_H].any()])
    print(f"clean candidate start hours in 2025: {len(cands):,}")

    records, false_alarms = [], []
    in_test = clean.index >= ts(VAL_END)
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(SEED + trial)
        d = clean.copy()
        starts = choose_starts(cands, len(KINDS), rng)
        truth = []
        for kind, pos in zip(KINDS, starts):
            dur, mag, detail = inject(d, kind, pos, rng)
            truth.append((kind, pos, dur, mag, detail))
        flags = run_all(recompute(d))

        injected = np.zeros(len(d), dtype=bool)
        for kind, pos, dur, mag, detail in truth:
            injected[pos:pos + dur + GRACE_H] = True
            win = flags.iloc[pos:pos + dur + GRACE_H]
            for det in flags.columns:
                hit = np.flatnonzero(win[det].to_numpy())
                records.append({"trial": trial, "kind": kind, "detector": det,
                                "detail": detail, "start": d.index[pos],
                                "duration_h": dur, "magnitude": mag,
                                "detected": hit.size > 0,
                                "delay_h": int(hit[0]) if hit.size else np.nan})
        for det in flags.columns:
            new = flags[det].to_numpy() & ~clean_flags[det].to_numpy() & ~injected & in_test
            false_alarms.append({"trial": trial, "detector": det, "new_fa_hours": int(new.sum())})
        print(f"trial {trial + 1}/{N_TRIALS} done")

    rec = pd.DataFrame(records)

    # ---- recall per fault type x detector, plus "any detector"
    any_det = rec.groupby(["trial", "kind"])["detected"].any().groupby("kind").mean()
    recall = rec.pivot_table(index="kind", columns="detector", values="detected", aggfunc="mean")
    recall["ANY"] = any_det
    recall = (100 * recall).round(1)

    # ---- expected-detector recall with 95 % CI and delay (report table)
    exp_rows = rec[rec["detector"] == rec["kind"].map(EXPECTED)]
    delay = exp_rows.groupby("kind")["delay_h"].median().rename("median_delay_h")
    summary = exp_rows.groupby("kind")["detected"].agg(detected="sum", trials="count")
    summary["recall_%"] = (100 * summary["detected"] / summary["trials"]).round(1)
    ci = pd.DataFrame([wilson(int(k), int(n)) for k, n in zip(summary["detected"], summary["trials"])],
                      index=summary.index, columns=["ci95_low_%", "ci95_high_%"])
    summary = summary.join(ci).join(delay)
    summary.insert(0, "expected_detector", summary.index.map(EXPECTED))

    # ---- overhead-excess recall by injected magnitude
    oe = exp_rows[exp_rows["kind"] == "overhead_excess"]
    by_mag = oe.groupby(pd.cut(oe["magnitude"], [0.15, 0.25, 0.40], include_lowest=True),
                        observed=True)["detected"].agg(["mean", "count"])
    by_mag["mean"] = (100 * by_mag["mean"]).round(1)
    by_mag = by_mag.rename(columns={"mean": "recall_%", "count": "faults"})

    # ---- stuck-meter recall by which meter was stuck
    sm = exp_rows[exp_rows["kind"] == "stuck_meter"]
    by_meter = sm.groupby("detail")["detected"].agg(["mean", "count"])
    by_meter["mean"] = (100 * by_meter["mean"]).round(1)
    by_meter = by_meter.rename(columns={"mean": "recall_%", "count": "faults"})

    # ---- false alarms
    n_test = int(in_test.sum())
    fa = pd.DataFrame(false_alarms).groupby("detector")["new_fa_hours"].mean().to_frame()
    fa["pct_of_test_hours"] = (100 * fa["new_fa_hours"] / n_test).round(3)

    misses = exp_rows[~exp_rows["detected"]][["trial", "kind", "detail", "start",
                                              "duration_h", "magnitude"]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_csv(RESULTS_DIR / "anomaly_injection_records.csv", index=False)
    recall.to_csv(RESULTS_DIR / "anomaly_recall.csv")
    summary.to_csv(RESULTS_DIR / "anomaly_recall_summary.csv")
    fa.to_csv(RESULTS_DIR / "anomaly_false_alarms.csv")

    pd.set_option("display.width", 160)
    print("\nRecall % (rows = injected fault type, cols = detector):")
    print(recall)
    print("\nExpected-detector recall with 95% CI (use this table in the report):")
    print(summary)
    print("\nOverhead-excess recall by injected magnitude:")
    print(by_mag)
    print("\nStuck-meter recall by meter:")
    print(by_meter)
    print("\nNew false-alarm hours per trial (outside injected windows):")
    print(fa)
    print("\nMissed faults (expected detector):")
    print(misses.round({"magnitude": 3}).to_string(index=False) if len(misses) else "none")
    print("\nNote: stuck-meter flags are back-filled to the fault start, so their real-time "
          f"delay is >= {D.STUCK_HOURS} h by construction; meter dropouts are confirmed as "
          f"meter faults after {D.UNDERREPORT_HOURS} h.")


if __name__ == "__main__":
    main()