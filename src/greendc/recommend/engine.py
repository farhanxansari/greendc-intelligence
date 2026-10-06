"""Recommendation engine (Objective 5): turns analytics into evidence-backed actions.

Each rule reads computed results (PUE, anomaly events, component breakdown), checks a
threshold from configs/recommendation_rules.yaml, and emits a Recommendation with
severity, action text, the evidence numbers, and estimated savings where estimable.
--as-of replays any point in history (detectors are trailing-window based).

Requires: anomaly.detect (for docs/results/anomaly_events.csv)
Run: python -m greendc.recommend.engine
     python -m greendc.recommend.engine --as-of 2024-06-01
"""
import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import yaml

from greendc.config import HOURLY_FILE, RESULTS_DIR, ROOT, TZ, ts
from greendc.pue.metrics import (COMPONENTS, FILTER_PUMP_KW, efficiency_gap,
                                 exclude_meter_faults, period_pue)
from greendc.anomaly.detect import SUSTAINED_HOURS

RULES_FILE = ROOT / "configs" / "recommendation_rules.yaml"
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Recommendation:
    rule_id: str
    title: str
    severity: str
    category: str
    action: str
    evidence: dict = field(default_factory=dict)
    est_savings_mwh: float | None = None
    period_start: str = ""
    period_end: str = ""


def _since(ctx: dict, days: int) -> pd.Timestamp:
    return ctx["as_of"] - pd.Timedelta(days=days)


def _meter_label(kind: str) -> str:
    if kind.startswith("stuck_meter:"):
        return kind.split(":")[1].replace("_kw", "") + " meter (stuck readings)"
    if kind == "overhead_underreport":
        return "overhead meters (under-reporting)"
    return kind


# ---------------------------------------------------------------- rules
def r_meter_fault(ctx: dict, cfg: dict) -> list[Recommendation]:
    ev = ctx["events"]
    sel = ev[(ev["category"] == "data_quality") & (ev["end"] >= _since(ctx, cfg["lookback_days"]))]
    out = []
    for kind, g in sel.groupby("kind"):
        hours, meter, first = int(g["duration_h"].sum()), _meter_label(kind), g["start"].min()
        out.append(Recommendation(
            rule_id="meter_fault", category="data_quality",
            severity="high" if hours >= cfg["high_hours"] else "medium",
            title=cfg["title"].format(meter=meter),
            action=cfg["action"].format(meter=meter, hours=hours, first=f"{first:%Y-%m-%d}"),
            evidence={"events": int(len(g)), "hours": hours},
            period_start=str(first), period_end=str(g["end"].max())))
    return out


def r_sustained_overhead_increase(ctx: dict, cfg: dict) -> list[Recommendation]:
    ev = ctx["events"]
    sel = ev[(ev["category"] == "sustained_change") & (ev["detector"] == "overhead_residual")
             & (ev["kind"] == "overhead_excess") & (ev["start"] >= _since(ctx, cfg["lookback_days"]))]
    out = []
    for _, e in sel.iterrows():
        mwh, kw = e["impact_kwh"] / 1000, e["impact_kwh"] / e["duration_h"]
        out.append(Recommendation(
            rule_id="sustained_overhead_increase", category="sustained_change", severity="high",
            title=cfg["title"].format(start=f"{e['start']:%Y-%m-%d}"),
            action=cfg["action"].format(mean_kw=kw, hours=int(e["duration_h"]), excess_mwh=mwh),
            evidence={"excess_mwh": round(float(mwh), 2), "mean_excess_kw": round(float(kw), 1),
                      "hours": int(e["duration_h"]), "confidence": e["confidence"]},
            period_start=str(e["start"]), period_end=str(e["end"])))
    return out


def _operational(ctx: dict, cfg: dict, detector: str, kind: str) -> pd.DataFrame:
    ev = ctx["events"]
    return ev[(ev["category"] == "operational") & (ev["detector"] == detector)
              & (ev["kind"] == kind) & ev["confidence"].isin(["high", "medium"])
              & (ev["end"] >= _since(ctx, cfg["lookback_days"]))]


def r_operational_overhead_excess(ctx: dict, cfg: dict) -> list[Recommendation]:
    sel = _operational(ctx, cfg, "overhead_residual", "overhead_excess")
    if len(sel) < cfg["min_events"]:
        return []
    mwh = float(sel["impact_kwh"].sum() / 1000)
    return [Recommendation(
        rule_id="operational_overhead_excess", category="operations",
        severity="high" if mwh >= cfg["high_mwh"] else "medium",
        title=cfg["title"].format(n=len(sel), days=cfg["lookback_days"]),
        action=cfg["action"].format(excess_mwh=mwh),
        evidence={"events": int(len(sel)), "excess_mwh": round(mwh, 2),
                  "hours": int(sel["duration_h"].sum())},
        est_savings_mwh=round(mwh, 2),
        period_start=str(sel["start"].min()), period_end=str(sel["end"].max()))]


def r_efficiency_gap(ctx: dict, cfg: dict) -> list[Recommendation]:
    gap = efficiency_gap(period_pue(ctx["clean"], "D"))
    win = gap[gap.index > _since(ctx, cfg["lookback_days"])].dropna(subset=["best_pue"])
    win = win[win["coverage_pct"] >= cfg.get("min_day_coverage_pct", 50)]
    if len(win) < cfg.get("min_valid_days", 14):
        return []  # not enough trustworthy data to judge efficiency
    mwh, oh = float(win["avoidable_mwh"].sum()), float(win["overhead_mwh"].sum())
    pct = 100 * mwh / oh if oh > 0 else 0.0
    if pct < cfg["min_pct_of_overhead"] or mwh < cfg.get("min_avoidable_mwh", 1.0):
        return []
    pue = float(win["facility_mwh"].sum() / win["it_mwh"].sum())
    best, days = float(win["best_pue"].iloc[-1]), len(win)
    annual = mwh * 365 / days

    action = cfg["action"].format(pct=pct, mwh=mwh, days=days, annual=annual)
    evidence = {"pue": round(pue, 4), "best_pue": round(best, 4),
                "avoidable_mwh": round(mwh, 2), "pct_of_overhead": round(pct, 1),
                "annualised_mwh": round(annual, 1), "valid_days": days}

    # The best-achieved baseline uses the trailing 90 days. If an overhead step-change
    # happened in that window, most of this gap IS that change: say so, don't double-count.
    ev = ctx["events"]
    step = ev[(ev["category"] == "sustained_change") & (ev["detector"] == "overhead_residual")
              & (ev["kind"] == "overhead_excess") & (ev["start"] >= _since(ctx, 90))]
    if len(step):
        first = step["start"].min()
        action += (f" Note: the best-achieved baseline includes the period before the overhead "
                   f"step-change on {first:%Y-%m-%d}, so most of this gap is that change "
                   f"(see the commissioning-review recommendation).")
        evidence["baseline_predates_step_change"] = True

    return [Recommendation(
        rule_id="efficiency_gap", category="efficiency",
        severity="high" if pct >= cfg["high_pct_of_overhead"] else "medium",
        title=cfg["title"].format(pue=pue, best=best),
        action=action, evidence=evidence, est_savings_mwh=round(mwh, 2),
        period_start=str(win.index.min()), period_end=str(win.index.max()))]


def r_hvac_share(ctx: dict, cfg: dict) -> list[Recommendation]:
    df, as_of = ctx["df"], ctx["as_of"]
    days = pd.Timedelta(days=cfg["lookback_days"])
    year = pd.Timedelta(days=365)

    def share(d: pd.DataFrame) -> tuple[float, float]:
        d = d.dropna(subset=COMPONENTS)
        oh = d[COMPONENTS].sum(axis=1) + FILTER_PUMP_KW
        return 100 * d["hvac_kw"].sum() / oh.sum(), d["hvac_kw"].mean()

    recent = df[df.index > as_of - days]
    prev = df[(df.index > as_of - year - days) & (df.index <= as_of - year)]
    if len(recent.dropna(subset=COMPONENTS)) < 168 or len(prev.dropna(subset=COMPONENTS)) < 168:
        return []
    s_now, kw_now = share(recent)
    s_prev, kw_prev = share(prev)
    if s_now < cfg["min_share_pct"] or s_now - s_prev < cfg["min_increase_pts"]:
        return []
    return [Recommendation(
        rule_id="hvac_share", category="efficiency", severity="medium",
        title=cfg["title"].format(share=s_now, prev=s_prev),
        action=cfg["action"].format(prev_kw=kw_prev, kw=kw_now),
        evidence={"hvac_share_pct": round(float(s_now), 1), "prev_share_pct": round(float(s_prev), 1),
                  "hvac_kw": round(float(kw_now), 1), "prev_hvac_kw": round(float(kw_prev), 1)},
        period_start=str(recent.index.min()), period_end=str(recent.index.max()))]


def r_hot_weather(ctx: dict, cfg: dict) -> list[Recommendation]:
    d = ctx["clean"]
    d = d[d.index > _since(ctx, cfg["lookback_days"])].dropna(subset=["outdoor_temp_c", "overhead_kw"])
    lo, hi = cfg["mild_band_c"]
    mild = d[(d["outdoor_temp_c"] >= lo) & (d["outdoor_temp_c"] < hi)]["overhead_kw"]
    hot_mask = d["outdoor_temp_c"] >= cfg["hot_temp_c"]
    hours = int(hot_mask.sum())
    if hours < cfg["min_hot_hours"] or len(mild) < cfg["min_hot_hours"]:
        return []
    uplift = float(d.loc[hot_mask, "overhead_kw"].mean() - mild.mean())
    if not np.isfinite(uplift) or uplift < cfg["min_uplift_kw"]:
        return []
    frac = cfg["recoverable_fraction"]
    mwh = uplift * hours / 1000 * frac
    return [Recommendation(
        rule_id="hot_weather", category="efficiency", severity="low",
        title=cfg["title"].format(uplift=uplift, hot=cfg["hot_temp_c"]),
        action=cfg["action"].format(uplift=uplift, hot=cfg["hot_temp_c"], lo=lo, hi=hi,
                                    hours=hours, days=cfg["lookback_days"], frac=frac),
        evidence={"uplift_kw": round(uplift, 1), "hot_hours": hours,
                  "mild_overhead_kw": round(float(mild.mean()), 1),
                  "assumed_recoverable_fraction": frac},
        est_savings_mwh=round(mwh, 2),
        period_start=str(d.index.min()), period_end=str(d.index.max()))]


def r_load_drops(ctx: dict, cfg: dict) -> list[Recommendation]:
    sel = _operational(ctx, cfg, "load_change", "load_drop")
    if len(sel) < cfg["min_events"]:
        return []
    return [Recommendation(
        rule_id="load_drops", category="operations", severity="medium",
        title=cfg["title"].format(n=len(sel), days=cfg["lookback_days"]),
        action=cfg["action"],
        evidence={"events": int(len(sel)), "hours": int(sel["duration_h"].sum()),
                  "lost_load_mwh": round(float(-sel["impact_kwh"].sum() / 1000), 2)},
        period_start=str(sel["start"].min()), period_end=str(sel["end"].max()))]


RULES = {
    "meter_fault": r_meter_fault,
    "sustained_overhead_increase": r_sustained_overhead_increase,
    "operational_overhead_excess": r_operational_overhead_excess,
    "efficiency_gap": r_efficiency_gap,
    "hvac_share": r_hvac_share,
    "hot_weather": r_hot_weather,
    "load_drops": r_load_drops,
}


# ---------------------------------------------------------------- engine
def load_events() -> pd.DataFrame:
    ev = pd.read_csv(RESULTS_DIR / "anomaly_events.csv")
    for c in ("start", "end"):
        ev[c] = pd.to_datetime(ev[c], utc=True).dt.tz_convert(TZ)
    return ev

def clip_events_to(events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return events as they would have looked at `as_of` (no look-ahead).

    Events still in progress are cut off at as_of: duration and impact are scaled
    to the elapsed part, and an event too young to be 'sustained' is re-labelled.
    """
    ev = events[events["start"] <= as_of].copy()
    ongoing = ev["end"] > as_of
    if ongoing.any():
        elapsed_h = ((as_of - ev.loc[ongoing, "start"]) / pd.Timedelta(hours=1)).astype(int) + 1
        frac = elapsed_h / ev.loc[ongoing, "duration_h"]
        ev.loc[ongoing, "impact_kwh"] = ev.loc[ongoing, "impact_kwh"] * frac
        ev.loc[ongoing, "duration_h"] = elapsed_h
        ev.loc[ongoing, "end"] = as_of
        too_young = ongoing & (ev["category"] == "sustained_change") & (ev["duration_h"] < SUSTAINED_HOURS)
        ev.loc[too_young, "category"] = "operational"
    ev["ongoing"] = ongoing
    return ev


def run(as_of: str | None = None) -> tuple[list[Recommendation], pd.Timestamp]:
    with open(RULES_FILE, encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    df = pd.read_parquet(HOURLY_FILE)
    as_of_ts = df.index.max() if as_of is None else ts(as_of)
    df = df[df.index <= as_of_ts]
    clean, _ = exclude_meter_faults(df)
    events = load_events()
    ctx = {"df": df, "clean": clean, "events": clip_events_to(events, as_of_ts),
           "as_of": as_of_ts}

    recs: list[Recommendation] = []
    for name, rule in RULES.items():
        recs += rule(ctx, cfg_all[name])

    factor = (cfg_all.get("global") or {}).get("grid_kg_co2_per_kwh")
    for r in recs:
        if factor and r.est_savings_mwh:
            r.evidence["est_co2_t"] = round(r.est_savings_mwh * factor, 2)

    recs.sort(key=lambda r: (SEVERITY_ORDER[r.severity], -(r.est_savings_mwh or 0)))
    return recs, as_of_ts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: latest data)")
    args = parser.parse_args()

    recs, as_of_ts = run(args.as_of)
    tag = f"{as_of_ts:%Y%m%d}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"recommendations_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"as_of": str(as_of_ts), "recommendations": [asdict(r) for r in recs]},
                  f, indent=2, default=str)

    print(f"Recommendations as of {as_of_ts:%Y-%m-%d %H:%M}  ({len(recs)} total)\n")
    for i, r in enumerate(recs, 1):
        saving = f"  | est. saving {r.est_savings_mwh} MWh" if r.est_savings_mwh else ""
        print(f"{i}. [{r.severity.upper()}] {r.title}  ({r.category}){saving}")
        print(f"   {r.action}")
        print(f"   evidence: {r.evidence}\n")
    print(f"Saved -> {RESULTS_DIR / f'recommendations_{tag}.json'}")


if __name__ == "__main__":
    main()