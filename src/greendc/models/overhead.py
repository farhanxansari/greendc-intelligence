"""Expected-overhead model: what overhead SHOULD be given IT load, weather,
time of day and the facility's recent configuration.

Used by: residual anomaly detection, recommendations, (later) what-if.
Target: ratio vs reference = overhead[T] / ref[T] - 1, where
ref = median overhead over the 7 days ending 24 h before T. The reference
adapts to configuration changes (e.g. the April 2024 HVAC step) within ~a week.
If weather features are missing (station gaps), expected = the 7-day reference.
Run: python -m greendc.models.overhead
"""
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from greendc.config import (HOURLY_FILE, MODEL_START, MODELS_DIR, PRED_DIR, RESULTS_DIR,
                            SEED, TRAIN_END, VAL_END, ts)

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
REF_LAG, REF_WINDOW = 24, 168
CLIP = (-0.9, 2.0)
META = ["target", "ref", "target_ratio"]


def make_overhead_frame(df: pd.DataFrame) -> pd.DataFrame:
    oh, it, temp = df["overhead_kw"], df["it_power_kw"], df["outdoor_temp_c"]

    def ref(s: pd.Series, how: str = "median") -> pd.Series:
        r = s.shift(REF_LAG).rolling(REF_WINDOW, min_periods=96)
        return r.median() if how == "median" else r.mean()

    out = pd.DataFrame(index=df.index)
    out["target"] = oh
    out["ref"] = ref(oh)
    out["target_ratio"] = oh / out["ref"] - 1
    out["it_rel"] = it / ref(it) - 1
    out["temp_c"] = temp
    out["temp_dev"] = temp - ref(temp, "mean")
    out["temp_max_24h"] = temp.rolling(24, min_periods=12).max()
    out["rh_pct"] = df["outdoor_rh_pct"]
    idx = df.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return out.replace([np.inf, -np.inf], np.nan)


def predict_expected(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Expected overhead for every hour that has a reference.

    Model prediction where all features exist; otherwise fall back to the
    7-day reference (e.g. during weather-station gaps) so detection never goes blind.
    """
    feats = bundle["features"]
    has_ref = frame["ref"].notna()
    full = has_ref & frame[feats].notna().all(axis=1)
    exp = pd.DataFrame({"actual": frame["target"], "expected": np.nan, "fallback": False},
                       index=frame.index)
    exp.loc[has_ref, "expected"] = frame.loc[has_ref, "ref"]
    exp.loc[has_ref & ~full, "fallback"] = True
    if full.any():
        exp.loc[full, "expected"] = frame.loc[full, "ref"].to_numpy() * (
            1 + bundle["model"].predict(frame.loc[full, feats]))
    return exp


def main() -> None:
    df = pd.read_parquet(HOURLY_FILE)
    frame = make_overhead_frame(df)
    frame = frame[frame.index >= ts(MODEL_START)]
    feats = [c for c in frame.columns if c not in META]
    data = frame.dropna()

    tr = data[data.index < ts(TRAIN_END)]
    va = data[(data.index >= ts(TRAIN_END)) & (data.index < ts(VAL_END))]
    te = data[data.index >= ts(VAL_END)]
    print(f"rows train={len(tr):,} val={len(va):,} test={len(te):,} features={len(feats)}")

    model = LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1",
        random_state=SEED, verbose=-1)
    model.fit(tr[feats], tr["target_ratio"].clip(*CLIP),
              eval_set=[(va[feats], va["target_ratio"].clip(*CLIP))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    bundle = {"model": model, "features": feats, "ref_lag": REF_LAG, "ref_window": REF_WINDOW}

    def to_kw(part: pd.DataFrame) -> np.ndarray:
        return part["ref"].to_numpy() * (1 + model.predict(part[feats]))

    rows = []
    for name, part in [("val", va), ("test", te)]:
        p = to_kw(part)
        mae_m = mean_absolute_error(part["target"], p)
        mae_r = mean_absolute_error(part["target"], part["ref"])
        rows.append({"split": name, "MAE_model_kW": mae_m, "MAE_ref_kW": mae_r,
                     "skill_vs_ref_%": (1 - mae_m / mae_r) * 100,
                     "MAPE_model_%": float(np.mean(np.abs((part["target"] - p) / part["target"])) * 100)})
    metrics = pd.DataFrame(rows).round(3)

    # Expected overhead for every hour with a reference (rows before TRAIN_END are in-sample)
    expected = predict_expected(frame, bundle)
    expected = expected[expected["expected"].notna()]
    expected["residual_kw"] = expected["actual"] - expected["expected"]
    expected["in_sample"] = expected.index < ts(TRAIN_END)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "overhead_model_metrics.csv", index=False)
    expected.to_parquet(PRED_DIR / "overhead_expected.parquet")
    joblib.dump(bundle, MODELS_DIR / "overhead_model.joblib")

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nFeature importance:\n", imp)
    print("\nExpected-overhead model vs 7-day reference:\n", metrics.to_string(index=False))
    print(f"\nHours using 7-day fallback (weather missing): "
          f"{expected['fallback'].sum():,} ({100 * expected['fallback'].mean():.1f}%)")


if __name__ == "__main__":
    main()