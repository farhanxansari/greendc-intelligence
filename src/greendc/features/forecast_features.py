"""Feature engineering for facility power forecasting.

Row at time T predicts facility_kw[T] using information available at
T - horizon (plus outdoor temperature at T, treated as a weather forecast).

The model target is the RATIO change vs persistence: y[T] / y[T-h] - 1.
All history features are relative too, so the model is scale-free and
robust to the IT regime shifts in the ESIF data (1.2 MW -> 3.7 MW -> 2.6 MW).
"""
import numpy as np
import pandas as pd

from greendc.config import HORIZON

TARGET = "facility_kw"
META_COLS = ["target", "base", "seasonal_168", "target_ratio"]


def make_forecast_frame(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    h = horizon
    y = df[TARGET]
    base = y.shift(h)                       # persistence forecast
    out = pd.DataFrame(index=df.index)

    # ---- meta (not features)
    out["target"] = y
    out["base"] = base
    out["seasonal_168"] = y.shift(168)      # seasonal-naive baseline
    out["target_ratio"] = y / base - 1

    # ---- relative history (all info available at T - h)
    for lag in sorted({h + 1, h + 2, h + 3, 2 * h, 168, 168 + h}):
        out[f"rel_lag{lag}"] = y.shift(lag) / base - 1
    roll24 = base.rolling(24, min_periods=18)
    out["rel_roll24_mean"] = roll24.mean() / base - 1
    out["rel_roll24_min"] = roll24.min() / base - 1          # recent outage signal
    out["cv_roll24"] = roll24.std() / roll24.mean()           # recent volatility
    out["rel_roll168_mean"] = base.rolling(168, min_periods=120).mean() / base - 1

    # ---- composition at forecast time
    out["pue_lag"] = df["pue_calc"].shift(h)
    out["overhead_share_lag"] = (df["overhead_kw"] / df["facility_kw"]).shift(h)

    # ---- weather (temp at T = weather-forecast assumption)
    temp = df["outdoor_temp_c"]
    out["temp_c"] = temp
    out["temp_change"] = temp - temp.shift(h)
    out["rh_pct_lag"] = df["outdoor_rh_pct"].shift(h)

    # ---- calendar: hour + weekday only.
    # Day-of-year was removed: it let tree models memorise regime levels.
    idx = df.index
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLS]