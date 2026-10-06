"""Train and evaluate facility power forecasting models.

Protocol:
  - Baselines: persistence (same hour, h hours earlier), seasonal naive (same hour last week)
  - Models predict the ratio change vs persistence; kW = base * (1 + ratio)
  - Fit on TRAIN, early-stop / select on VAL; refit on TRAIN+VAL, score once on TEST
Run:
  python -m greendc.models.forecast              # day-ahead (24 h)
  python -m greendc.models.forecast --horizon 1  # hour-ahead
"""
import argparse
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from greendc.config import (HORIZON, HOURLY_FILE, MODEL_START, MODELS_DIR, PRED_DIR,
                            RESULTS_DIR, SEED, TRAIN_END, VAL_END, ts)
from greendc.features.forecast_features import feature_columns, make_forecast_frame

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
RATIO_CLIP = (-0.9, 2.0)  # training target clip: limits outage/recovery extremes


def load_dataset(horizon: int) -> pd.DataFrame:
    frame = make_forecast_frame(pd.read_parquet(HOURLY_FILE), horizon)
    frame = frame[frame.index >= ts(MODEL_START)]
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def split(data: pd.DataFrame):
    tr = data[data.index < ts(TRAIN_END)]
    va = data[(data.index >= ts(TRAIN_END)) & (data.index < ts(VAL_END))]
    te = data[data.index >= ts(VAL_END)]
    return tr, va, te


def metrics(name: str, part: str, y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "model": name, "split": part,
        "MAE_kW": mean_absolute_error(y, p),
        "RMSE_kW": np.sqrt(mean_squared_error(y, p)),
        "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100),
        "R2": r2_score(y, p),
    }


def build_models(n_xgb: int | None = None, n_lgb: int | None = None) -> dict:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=20, max_features=0.5,
            n_jobs=-1, random_state=SEED),
        "xgboost": XGBRegressor(
            n_estimators=n_xgb or 3000, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, objective="reg:absoluteerror",
            n_jobs=-1, random_state=SEED,
            early_stopping_rounds=None if n_xgb else 100),
        "lightgbm": LGBMRegressor(
            n_estimators=n_lgb or 3000, learning_rate=0.03, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, objective="l1", random_state=SEED, verbose=-1),
    }


def fit(name, model, X, y, X_val=None, y_val=None):
    if name == "xgboost" and X_val is not None:
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "lightgbm" and X_val is not None:
        model.fit(X, y, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X, y)
    return model


def to_kw(model, part: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return part["base"].to_numpy() * (1 + model.predict(part[feats]))


def main(horizon: int) -> None:
    data = load_dataset(horizon)
    tr, va, te = split(data)
    feats = feature_columns(data)
    print(f"horizon={horizon}h  rows train={len(tr):,} val={len(va):,} test={len(te):,}  features={len(feats)}")

    y_tr = tr["target_ratio"].clip(*RATIO_CLIP)
    y_va = va["target_ratio"].clip(*RATIO_CLIP)
    persist = f"persistence_{horizon}h"

    results = []
    test_preds = pd.DataFrame({"actual": te["target"]}, index=te.index)

    # ---- baselines
    for name, col in [(persist, "base"), ("seasonal_naive_168h", "seasonal_168")]:
        results.append(metrics(name, "val", va["target"], va[col]))
        results.append(metrics(name, "test", te["target"], te[col]))
        test_preds[name] = te[col]

    # ---- stage 1: fit on train, score on val
    models = build_models()
    for name, model in models.items():
        t0 = time.time()
        fit(name, model, tr[feats], y_tr, va[feats], y_va)
        results.append(metrics(name, "val", va["target"], to_kw(model, va, feats)))
        print(f"[val ] {name:<18} fitted in {time.time() - t0:5.1f}s")

    n_xgb = models["xgboost"].best_iteration + 1
    n_lgb = models["lightgbm"].best_iteration_ or 3000

    # ---- stage 2: refit on train+val, score once on test
    trva = pd.concat([tr, va])
    final = build_models(n_xgb=n_xgb, n_lgb=n_lgb)
    for name, model in final.items():
        t0 = time.time()
        fit(name, model, trva[feats], trva["target_ratio"].clip(*RATIO_CLIP))
        p = to_kw(model, te, feats)
        test_preds[name] = p
        results.append(metrics(name, "test", te["target"], p))
        print(f"[test] {name:<18} refitted in {time.time() - t0:5.1f}s")

    # ---- results with skill vs persistence
    res = pd.DataFrame(results)
    base_mae = res[res.model == persist].set_index("split")["MAE_kW"]
    res["skill_vs_persistence_%"] = (1 - res["MAE_kW"] / res["split"].map(base_mae)) * 100
    res = res.sort_values(["split", "MAE_kW"]).round(4)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / f"forecast_metrics_h{horizon}.csv", index=False)
    test_preds.to_parquet(PRED_DIR / f"forecast_test_predictions_h{horizon}.parquet")

    # ---- best model chosen on VAL only
    trained = res[(res.split == "val") & res.model.isin(final)]
    best = trained.sort_values("MAE_kW").iloc[0]["model"]
    joblib.dump({"model": final[best], "features": feats, "horizon": horizon,
                 "target": "ratio_vs_persistence"},
                MODELS_DIR / f"forecast_best_h{horizon}.joblib")
    if hasattr(final[best], "feature_importances_"):
        imp = pd.Series(final[best].feature_importances_, index=feats).sort_values(ascending=False)
        imp.to_csv(RESULTS_DIR / f"forecast_feature_importance_h{horizon}.csv", header=["importance"])
        print("\nTop 10 features:\n", imp.head(10).round(4))

    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print("\n", res.to_string(index=False))
    print(f"\nBest model on validation: {best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON)
    main(parser.parse_args().horizon)