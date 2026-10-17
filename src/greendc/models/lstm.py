"""LSTM facility power forecaster (Objective 2), same protocol as models/forecast.py.

- Target: ratio change vs persistence, y[T] / y[T-h] - 1 (clipped like the tree models)
- History branch: last 72 h ending at T-h of scale-free signals
  (load relative to persistence, PUE, overhead share, temperature, hour of day)
- Target-time branch: temperature at T (weather-forecast assumption), temperature change,
  hour, weekday, weekend flag
- Fit on TRAIN with early stopping on VAL; refit on TRAIN+VAL for the same number of
  epochs; score once on TEST, on the hours shared with the tree models.
Results are merged into forecast_metrics_h{h}.csv and forecast_test_predictions_h{h}.parquet,
so the dashboard shows the LSTM automatically. Run AFTER models.forecast (which rewrites them).

Run: python -m greendc.models.lstm               (day-ahead, ~10-20 min on CPU)
     python -m greendc.models.lstm --horizon 1
"""
import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import pandas as pd
from keras import layers
from numpy.lib.stride_tricks import sliding_window_view

from greendc.config import (
    HORIZON,
    HOURLY_FILE,
    MODEL_START,
    MODELS_DIR,
    PRED_DIR,
    RESULTS_DIR,
    SEED,
    TRAIN_END,
    VAL_END,
    ts,
)
from greendc.models.forecast import RATIO_CLIP, metrics

LOOKBACK = 72
MAX_EPOCHS = 40
BATCH = 256


def build_arrays(df: pd.DataFrame, h: int):
    """Return (times, X_seq, X_static, target_kw, base_kw, target_ratio) for every usable hour."""
    idx = df.index
    y = df["facility_kw"].to_numpy(np.float32)
    temp = df["outdoor_temp_c"].to_numpy(np.float32)
    pue = df["pue_calc"].to_numpy(np.float32)
    share = (df["overhead_kw"] / df["facility_kw"]).to_numpy(np.float32)
    hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    hs = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hc = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    # channels per hour: [load, pue, overhead share, temp, hour_sin, hour_cos]
    chans = np.stack([y, (pue - 1) * 10, share * 10, temp, hs, hc], axis=1)
    windows = sliding_window_view(chans, LOOKBACK, axis=0).transpose(0, 2, 1)  # (n-L+1, L, C)

    i = np.arange(len(df))
    k = i - h - LOOKBACK + 1          # window covers rows k .. i-h (all known at forecast time)
    ok = k >= 0
    i, k = i[ok], k[ok]

    X_seq = windows[k].copy()
    base = y[i - h]
    X_seq[:, :, 0] = np.clip(X_seq[:, :, 0] / base[:, None] - 1, -1, 3)  # relative history

    X_static = np.stack([
        temp[i], temp[i] - temp[i - h], hs[i], hc[i],
        np.sin(2 * np.pi * dow[i] / 7), np.cos(2 * np.pi * dow[i] / 7), (dow[i] >= 5),
    ], axis=1).astype(np.float32)

    target = y[i]
    ratio = target / base - 1
    valid = (np.isfinite(X_seq).all(axis=(1, 2)) & np.isfinite(X_static).all(axis=1)
             & np.isfinite(ratio) & (base > 0))
    return idx[i][valid], X_seq[valid], X_static[valid], target[valid], base[valid], ratio[valid]


def build_model(n_seq: int, n_static: int) -> keras.Model:
    seq_in = keras.Input((LOOKBACK, n_seq), name="history")
    st_in = keras.Input((n_static,), name="target_time")
    x = layers.LSTM(64, return_sequences=True)(seq_in)
    x = layers.LSTM(32)(x)
    s = layers.Dense(16, activation="relu")(st_in)
    z = layers.concatenate([x, s])
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.1)(z)
    out = layers.Dense(1, name="ratio")(z)
    model = keras.Model([seq_in, st_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def main(h: int) -> None:
    keras.utils.set_random_seed(SEED)
    print(f"Keras backend: {keras.backend.backend()}")

    times, Xs, Xt, target, base, ratio = build_arrays(pd.read_parquet(HOURLY_FILE), h)
    keep = times >= ts(MODEL_START)
    times, Xs, Xt, target, base, ratio = (times[keep], Xs[keep], Xt[keep],
                                          target[keep], base[keep], ratio[keep])

    tr = times < ts(TRAIN_END)
    va = (times >= ts(TRAIN_END)) & (times < ts(VAL_END))
    te = times >= ts(VAL_END)
    print(f"horizon={h}h  samples train={tr.sum():,} val={va.sum():,} test={te.sum():,}  "
          f"seq={Xs.shape[1:]} static={Xt.shape[1]}")

    # normalise temperature with TRAIN statistics only
    mu, sd = float(Xs[tr][:, :, 3].mean()), float(Xs[tr][:, :, 3].std())
    Xs[:, :, 3] = (Xs[:, :, 3] - mu) / sd
    Xt[:, 0] = (Xt[:, 0] - mu) / sd
    Xt[:, 1] = Xt[:, 1] / sd
    y = np.clip(ratio, *RATIO_CLIP).astype(np.float32)
    y_scale = float(y[tr].std())   # unit-scale target: tiny 1-h ratios otherwise collapse to a constant
    y = y / y_scale

    # ---- stage 1: train, early-stop on val
    t0 = time.time()
    model = build_model(Xs.shape[2], Xt.shape[1])
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit([Xs[tr], Xt[tr]], y[tr], validation_data=([Xs[va], Xt[va]], y[va]),
                     epochs=MAX_EPOCHS, batch_size=BATCH, callbacks=[early], verbose=2)
    best_epochs = int(np.argmin(hist.history["val_loss"])) + 1
    p_val = base[va] * (1 + y_scale * model.predict([Xs[va], Xt[va]], batch_size=1024, verbose=0).ravel())
    print(f"[val ] best epoch {best_epochs}, fitted in {time.time() - t0:.0f}s")

    # ---- stage 2: refit on train+val for best_epochs, score on test
    t0 = time.time()
    keras.utils.set_random_seed(SEED)
    final = build_model(Xs.shape[2], Xt.shape[1])
    trva = tr | va
    final.fit([Xs[trva], Xt[trva]], y[trva], epochs=best_epochs, batch_size=BATCH, verbose=2)
    p_test = base[te] * (1 + y_scale * final.predict([Xs[te], Xt[te]], batch_size=1024, verbose=0).ravel())
    print(f"[test] refitted in {time.time() - t0:.0f}s")

    # ---- score on the hours shared with the tree models (fair comparison)
    persist = f"persistence_{h}h"
    preds = pd.read_parquet(PRED_DIR / f"forecast_test_predictions_h{h}.parquet")
    lstm_test = pd.Series(p_test, index=times[te])
    common = preds.index.intersection(lstm_test.index)

    rows = [
        metrics("lstm", "val", target[va], p_val),
        metrics(persist, "val", target[va], base[va]),
        metrics("lstm", "test", preds.loc[common, "actual"], lstm_test[common]),
        metrics(persist, "test", preds.loc[common, "actual"], preds.loc[common, persist]),
    ]
    cmp = pd.DataFrame(rows)
    base_mae = cmp[cmp.model == persist].set_index("split")["MAE_kW"]
    cmp["skill_vs_persistence_%"] = (1 - cmp["MAE_kW"] / cmp["split"].map(base_mae)) * 100
    lstm_rows = cmp[cmp.model == "lstm"].round(4)

    # ---- merge into the shared results so the API/dashboard pick it up
    mpath = RESULTS_DIR / f"forecast_metrics_h{h}.csv"
    res = pd.read_csv(mpath)
    res = pd.concat([res[res.model != "lstm"], lstm_rows], ignore_index=True)
    res.sort_values(["split", "MAE_kW"]).to_csv(mpath, index=False)
    preds["lstm"] = lstm_test.reindex(preds.index)
    preds.to_parquet(PRED_DIR / f"forecast_test_predictions_h{h}.parquet")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final.save(MODELS_DIR / f"lstm_h{h}.keras")

    pd.set_option("display.width", 140)
    print(f"\nLSTM vs persistence on identical hours (test rows compared: {len(common):,} "
          f"of {len(preds):,} tree-model test rows):")
    print(cmp.round(4).to_string(index=False))
    print("\nFull test table (tree models scored on all their test rows):")
    print(res[res.split == "test"].sort_values("MAE_kW").round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON)
    main(parser.parse_args().horizon)