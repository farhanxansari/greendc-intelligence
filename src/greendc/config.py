"""Project-wide constants: paths, splits, horizon."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOURLY_FILE = ROOT / "data" / "processed" / "esif_hourly.parquet"
MODELS_DIR = ROOT / "models_store"
RESULTS_DIR = ROOT / "docs" / "results"
PRED_DIR = ROOT / "data" / "processed"

TZ = "America/Denver"
MODEL_START = "2019-01-01"   # earlier years have heavy gaps
TRAIN_END = "2024-07-01"     # exclusive: train = 2019-01 .. 2024-06
VAL_END = "2025-01-01"       # exclusive: val = 2024-07 .. 2024-12, test = 2025+
HORIZON = 24                 # day-ahead forecast (hours)
SEED = 42


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)