"""Download, clean and resample the NLR ESIF HPC data-centre PUE dataset.

Source: Clark & Strelka (2025), "NLR HPC Facility Power Usage Effectiveness
(PUE) Data", NLR Data Catalog, DOI: 10.7799/3015212.

Run from the repo root:
    python -m greendc.data.esif --inspect  # print raw schema
    python -m greendc.data.esif            # build data/processed/esif_hourly.parquet
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/esif")
PROCESSED_DIR = Path("data/processed")
OUTPUT = PROCESSED_DIR / "esif_hourly.parquet"

URLS = {
    "power": "https://data.nlr.gov/system/files/300/1757103411-esif.influx.buildingData.PUE.combined.parquet",
    "weather": "https://data.nlr.gov/system/files/300/1757105566-esif.influx.buildingData.outside.combined_2.parquet",
}

POWER_COLS = ["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]
WEATHER_COLS = ["outdoor_air_temp", "outdoor_air_humidity"]
FILTER_PUMP_KW = 2.67        # per dataset docs: constant, not included in pump_kw
DATA_START = "2015-11-01"    # drops bogus 1970 timestamps in the weather file
LOCAL_TZ = "America/Denver"  # facility is in Golden, Colorado
MIN_VALID_MINUTES = 30       # an hour needs >= 30 good 1-min readings to count
MAX_GAP_HOURS = 3            # interpolate gaps up to this long; longer stay NaN
SPIKE_K = 8.0                # glitch threshold (multiples of local spread)
SPIKE_WINDOW = 61            # minutes, centred rolling window
MIN_IT_KW = 100.0            # IT below this is a sensor dropout, not real load


# ---------------------------------------------------------------- download
def _is_parquet(path: Path) -> bool:
    """Real parquet files start and end with the bytes b'PAR1'."""
    if not path.exists() or path.stat().st_size < 8:
        return False
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(-4, 2)
        tail = f.read(4)
    return head == b"PAR1" and tail == b"PAR1"


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = RAW_DIR / f"{name}.parquet"
        if _is_parquet(dest) and not force:
            print(f"[skip] {dest} already exists")
            continue
        print(f"[download] {name} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        if not _is_parquet(dest):
            raise RuntimeError(f"{dest} is not a valid parquet file - download it manually.")


# ---------------------------------------------------------------- loading
def _load_raw(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    cols = None if columns is None else ["ts", *columns]
    df = pd.read_parquet(RAW_DIR / f"{name}.parquet", columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop(columns=["day", "tags"], errors="ignore")
    df = df[df["ts"] >= pd.Timestamp(DATA_START, tz="UTC")]
    return df.set_index("ts").sort_index()


def inspect() -> None:
    for name in URLS:
        df = _load_raw(name)
        print(f"\n===== {name} =====")
        print(f"rows: {len(df):,}   range: {df.index.min()} -> {df.index.max()}")
        print("median sampling interval:", df.index.to_series().diff().median())
        if "unit" in df:
            print("unit values:", df["unit"].value_counts().to_dict())
        print(df.describe().T.round(2))


# ---------------------------------------------------------------- cleaning
def _remove_glitches(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Set physically implausible readings to NaN.

    - negative power
    - isolated spikes far from the centred rolling median
    These are sensor glitches, NOT operational anomalies (those are detected later).
    """
    out = df.copy()
    for c in cols:
        s = out[c].mask(out[c] < 0)
        med = s.rolling(SPIKE_WINDOW, center=True, min_periods=10).median()
        mad = (s - med).abs().rolling(SPIKE_WINDOW, center=True, min_periods=10).median()
        tolerance = SPIKE_K * mad + 0.05 * med.abs() + 1.0
        spike = (s - med).abs() > tolerance
        removed = int(spike.sum() + (out[c] < 0).sum())
        out[c] = s.mask(spike)
        print(f"  {c:<20} removed {removed:>8,} glitch readings")
    return out


def _to_hourly(minute: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    hourly = minute[cols].resample("1h").mean()
    counts = minute[cols].resample("1h").count()
    return hourly.where(counts >= MIN_VALID_MINUTES)


def build_hourly() -> pd.DataFrame:
    print("Loading power data ...")
    power = _load_raw("power", POWER_COLS + ["pue"])
    power = power[~power.index.duplicated()].resample("1min").mean()
    print("Removing glitches ...")
    power = _remove_glitches(power, POWER_COLS)
    power["it_power_kw"] = power["it_power_kw"].mask(power["it_power_kw"] < MIN_IT_KW)

    print("Loading weather data ...")
    weather = _load_raw("weather", WEATHER_COLS)
    bad_temp = weather["outdoor_air_temp"] == 0.0  # logger default, not real weather
    weather.loc[bad_temp, WEATHER_COLS] = float("nan")
    weather = weather[~weather.index.duplicated()].resample("1min").mean()

    df = _to_hourly(power, POWER_COLS + ["pue"]).join(
        _to_hourly(weather, WEATHER_COLS), how="left"
    )

    # Units / naming
    df["outdoor_temp_c"] = (df.pop("outdoor_air_temp") - 32) * 5 / 9
    df = df.rename(columns={"outdoor_air_humidity": "outdoor_rh_pct", "pue": "pue_reported"})

    # Fill only short gaps; long outages stay missing and are flagged
    df = df.interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    df["is_gap"] = df[POWER_COLS].isna().any(axis=1)

    # Derived efficiency metrics (recomputed - reported PUE contains glitches)
    df["overhead_kw"] = df[POWER_COLS[1:]].sum(axis=1, min_count=4) + FILTER_PUMP_KW
    df["facility_kw"] = df["it_power_kw"] + df["overhead_kw"]
    df["pue_calc"] = df["facility_kw"] / df["it_power_kw"]
    df["cooling_share"] = (
        df["cooling_kw"] + df["hvac_kw"] + df["pump_kw"] + FILTER_PUMP_KW
    ) / df["facility_kw"]

    df.index = df.index.tz_convert(LOCAL_TZ)
    df.index.name = "ts"
    return df


def summarise(df: pd.DataFrame) -> None:
    print(f"\nHourly rows: {len(df):,}  ({df.index.min()} -> {df.index.max()})")
    print(f"Gap hours: {df['is_gap'].sum():,} ({df['is_gap'].mean():.1%})")
    diff = (df["pue_calc"] - df["pue_reported"]).abs()
    print(f"PUE calc vs reported - median abs diff: {diff.median():.4f}")
    print(df[["it_power_kw", "overhead_kw", "pue_calc", "outdoor_temp_c"]].describe().T.round(3))

    yearly = df.groupby(df.index.year).agg(
        it_mw_mean=("it_power_kw", lambda s: s.mean() / 1000),
        pue_median=("pue_calc", "median"),
        temp_c_mean=("outdoor_temp_c", "mean"),
        gap_pct=("is_gap", "mean"),
    )
    yearly["gap_pct"] *= 100
    print("\nPer-year summary:")
    print(yearly.round(3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    download(force=args.force_download)
    if args.inspect:
        inspect()
        return
    df = build_hourly()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)
    summarise(df)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()