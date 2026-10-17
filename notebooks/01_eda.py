# %% [markdown]
# # 01 - EDA: NLR ESIF HPC data centre (hourly)

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data" / "processed" / "esif_hourly.parquet")
plt.rcParams.update({"figure.figsize": (12, 4), "axes.grid": True, "grid.alpha": 0.3})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150)
    plt.show()


# %% 1. Ten-year power timeline (shows the regime shift)
daily = df[["it_power_kw", "overhead_kw", "facility_kw"]].resample("D").mean() / 1000
ax = daily.plot(lw=0.6)
ax.set_ylabel("MW")
ax.set_title("Daily mean power")
save("01_power_timeline")

# %% 2. PUE over time
pue_d = df["pue_calc"].resample("D").median()
ax = pue_d.plot(lw=0.5, alpha=0.5, label="daily median")
pue_d.rolling(30, min_periods=10).median().plot(ax=ax, lw=2, label="30-day rolling")
ax.set_ylim(1.0, 1.15)
ax.set_ylabel("PUE")
ax.set_title("PUE over time")
ax.legend()
save("02_pue_timeline")

# %% 3. Overhead vs outdoor temperature, coloured by year
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df["outdoor_temp_c"], df["overhead_kw"], c=df.index.year, s=1, alpha=0.3)
plt.colorbar(sc, label="year")
ax.set_xlabel("Outdoor temperature (deg C)")
ax.set_ylabel("Overhead power (kW)")
ax.set_title("Infrastructure overhead vs outdoor temperature")
save("03_overhead_vs_temp")

# %% 4. PUE by temperature band (2019+)
d = df.loc["2019":]
bands = pd.cut(d["outdoor_temp_c"], np.arange(-25, 45, 5))
d.groupby(bands, observed=True)["pue_calc"].median().plot(kind="bar")
plt.ylim(1.0, 1.1)
plt.ylabel("median PUE")
plt.title("PUE by outdoor temperature band (2019+)")
save("04_pue_by_temp_band")

# %% 5. Daily and weekly load profiles (2024+)
r = df.loc["2024":]
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
r.groupby(r.index.hour)["it_power_kw"].mean().plot(ax=axes[0, 0], title="IT kW by hour of day")
r.groupby(r.index.hour)["overhead_kw"].mean().plot(ax=axes[0, 1], title="Overhead kW by hour of day")
r.groupby(r.index.dayofweek)["it_power_kw"].mean().plot(ax=axes[1, 0], title="IT kW by weekday (0=Mon)")
r.groupby(r.index.month)["overhead_kw"].mean().plot(ax=axes[1, 1], title="Overhead kW by month")
save("05_load_profiles")

# %% 6. Overhead component breakdown (monthly)
comp = df[["cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]].resample("MS").mean()
ax = comp.plot.area()
ax.set_ylabel("kW")
ax.set_title("Overhead components, monthly mean")
save("06_overhead_components")

# %% 7. Autocorrelation of facility power (justifies lag features)
s = df.loc["2024":, "facility_kw"].interpolate()
lags = np.arange(1, 24 * 8 + 1)
acf = [s.autocorr(int(l)) for l in lags]
plt.plot(lags, acf)
for x in (24, 168):
    plt.axvline(x, ls="--", c="grey")
plt.xlabel("lag (hours)")
plt.ylabel("autocorrelation")
plt.title("Facility power autocorrelation (2024+)")
save("07_autocorrelation")

# %% 8. Missing data by year and month
gap = df["is_gap"].groupby([df.index.year, df.index.month]).mean().unstack() * 100
plt.figure(figsize=(10, 5))
plt.imshow(gap, aspect="auto", cmap="Reds", vmin=0, vmax=100)
plt.colorbar(label="% hours missing")
plt.yticks(range(len(gap.index)), gap.index)
plt.xticks(range(12), range(1, 13))
plt.xlabel("month")
plt.title("Data gaps")
save("08_gaps_heatmap")

# %% 9. Key numbers for the report
cols = ["it_power_kw", "overhead_kw", "facility_kw", "pue_calc", "outdoor_temp_c", "cooling_share"]
print("Correlation (2019+):")
print(d[cols].corr().round(2))
print("\nIT kW quantiles by era:")
print(df["it_power_kw"].groupby(df.index.year >= 2023).describe().round(1))