import pandas as pd

from greendc.pue.metrics import period_pue


def test_pue_is_energy_weighted_not_mean_of_ratios():
    idx = pd.date_range("2025-01-01", periods=2, freq="h", tz="America/Denver")
    df = pd.DataFrame({"it_power_kw": [100.0, 300.0], "facility_kw": [150.0, 330.0]}, index=idx)
    out = period_pue(df, "D")
    # energy-weighted: 480 / 400 = 1.20 ; naive mean of ratios would be (1.5 + 1.1) / 2 = 1.30
    assert abs(out["pue"].iloc[0] - 1.20) < 1e-9
    assert out["coverage_pct"].iloc[0] == 100