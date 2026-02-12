"""
Build next-day PM2.5 prediction table (region-day level).

Input:
../Datasets/Ontario/processed_datasets/pm25_weather_daily_region_joined.csv

Output:
../Datasets/Ontario/processed_datasets/prediction_table.csv

Each row corresponds to (region, date=t)
- Features use information available at or before t (including weather at t)
- Target is next-day PM2.5: pm25_region_daily_avg at t+1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IN_PATH = "../Datasets/Ontario/processed_datasets/pm25_weather_daily_region_joined.csv"
OUT_PATH = "../Datasets/Ontario/processed_datasets/prediction_table.csv"


PM_COL = "pm25_region_daily_avg"
DATE_COL = "date"
REGION_COL = "region"

# Weather numeric columns available in the joined table
WEATHER_COLS = [
    "max_temp_degc",
    "min_temp_degc",
    "mean_temp_degc",
    "heat_deg_days_degc",
    "cool_deg_days_degc",
    "total_rain_mm",
    "total_snow_cm",
    "total_precip_mm",
    "snow_on_grnd_cm",
    "spd_of_max_gust_km_h",
    "dir_of_max_gust_deg",
]

# Optional context columns (can help model understand reliability)
CONTEXT_COLS = [
    "n_stations_total",
    "n_stations_used",
    "n_weather_stations",
]


def assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


def add_group_lags(df: pd.DataFrame, group_col: str, col: str, lags: list[int]) -> pd.DataFrame:
    g = df.groupby(group_col)[col]
    for k in lags:
        df[f"{col}_lag_{k}"] = g.shift(k)
    return df


def add_group_roll(df: pd.DataFrame, group_col: str, col: str, window: int) -> pd.DataFrame:
    """
    Rolling stats ending at t (inclusive), computed within each group.
    min_periods=window ensures consistent window length (avoids partial windows).
    """
    g = df.groupby(group_col)[col]
    rolled = g.rolling(window=window, min_periods=window)

    df[f"{col}_roll_mean_{window}"] = rolled.mean().reset_index(level=0, drop=True)
    df[f"{col}_roll_std_{window}"] = rolled.std().reset_index(level=0, drop=True)
    df[f"{col}_roll_max_{window}"] = rolled.max().reset_index(level=0, drop=True)
    df[f"{col}_roll_min_{window}"] = rolled.min().reset_index(level=0, drop=True)
    return df


def main() -> None:
    df = pd.read_csv(IN_PATH)

    # --- validate schema ---
    assert_columns(df, [DATE_COL, REGION_COL, PM_COL] + WEATHER_COLS)

    # --- parse date + sort ---
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        bad = df[df[DATE_COL].isna()].head(5)
        raise ValueError(f"Found unparsable dates. Example rows:\n{bad}")

    df = df.sort_values([REGION_COL, DATE_COL]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 1) Target: next-day PM2.5
    # ------------------------------------------------------------
    df["target_pm25_next_day"] = df.groupby(REGION_COL)[PM_COL].shift(-1)

    # ------------------------------------------------------------
    # 2) PM2.5 history features
    # ------------------------------------------------------------
    # Lags (t-1, t-2, t-3, t-7, t-14)
    df = add_group_lags(df, REGION_COL, PM_COL, lags=[1, 2, 3, 7, 14])

    # Rolling windows ending at t (3, 7, 14)
    for w in [3, 7, 14]:
        df = add_group_roll(df, REGION_COL, PM_COL, window=w)

    # Differences / momentum
    df[f"{PM_COL}_diff_1"] = df[PM_COL] - df[f"{PM_COL}_lag_1"]
    df[f"{PM_COL}_diff_7"] = df[PM_COL] - df[f"{PM_COL}_lag_7"]

    # ------------------------------------------------------------
    # 3) Weather history features (allowed to use weather at t)
    # ------------------------------------------------------------
    # Lags for weather too (yesterday / last week)
    for c in WEATHER_COLS:
        df = add_group_lags(df, REGION_COL, c, lags=[1, 7])

    # Rolling windows for key weather drivers (3, 7)
    # You can expand to all WEATHER_COLS, but this keeps table size manageable.
    KEY_WEATHER = [
        "mean_temp_degc",
        "total_precip_mm",
        "total_rain_mm",
        "total_snow_cm",
        "snow_on_grnd_cm",
        "spd_of_max_gust_km_h",
    ]
    for c in KEY_WEATHER:
        for w in [3, 7]:
            df = add_group_roll(df, REGION_COL, c, window=w)

    # Event-style indicators (good for trees, interpretable)
    df["is_precip_today"] = (df["total_precip_mm"].fillna(0) > 0).astype(int)
    df["is_rain_today"] = (df["total_rain_mm"].fillna(0) > 0).astype(int)
    df["is_snow_today"] = (df["total_snow_cm"].fillna(0) > 0).astype(int)
    df["is_freezing_today"] = (df["mean_temp_degc"] < 0).astype(int)

    # ------------------------------------------------------------
    # 4) Calendar / seasonality
    # ------------------------------------------------------------
    df["month"] = df[DATE_COL].dt.month
    df["day_of_year"] = df[DATE_COL].dt.dayofyear

    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    # ------------------------------------------------------------
    # 5) Optional reliability/context features
    # ------------------------------------------------------------
    for c in CONTEXT_COLS:
        if c in df.columns:
            # fraction of PM stations used in region-day
            if c == "n_stations_used" and "n_stations_total" in df.columns:
                df["pm_station_use_frac"] = df["n_stations_used"] / df["n_stations_total"].replace(0, np.nan)

    # ------------------------------------------------------------
    # 6) Final selection + drop NA rows introduced by shifting/rolling
    # ------------------------------------------------------------
    # Keep original base columns too (handy for debugging)
    keep_cols = (
        [DATE_COL, REGION_COL, PM_COL, "target_pm25_next_day"]
        + WEATHER_COLS
        + [c for c in CONTEXT_COLS if c in df.columns]
        + [c for c in df.columns if c.endswith(("_lag_1", "_lag_2", "_lag_3", "_lag_7", "_lag_14"))]
        + [c for c in df.columns if "_roll_" in c]
        + [f"{PM_COL}_diff_1", f"{PM_COL}_diff_7"]
        + ["is_precip_today", "is_rain_today", "is_snow_today", "is_freezing_today",
           "month", "day_of_year", "sin_doy", "cos_doy"]
    )

    # Deduplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c in df.columns and not (c in seen or seen.add(c))]

    out = df[keep_cols].copy()

    # Drop rows without target (last day of each region) and incomplete lag/rolling windows
    out = out.dropna(subset=["target_pm25_next_day"]).copy()
    out = out.dropna().reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)

    print(f"Saved prediction table: {OUT_PATH}")
    print(f"Rows: {len(out):,} | Cols: {out.shape[1]:,}")
    print("\nColumns written:")
    for i, c in enumerate(out.columns, 1):
        print(f"{i:3d}. {c}")


if __name__ == "__main__":
    main()
