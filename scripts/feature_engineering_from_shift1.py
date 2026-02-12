#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature engineering based on shift1 output (strict no-leakage)."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region-col", default="region")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--pm-col", default="pm25_region_daily_avg")
    parser.add_argument("--label-col", default="pm25_label")
    return parser.parse_args()


def add_group_lags(df, group_col, col, lags):
    g = df.groupby(group_col)[col]
    for k in lags:
        df[f"{col}_lag_{k}"] = g.shift(k)
    return df


def add_group_roll(df, group_col, col, window):
    g = df.groupby(group_col)[col]
    rolled = g.rolling(window=window, min_periods=window)
    df[f"{col}_roll_mean_{window}"] = rolled.mean().reset_index(level=0, drop=True)
    df[f"{col}_roll_std_{window}"] = rolled.std().reset_index(level=0, drop=True)
    return df


def main():

    args = parse_args()

    print(f"[INFO] Reading: {args.input}")
    df = pd.read_csv(args.input)

    required = {args.region_col, args.date_col, args.pm_col, args.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values([args.region_col, args.date_col]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 1️⃣ PM history features
    # ------------------------------------------------------------
    print("[INFO] Creating PM lag features...")
    df = add_group_lags(df, args.region_col, args.pm_col, [1, 2, 3, 7, 14])

    print("[INFO] Creating PM rolling features...")
    for w in [3, 7, 14]:
        df = add_group_roll(df, args.region_col, args.pm_col, w)

    df[f"{args.pm_col}_diff_1"] = df[args.pm_col] - df[f"{args.pm_col}_lag_1"]
    df[f"{args.pm_col}_diff_7"] = df[args.pm_col] - df[f"{args.pm_col}_lag_7"]

    # ------------------------------------------------------------
    # 2️⃣ Weather history features
    # ------------------------------------------------------------
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

    WEATHER_COLS = [c for c in WEATHER_COLS if c in df.columns]

    print("[INFO] Creating weather lag features...")
    for c in WEATHER_COLS:
        df = add_group_lags(df, args.region_col, c, [1, 7])

    print("[INFO] Creating weather rolling features...")
    KEY_WEATHER = [
        "mean_temp_degc",
        "total_precip_mm",
        "total_rain_mm",
        "total_snow_cm",
        "snow_on_grnd_cm",
        "spd_of_max_gust_km_h",
    ]
    KEY_WEATHER = [c for c in KEY_WEATHER if c in df.columns]

    for c in KEY_WEATHER:
        for w in [3, 7]:
            df = add_group_roll(df, args.region_col, c, w)

    # ------------------------------------------------------------
    # 3️⃣ Calendar encoding
    # ------------------------------------------------------------
    df["day_of_year"] = df[args.date_col].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    # ------------------------------------------------------------
    # 4️⃣ Drop NA safely (STRICT)
    # ------------------------------------------------------------
    before = len(df)

    feature_cols = [
        c for c in df.columns
        if c not in [args.region_col, args.date_col]
    ]

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    after = len(df)

    # ------------------------------------------------------------
    # 5️⃣ Report completeness
    # ------------------------------------------------------------
    n_regions = df[args.region_col].nunique()
    expected_days = 1826
    expected_total = n_regions * expected_days

    completeness = after / expected_total * 100

    print("\n==============================")
    print("FEATURE ENGINEERING SUMMARY")
    print("==============================")
    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna:  {after}")
    print(f"Rows dropped:       {before - after}")
    print(f"Data completeness:  {completeness:.2f}%")
    print("==============================\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"[SUCCESS] Saved to: {args.output}")


if __name__ == "__main__":
    main()
