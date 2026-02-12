#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    print("\n==============================")
    print("DATA PREPROCESSING STARTED")
    print("==============================\n")

    df = pd.read_csv(args.in_path)
    df.columns = df.columns.str.lower()

    initial_rows, initial_cols = df.shape
    print(f"Initial shape: {df.shape}")

    # --------------------------------------------------
    # 1️⃣ Drop severely incomplete region
    # --------------------------------------------------
    if "region" in df.columns:
        before = len(df)
        df = df[df["region"] != "Renfrew County"]
        print(f"[DROP] Region 'Renfrew County' removed ({before - len(df)} rows dropped)")

    # --------------------------------------------------
    # 2️⃣ Drop metadata columns
    # --------------------------------------------------
    metadata_cols = ["n_stations_total", "n_stations_used"]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    for col in metadata_cols:
        print(f"[DROP] {col} -> metadata column")

    df = df.drop(columns=metadata_cols, errors="ignore")

    # --------------------------------------------------
    # 3️⃣ Drop flag / quality / outlier columns
    # --------------------------------------------------
    flag_cols = [
        c for c in df.columns
        if "_is_" in c or "data_quality" in c
    ]

    for col in flag_cols:
        print(f"[DROP] {col} -> flag/quality/outlier column")

    df = df.drop(columns=flag_cols, errors="ignore")

    # --------------------------------------------------
    # 4️⃣ Drop all-zero numeric columns
    # --------------------------------------------------
    numeric_df = df.select_dtypes(include="number")
    zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()

    for col in zero_cols:
        print(f"[DROP] {col} -> all-zero column")

    df = df.drop(columns=zero_cols, errors="ignore")

    # --------------------------------------------------
    # 5️⃣ Drop near-constant columns
    # --------------------------------------------------
    numeric_df = df.select_dtypes(include="number")
    var_series = numeric_df.var()
    near_constant_cols = var_series[var_series < 1e-8].index.tolist()

    for col in near_constant_cols:
        print(f"[DROP] {col} -> near-constant variance")

    df = df.drop(columns=near_constant_cols, errors="ignore")

    # --------------------------------------------------
    # 6️⃣ Missing Value Handling
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])

    # ---- Snow (structural missing) ----
    snow_cols = ["total_snow_cm", "snow_on_grnd_cm"]
    for col in snow_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"[IMPUTE] {col} -> fill 0 (structural snow missing)")

    # ---- Rain ----
    rain_cols = ["total_rain_mm", "total_precip_mm"]
    for col in rain_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"[IMPUTE] {col} -> fill 0 (rain assumed 0 when missing)")

    # ---- Temperature ----
    temp_cols = [
        "max_temp_degc", "min_temp_degc", "mean_temp_degc",
        "heat_deg_days_degc", "cool_deg_days_degc"
    ]
    for col in temp_cols:
        if col in df.columns:
            df[col] = (
                df.groupby("region")[col]
                  .apply(lambda x: x.interpolate(
                      method="linear",
                      limit=2,
                      limit_direction="forward"
                  ))
                  .reset_index(level=0, drop=True)
            )
            print(f"[IMPUTE] {col} -> forward linear interpolate (limit=2)")

    # ---- Wind Speed ----
    wind_speed_cols = ["spd_of_max_gust_km_h"]
    for col in wind_speed_cols:
        if col in df.columns:
            df[col] = (
                df.groupby("region")[col]
                  .apply(lambda x: x.interpolate(
                      method="linear",
                      limit=2,
                      limit_direction="forward"
                  ))
                  .reset_index(level=0, drop=True)
            )
            print(f"[IMPUTE] {col} -> forward linear interpolate (limit=2)")

    # ---- Wind Direction ----
    wind_dir_cols = ["dir_of_max_gust_deg", "dir_of_max_gust_10s_deg"]
    for col in wind_dir_cols:
        if col in df.columns:
            df[col] = (
                df.groupby("region")[col]
                  .ffill(limit=1)
            )
            print(f"[IMPUTE] {col} -> ffill(limit=1) (wind direction circular safe)")

    # pm25 untouched
    print("[INFO] pm25_region_daily_avg -> no imputation applied")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    final_rows, final_cols = df.shape

    print("\n==============================")
    print("DATA PREPROCESSING SUMMARY")
    print("==============================")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows:   {final_rows}")
    print(f"Initial cols: {initial_cols}")
    print(f"Final cols:   {final_cols}")
    print(f"Remaining missing values: {df.isna().sum().sum()}")
    print("==============================\n")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)

    print("Clean dataset written to:", args.out_path)
    print("\nDATA PREPROCESSING COMPLETED\n")


if __name__ == "__main__":
    main()
