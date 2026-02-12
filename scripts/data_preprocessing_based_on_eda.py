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
    df = df[df["region"] != "Renfrew County"]

    # --------------------------------------------------
    # 2️⃣ Drop metadata columns
    # --------------------------------------------------
    metadata_cols = ["n_stations_total", "n_stations_used"]
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    if metadata_cols:
        print("Dropping metadata columns:", metadata_cols)
        df = df.drop(columns=metadata_cols)

    # --------------------------------------------------
    # 3️⃣ Drop flag / quality columns
    # --------------------------------------------------
    flag_cols = [c for c in df.columns if "_is_" in c or "data_quality" in c]

    if flag_cols:
        print("Dropping flag columns:", flag_cols)
        df = df.drop(columns=flag_cols)

    # --------------------------------------------------
    # 4️⃣ Drop all-zero numeric columns
    # --------------------------------------------------
    numeric_df = df.select_dtypes(include="number")
    zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()

    if zero_cols:
        print("Dropping all-zero columns:", zero_cols)
        df = df.drop(columns=zero_cols)

    # --------------------------------------------------
    # 5️⃣ Drop near-constant columns (variance threshold)
    # --------------------------------------------------
    numeric_df = df.select_dtypes(include="number")
    var_series = numeric_df.var()

    near_constant_cols = var_series[var_series < 1e-8].index.tolist()

    if near_constant_cols:
        print("Dropping near-constant columns:", near_constant_cols)
        df = df.drop(columns=near_constant_cols)

    # --------------------------------------------------
    # 6️⃣ Handle structural missing (snow-related)
    # --------------------------------------------------
    snow_cols = ["total_snow_cm", "snow_on_grnd_cm"]
    for col in snow_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --------------------------------------------------
    # 7️⃣ Forward fill continuous physical variables
    # --------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    ffill_cols = [c for c in numeric_cols if "pm25" not in c]

    df[ffill_cols] = df.groupby("region")[ffill_cols].ffill()

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    final_rows, final_cols = df.shape

    print("\n==============================")
    print("DATA PREPROCESSING V2 SUMMARY")
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
    print("\nDATA PREPROCESSING V2 COMPLETED\n")


if __name__ == "__main__":
    main()
