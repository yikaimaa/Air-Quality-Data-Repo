
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


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

    print(f"Initial shape: {df.shape}")
    initial_rows = df.shape[0]
    initial_cols = df.shape[1]

    # --------------------------------------------------
    # 1️⃣ Drop severely incomplete region
    # --------------------------------------------------
    print("\n[STEP 1] Dropping Renfrew County (severely incomplete region)")
    before_region = df.shape[0]
    df = df[df["region"] != "Renfrew County"]
    after_region = df.shape[0]
    print(f"Rows removed: {before_region - after_region}")

    # --------------------------------------------------
    # 2️⃣ Drop non-modeling metadata columns
    # --------------------------------------------------
    print("\n[STEP 2] Dropping metadata columns (n_stations_total, n_stations_used)")

    drop_cols = ["n_stations_total", "n_stations_used"]
    existing_drop = [c for c in drop_cols if c in df.columns]

    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"Columns removed: {existing_drop}")
    else:
        print("No metadata columns found.")

    # --------------------------------------------------
    # 3️⃣ Drop all-zero numeric columns
    # --------------------------------------------------
    print("\n[STEP 3] Dropping all-zero numeric columns")

    numeric_df = df.select_dtypes(include="number")
    zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()

    if zero_cols:
        df = df.drop(columns=zero_cols)
        print(f"Zero columns removed: {zero_cols}")
    else:
        print("No all-zero columns found.")

    # --------------------------------------------------
    # 4️⃣ Handle structural missing (snow-related)
    # --------------------------------------------------
    print("\n[STEP 4] Filling structural snow-related missing values with 0")

    snow_cols = ["total_snow_cm", "snow_on_grnd_cm"]
    for col in snow_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"Filled missing in {col} with 0")

    # --------------------------------------------------
    # 5️⃣ Forward fill remaining numeric columns within region
    # --------------------------------------------------
    print("\n[STEP 5] Forward filling numeric columns within each region (safe, no leakage)")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    df[numeric_cols] = (
        df.groupby("region")[numeric_cols]
        .ffill()
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    final_rows = df.shape[0]
    final_cols = df.shape[1]

    print("\n==============================")
    print("DATA PREPROCESSING SUMMARY")
    print("==============================")
    print(f"Initial rows: {initial_rows}")
    print(f"Final rows:   {final_rows}")
    print(f"Initial cols: {initial_cols}")
    print(f"Final cols:   {final_cols}")
    print(f"Remaining missing values: {df.isna().sum().sum()}")
    print("==============================\n")

    # --------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)

    print("Clean dataset written to:", args.out_path)
    print("\nDATA PREPROCESSING COMPLETED\n")


if __name__ == "__main__":
    main()
