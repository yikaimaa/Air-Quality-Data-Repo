#!/usr/bin/env python3
"""
shift1_prevent_data_leakage.py

Updated Version:
- Rename pm25 column to pm25_label
- Create next-day target from pm25_label
- Drop year/month/day columns
- Shift ALL remaining feature columns by +1
- Rename shifted columns with suffix _lag_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply shift(1) to all features to prevent data leakage."
    )

    parser.add_argument("--input", required=True, help="Path to cleaned dataset")
    parser.add_argument("--output", required=True, help="Path to save aligned dataset")
    parser.add_argument("--region-col", default="region")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--pm-col", default="pm25_region_daily_avg")

    return parser.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Reading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {args.region_col, args.date_col, args.pm_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure proper order
    df = df.sort_values([args.region_col, args.date_col]).reset_index(drop=True)

    # ============================================================
    # 1️⃣ Rename PM column to pm25_label
    # ============================================================
    df = df.rename(columns={args.pm_col: "pm25_label"})

    # ============================================================
    # 2️⃣ Create next-day label
    # ============================================================
    df["pm25_label_next_day"] = (
        df.groupby(args.region_col)["pm25_label"]
        .shift(-1)
    )

    # ============================================================
    # 3️⃣ Drop unwanted calendar columns
    # ============================================================
    drop_cols = [c for c in ["year", "month", "day"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ============================================================
    # 4️⃣ Identify feature columns to shift
    # Exclude:
    #   region, date, pm25_label, pm25_label_next_day
    # ============================================================
    exclude_cols = {
        args.region_col,
        args.date_col,
        "pm25_label",
        "pm25_label_next_day"
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"[INFO] Shifting {len(feature_cols)} feature columns by +1")

    # ============================================================
    # 5️⃣ Shift features
    # ============================================================
    shifted = (
        df.groupby(args.region_col)[feature_cols]
        .shift(1)
    )

    # Rename shifted columns
    shifted.columns = [f"{c}_lag_1" for c in feature_cols]

    # Drop original feature columns
    df = df.drop(columns=feature_cols)

    # Concatenate shifted features
    df = pd.concat([df, shifted], axis=1)

    # ============================================================
    # 6️⃣ Drop misaligned rows
    # ============================================================
    before_rows = len(df)

    df = df.dropna(subset=["pm25_label_next_day"])
    df = df.dropna()

    after_rows = len(df)

    print(f"[INFO] Rows before alignment: {before_rows}")
    print(f"[INFO] Rows after alignment: {after_rows}")
    print(f"[INFO] Dropped {before_rows - after_rows} rows")

    # ============================================================
    # 7️⃣ Save
    # ============================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    main()
