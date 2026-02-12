#!/usr/bin/env python3
"""
shift1_prevent_data_leakage.py

Purpose:
    Perform strict causal alignment to prevent data leakage:
        - target = pm25_next_day (shift -1)
        - ALL other features shift +1
    This is NOT feature engineering.
    This is time alignment before modeling.

Usage:
    python shift1_prevent_data_leakage.py \
        --input outputs/clean_dataset.csv \
        --output outputs/model_ready_shift1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply shift(1) to all features to prevent data leakage."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned dataset"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save aligned dataset"
    )

    parser.add_argument(
        "--region-col",
        default="region",
        help="Region column name (default: region)"
    )

    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column name (default: date)"
    )

    parser.add_argument(
        "--pm-col",
        default="pm25_daily_avg",
        help="PM2.5 column name (default: pm25_daily_avg)"
    )

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

    # Ensure proper temporal order
    df = df.sort_values([args.region_col, args.date_col]).reset_index(drop=True)

    # ============================================================
    # 1️⃣ Create target: next-day PM2.5
    # ============================================================

    df["target_pm25_next_day"] = (
        df.groupby(args.region_col)[args.pm_col]
        .shift(-1)
    )

    # ============================================================
    # 2️⃣ Identify feature columns
    # ============================================================

    exclude_cols = {
        args.region_col,
        args.date_col,
        args.pm_col,
        "target_pm25_next_day"
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"[INFO] Shifting {len(feature_cols)} feature columns by +1")

    # ============================================================
    # 3️⃣ Shift all features
    # ============================================================

    df[feature_cols] = (
        df.groupby(args.region_col)[feature_cols]
        .shift(1)
    )

    # ============================================================
    # 4️⃣ Drop misaligned rows
    # ============================================================

    before_rows = len(df)

    df = df.dropna(subset=["target_pm25_next_day"])
    df = df.dropna(subset=feature_cols)

    after_rows = len(df)

    print(f"[INFO] Rows before alignment: {before_rows}")
    print(f"[INFO] Rows after alignment: {after_rows}")
    print(f"[INFO] Dropped {before_rows - after_rows} rows")

    # ============================================================
    # 5️⃣ Save
    # ============================================================

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    main()
