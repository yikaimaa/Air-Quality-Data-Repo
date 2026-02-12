#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift all non-pm25 features by +1 to prevent data leakage."
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

    # Ensure correct temporal order
    df = df.sort_values([args.region_col, args.date_col]).reset_index(drop=True)

    # Rename pm25 column (NO SHIFT)
    df = df.rename(columns={args.pm_col: "pm25_label"})
    print("[INFO] pm25_region_daily_avg -> renamed to pm25_label")

    # Identify feature columns to shift
    exclude_cols = {
        args.region_col,
        args.date_col,
        "pm25_label",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"[INFO] Number of feature columns to shift: {len(feature_cols)}")

    # Shift features
    shifted = (
        df.groupby(args.region_col)[feature_cols]
        .shift(1)
    )

    # Rename shifted columns
    shifted.columns = [f"{c}_lag" for c in feature_cols]

    # Drop original feature columns
    df = df.drop(columns=feature_cols)

    # Concatenate shifted features
    df = pd.concat([df, shifted], axis=1)

    # Drop NA rows after shift
    before_rows = len(df)
    df = df.dropna()
    after_rows = len(df)
    dropped_rows = before_rows - after_rows

    print("\n==============================")
    print("SHIFT1 SUMMARY")
    print("==============================")
    print(f"Rows before dropna: {before_rows}")
    print(f"Rows after dropna:  {after_rows}")
    print(f"Rows dropped due to shift: {dropped_rows}")
    print("==============================\n")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    main()
