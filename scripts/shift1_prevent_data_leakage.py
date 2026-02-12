#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create next-day PM2.5 label for prediction task."
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
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values([args.region_col, args.date_col]).reset_index(drop=True)

    # ------------------------------------------------------------
    # 1️⃣ Create next-day label (shift -1)
    # ------------------------------------------------------------
    df["pm25_label"] = (
        df.groupby(args.region_col)[args.pm_col]
        .shift(-1)
    )

    print("[INFO] pm25_label created using next-day PM2.5 (shift -1)")

    # ------------------------------------------------------------
    # 3️⃣ Drop rows where label is NA (last day per region)
    # ------------------------------------------------------------
    before_rows = len(df)

    df = df.dropna(subset=["pm25_label"])

    after_rows = len(df)
    dropped_rows = before_rows - after_rows

    print("\n==============================")
    print("NEXT-DAY LABEL SUMMARY")
    print("==============================")
    print(f"Rows before drop: {before_rows}")
    print(f"Rows after drop:  {after_rows}")
    print(f"Rows dropped (last day per region): {dropped_rows}")
    print("==============================\n")

    # ------------------------------------------------------------
    # 4️⃣ Save
    # ------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    main()
