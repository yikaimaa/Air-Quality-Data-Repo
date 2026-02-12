#!/usr/bin/env python3
"""
fill_missing_time_series.py

Purpose:
    Ensure complete daily time series per region from
    2020-01-01 to 2024-12-31.

Behavior:
    - For each region, create full date range
    - Reindex to complete region × date grid
    - Newly created rows will contain NaN for feature columns
    - Does NOT modify existing values
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill missing daily dates for each region."
    )

    parser.add_argument("--input", required=True, help="Path to input dataset")
    parser.add_argument("--output", required=True, help="Path to save completed dataset")
    parser.add_argument("--region-col", default="region")
    parser.add_argument("--date-col", default="date")

    return parser.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Reading: {input_path}")
    df = pd.read_csv(input_path)

    if args.region_col not in df.columns or args.date_col not in df.columns:
        raise ValueError("Input must contain region and date columns")

    df[args.date_col] = pd.to_datetime(df[args.date_col])

    # Create full date range
    full_dates = pd.date_range(START_DATE, END_DATE, freq="D")

    regions = df[args.region_col].unique()

    print(f"[INFO] Regions found: {len(regions)}")
    print(f"[INFO] Date range length: {len(full_dates)} days")

    completed_frames = []

    for region in regions:
        region_df = df[df[args.region_col] == region].copy()
        region_df = region_df.set_index(args.date_col)

        # Reindex to full date range
        region_df = region_df.reindex(full_dates)

        # Restore region column
        region_df[args.region_col] = region

        # Reset index back to column
        region_df = region_df.reset_index().rename(columns={"index": args.date_col})

        completed_frames.append(region_df)

    full_df = pd.concat(completed_frames, ignore_index=True)

    before_rows = len(df)
    after_rows = len(full_df)

    print(f"[INFO] Rows before fill: {before_rows}")
    print(f"[INFO] Rows after fill: {after_rows}")
    print(f"[INFO] Added rows: {after_rows - before_rows}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved to: {output_path}")


if __name__ == "__main__":
    main()
