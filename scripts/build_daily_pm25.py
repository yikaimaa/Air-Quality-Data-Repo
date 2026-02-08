#!/usr/bin/env python3
"""
Build Ontario PM2.5 *daily* dataset from an hourly-wide CSV.

Input:
- A CSV that contains hourly columns H01..H24 (e.g., pm25_hourly_wide_final.csv or pm25_hourly_wide_panel.csv)

Output:
- Daily dataset with:
  - n_hours: number of non-missing hourly readings in the day
  - pm25_daily_avg: mean of available hourly readings (NaN if n_hours == 0)

This script was refactored from an exploratory EDA notebook and keeps only data processing steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def compute_daily(df: pd.DataFrame, min_hours_low_quality: int = 15, keep_hourly: bool = False) -> pd.DataFrame:
    hour_cols = [f"H{i:02d}" for i in range(1, 25)]
    missing = [c for c in hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing hour columns in input: {missing}")

    out = df.copy()

    # ensure numeric hourly columns (in case they were read as strings)
    out[hour_cols] = out[hour_cols].apply(pd.to_numeric, errors="coerce")

    out["n_hours"] = out[hour_cols].notna().sum(axis=1)
    out["pm25_daily_avg"] = out[hour_cols].mean(axis=1, skipna=True)
    out.loc[out["n_hours"] == 0, "pm25_daily_avg"] = np.nan

    out["is_low_quality"] = out["n_hours"] < int(min_hours_low_quality)

    if not keep_hourly:
        out = out.drop(columns=hour_cols)

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-hourly",
        type=str,
        default="data/processed/pm25_hourly_wide_final.csv",
        help="Input hourly-wide CSV (must include H01..H24).",
    )
    ap.add_argument(
        "--out-daily",
        type=str,
        default="data/processed/pm25_daily_cleaned.csv",
        help="Output daily CSV path.",
    )
    ap.add_argument(
        "--min-hours",
        type=int,
        default=15,
        help="Threshold for low-quality day flag: is_low_quality = (n_hours < min_hours).",
    )
    ap.add_argument(
        "--keep-hourly",
        action="store_true",
        help="If set, keep H01..H24 columns in output.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_hourly)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_csv(in_path)

    df_daily = compute_daily(df, min_hours_low_quality=args.min_hours, keep_hourly=args.keep_hourly)

    out_path = Path(args.out_daily)
    ensure_parent_dir(out_path)
    df_daily.to_csv(out_path, index=False)

    low_cnt = int(df_daily["is_low_quality"].sum())
    print(f"[OK] wrote daily: {out_path}  rows={len(df_daily):,}  low_quality_rows={low_cnt:,}")


if __name__ == "__main__":
    main()
