#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Ontario daily PM2.5 dataset from the hourly-wide output.

Input (default):
  outputs/pm25_hourly_wide_final.csv   (produced by scripts/build_hourly_pm25.py)

Output (default):
  outputs/pm25_daily.csv

Computation:
- hour_cols = H01..H24
- n_hours = count of non-missing hourly values
- pm25_daily_avg = mean of H01..H24 with skipna=True
- if n_hours == 0 -> pm25_daily_avg = NaN
- (optional) if n_hours < min_hours -> pm25_daily_avg = NaN
- drop hour columns by default, keep metadata + n_hours + pm25_daily_avg

NEW (region mapping):
- Attach/ensure a 'region' column via Station ID -> region lookup CSV.
- Default lookup path (repo): Datasets/Ontario/pm25_station_region_lookup.csv
  Expected columns: Station ID, region
- If input already has a 'region' column, we fill missing values from the lookup.

GitHub Actions example:
  uv run --no-project python scripts/build_daily_pm25.py \
    --in "outputs/pm25_hourly_wide_final.csv" \
    --station-region-lookup "Datasets/Ontario/pm25_station_region_lookup.csv" \
    --out "outputs/pm25_daily.csv" \
    --min-hours 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build daily PM2.5 averages from hourly-wide CSV.")
    ap.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="outputs/pm25_hourly_wide_final.csv",
        help="Input hourly-wide CSV (from build_hourly_pm25.py).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="outputs/pm25_daily.csv",
        help="Output daily CSV path.",
    )
    ap.add_argument(
        "--min-hours",
        dest="min_hours",
        type=int,
        default=0,
        help="If n_hours < min_hours, set pm25_daily_avg to NaN. Default 0 (no extra filtering).",
    )
    ap.add_argument(
        "--keep-hour-cols",
        action="store_true",
        help="Keep H01..H24 columns in output (default drops them).",
    )
    ap.add_argument(
        "--station-region-lookup",
        dest="station_region_lookup",
        type=str,
        default="Datasets/Ontario/pm25_station_region_lookup.csv",
        help="CSV path with Station ID -> region mapping (columns: Station ID, region).",
    )
    ap.add_argument(
        "--require-region",
        action="store_true",
        help="Fail if region lookup is missing or if any Station ID has no region mapping.",
    )

    # Notebook-friendly: ignore injected args, strict in CLI
    args, unknown = ap.parse_known_args()
    is_notebook = ("ipykernel" in sys.modules) or ("google.colab" in sys.modules)
    if unknown:
        if is_notebook:
            log(f"[INFO] Ignoring notebook args: {unknown}")
        else:
            ap.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.min_hours < 0 or args.min_hours > 24:
        ap.error("--min-hours must be between 0 and 24.")

    return args


def _load_region_lookup(path: Path) -> pd.DataFrame:
    reg = pd.read_csv(path)
    # normalize column names (case-insensitive)
    cols_lower = {c.lower(): c for c in reg.columns}
    if "station id" not in cols_lower or "region" not in cols_lower:
        raise ValueError(
            f"Region lookup must contain columns 'Station ID' and 'region'. "
            f"Got columns: {list(reg.columns)}"
        )
    reg = reg.rename(columns={cols_lower["station id"]: "Station ID", cols_lower["region"]: "region"})
    reg["Station ID"] = pd.to_numeric(reg["Station ID"], errors="coerce").astype("Int64")
    reg["region"] = reg["region"].astype("string").str.strip()
    reg = reg.dropna(subset=["Station ID"]).drop_duplicates(subset=["Station ID"], keep="first")
    return reg[["Station ID", "region"]]


def attach_region(df: pd.DataFrame, lookup_path: Path, require_region: bool) -> pd.DataFrame:
    """
    Ensure df has a 'region' column by mapping Station ID via lookup CSV.
    - If df already has region: fill missing values from lookup (do not overwrite non-missing)
    - Else: create region from lookup
    """
    if "Station ID" not in df.columns:
        raise KeyError("Input is missing required column: 'Station ID'.")

    df_out = df.copy()
    df_out["Station ID"] = pd.to_numeric(df_out["Station ID"], errors="coerce").astype("Int64")

    if not lookup_path.exists():
        msg = f"Station region lookup not found: {lookup_path.as_posix()}"
        if require_region:
            raise FileNotFoundError(msg)
        log(f"[WARN] {msg} (continuing without region mapping)")
        if "region" not in df_out.columns:
            df_out["region"] = pd.NA
        return df_out

    reg = _load_region_lookup(lookup_path)

    if "region" in df_out.columns:
        # Fill missing from lookup
        df_out["region"] = df_out["region"].astype("string")
        tmp = df_out.merge(reg, on="Station ID", how="left", suffixes=("", "_lkp"))
        df_out["region"] = tmp["region"].where(tmp["region"].notna(), tmp["region_lkp"])
    else:
        df_out = df_out.merge(reg, on="Station ID", how="left")

    missing = int(df_out["region"].isna().sum())
    if missing > 0:
        log(f"[WARN] daily rows missing region after mapping: {missing:,} / {len(df_out):,}")
        if require_region:
            # show missing station ids (unique)
            miss_ids = (
                df_out.loc[df_out["region"].isna(), "Station ID"]
                .dropna()
                .drop_duplicates()
                .astype("Int64")
                .tolist()
            )
            raise RuntimeError(
                "Region mapping required but missing for Station ID(s): "
                + ", ".join(str(x) for x in miss_ids)
            )
    else:
        log("[INFO] region mapping complete (no missing).")

    return df_out


def build_daily(df: pd.DataFrame, min_hours: int, keep_hour_cols: bool) -> pd.DataFrame:
    hour_cols: List[str] = [f"H{i:02d}" for i in range(1, 25)]
    missing = [c for c in hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing hour columns: {missing}")

    # Ensure numeric
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")

    df["n_hours"] = df[hour_cols].notna().sum(axis=1)

    df["pm25_daily_avg"] = df[hour_cols].mean(axis=1, skipna=True)

    # If no hours at all -> NaN
    df.loc[df["n_hours"] == 0, "pm25_daily_avg"] = np.nan

    # Optional quality threshold
    if min_hours > 0:
        df.loc[df["n_hours"] < min_hours, "pm25_daily_avg"] = np.nan

    if not keep_hour_cols:
        df = df.drop(columns=hour_cols)

    return df


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path.as_posix()}. "
            f"Run scripts/build_hourly_pm25.py first (or pass --in)."
        )

    log(f"[INFO] reading: {in_path.as_posix()}")
    df = pd.read_csv(in_path)

    # Attach/ensure region
    df = attach_region(df, lookup_path=Path(args.station_region_lookup), require_region=bool(args.require_region))

    before_shape = df.shape
    df_out = build_daily(df, min_hours=args.min_hours, keep_hour_cols=args.keep_hour_cols)

    log(f"[INFO] input shape: {before_shape} -> output shape: {df_out.shape}")

    n_total = len(df_out)
    n_nan = int(df_out["pm25_daily_avg"].isna().sum())
    log(f"[INFO] pm25_daily_avg NaN rows: {n_nan:,} / {n_total:,}")

    log(f"[INFO] writing: {out_path.as_posix()}")
    df_out.to_csv(out_path, index=False)

    log("[OK] Done.")


if __name__ == "__main__":
    main()
