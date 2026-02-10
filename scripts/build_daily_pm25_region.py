#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build region-day PM2.5 (daily mean) from an hourly-wide dataset.

Input (default):
  outputs/pm25_hourly_wide_final.csv   (produced by scripts/build_hourly_pm25.py)

Output (default):
  outputs/pm25_daily_region.csv

Definition:
- First compute station-day daily mean from H01..H24:
    pm25_daily_avg = mean(H01..H24, skip NaN)
    n_hours        = count of non-missing hourly values
    if n_hours == 0 -> pm25_daily_avg = NaN
    if min_hours > 0 and n_hours < min_hours -> pm25_daily_avg = NaN
- Then aggregate to region-day:
    pm25_region_daily_avg = mean(pm25_daily_avg across stations in the region, skip NaN)
    (so NA station-day values do NOT enter the denominator)

Region mapping:
- Requires a 'region' column. If missing / partially missing, it can be filled via a
  Station ID -> region lookup CSV (default: Datasets/Ontario/pm25_station_region_lookup.csv)
  Expected columns: Station ID, region

Example:
  uv run --no-project python scripts/build_daily_pm25.py \
    --in "outputs/pm25_hourly_wide_final.csv" \
    --station-region-lookup "Datasets/Ontario/pm25_station_region_lookup.csv" \
    --out "outputs/pm25_daily_region.csv" \
    --min-hours 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build region-day PM2.5 daily averages from hourly-wide CSV.")

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
        default="outputs/pm25_daily_region.csv",
        help="Output region-day CSV path.",
    )
    ap.add_argument(
        "--min-hours",
        dest="min_hours",
        type=int,
        default=0,
        help="If a station-day has n_hours < min_hours, set station-day daily avg to NaN before aggregation.",
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
    ap.add_argument(
        "--date-col",
        dest="date_col",
        type=str,
        default="",
        help="Optional: explicitly set the date column name (otherwise auto-detected).",
    )

    # Notebook-friendly parsing
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
    cols_lower = {c.lower(): c for c in reg.columns}
    if "station id" not in cols_lower or "region" not in cols_lower:
        raise ValueError(
            "Region lookup must contain columns 'Station ID' and 'region'. "
            f"Got columns: {list(reg.columns)}"
        )
    reg = reg.rename(columns={cols_lower["station id"]: "Station ID", cols_lower["region"]: "region"})
    reg["Station ID"] = pd.to_numeric(reg["Station ID"], errors="coerce").astype("Int64")
    reg["region"] = reg["region"].astype("string").str.strip()
    reg = reg.dropna(subset=["Station ID"]).drop_duplicates(subset=["Station ID"], keep="first")
    return reg[["Station ID", "region"]]


def attach_region(df: pd.DataFrame, lookup_path: Path, require_region: bool) -> pd.DataFrame:
    if "Station ID" not in df.columns:
        raise KeyError("Input is missing required column: 'Station ID'.")

    out = df.copy()
    out["Station ID"] = pd.to_numeric(out["Station ID"], errors="coerce").astype("Int64")

    if not lookup_path.exists():
        msg = f"Station region lookup not found: {lookup_path.as_posix()}"
        if require_region:
            raise FileNotFoundError(msg)
        log(f"[WARN] {msg} (continuing without region mapping)")
        if "region" not in out.columns:
            out["region"] = pd.NA
        return out

    reg = _load_region_lookup(lookup_path)

    if "region" in out.columns:
        out["region"] = out["region"].astype("string")
        tmp = out.merge(reg, on="Station ID", how="left", suffixes=("", "_lkp"))
        out["region"] = tmp["region"].where(tmp["region"].notna(), tmp["region_lkp"])
    else:
        out = out.merge(reg, on="Station ID", how="left")

    missing = int(out["region"].isna().sum())
    if missing > 0:
        log(f"[WARN] rows missing region after mapping: {missing:,} / {len(out):,}")
        if require_region:
            miss_ids = (
                out.loc[out["region"].isna(), "Station ID"].dropna().drop_duplicates().astype("Int64").tolist()
            )
            raise RuntimeError(
                "Region mapping required but missing for Station ID(s): " + ", ".join(str(x) for x in miss_ids)
            )
    else:
        log("[INFO] region mapping complete (no missing).")

    return out


def detect_date_col(df: pd.DataFrame) -> str:
    candidates = [
        "DATE",
        "Date",
        "date",
        "DATE_LOCAL",
        "date_local",
        "DATE_PST",
        "DATE_EST",
        "DATE_UTC",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        if str(c).strip().lower() == "date":
            return str(c)

    raise KeyError(
        "Could not find a date column to aggregate by. "
        "Expected one of: DATE/Date/date/DATE_PST/DATE_EST/DATE_UTC, etc. "
        "Or pass --date-col explicitly."
    )


def compute_station_day(df: pd.DataFrame, min_hours: int) -> pd.DataFrame:
    hour_cols: List[str] = [f"H{i:02d}" for i in range(1, 25)]
    missing = [c for c in hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing hour columns: {missing}")

    out = df.copy()
    out[hour_cols] = out[hour_cols].apply(pd.to_numeric, errors="coerce")

    out["n_hours"] = out[hour_cols].notna().sum(axis=1)
    out["pm25_daily_avg"] = out[hour_cols].mean(axis=1, skipna=True)

    out.loc[out["n_hours"] == 0, "pm25_daily_avg"] = np.nan
    if min_hours > 0:
        out.loc[out["n_hours"] < min_hours, "pm25_daily_avg"] = np.nan

    # keep only what we need for aggregation
    keep_cols = [c for c in out.columns if c not in hour_cols]
    return out[keep_cols]


def build_region_day(df_station_day: pd.DataFrame, date_col: str) -> pd.DataFrame:
    required = {date_col, "region", "Station ID", "pm25_daily_avg"}
    missing = [c for c in required if c not in df_station_day.columns]
    if missing:
        raise KeyError(f"Missing required columns for aggregation: {missing}")

    df = df_station_day.copy()
    df["Station ID"] = pd.to_numeric(df["Station ID"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype("string")

    g = df.groupby([date_col, "region"], dropna=False)
    out = (
        g.agg(
            pm25_region_daily_avg=("pm25_daily_avg", "mean"),
            n_stations_total=("Station ID", "nunique"),
            n_stations_used=("pm25_daily_avg", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )

    return out


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {in_path.as_posix()}. "
            "Run scripts/build_hourly_pm25.py first (or pass --in)."
        )

    log(f"[INFO] reading: {in_path.as_posix()}")
    df = pd.read_csv(in_path)

    # Attach/ensure region
    df = attach_region(df, lookup_path=Path(args.station_region_lookup), require_region=bool(args.require_region))

    date_col = args.date_col.strip() or detect_date_col(df)
    if date_col not in df.columns:
        raise KeyError(f"--date-col='{date_col}' not found in input columns.")

    df_station_day = compute_station_day(df, min_hours=args.min_hours)

    # Quick sanity logs
    n_total = len(df_station_day)
    n_nan = int(df_station_day["pm25_daily_avg"].isna().sum())
    log(f"[INFO] station-day computed rows: {n_total:,}; NaN daily avg: {n_nan:,}")

    df_region_day = build_region_day(df_station_day, date_col=date_col)

    n_total_r = len(df_region_day)
    n_nan_r = int(df_region_day["pm25_region_daily_avg"].isna().sum())
    log(f"[INFO] region-day rows: {n_total_r:,}; NaN region daily avg: {n_nan_r:,}")

    log(f"[INFO] writing region-day: {out_path.as_posix()}")
    df_region_day.to_csv(out_path, index=False)

    log("[OK] Done.")


if __name__ == "__main__":
    main()
