#!/usr/bin/env python3
"""
Build cleaned Ontario PM2.5 *hourly wide* dataset (and optional complete date panel).

This script was refactored from an exploratory EDA notebook:
- Keeps data ingestion + cleaning + de-duplication + (optional) panel expansion
- Removes all EDA/plotting

Outputs (CSV):
- pm25_hourly_wide_final.csv (default): de-duplicated, merged station meta, one row per (Station ID, Pollutant, Date)
- pm25_hourly_wide_panel.csv (optional): full grid of (Station ID, Pollutant) x Date within [start_year, end_year], left-joined with hourly data
"""

from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_BASE_RAW = "https://raw.githubusercontent.com/yikaimaa/Air-Quality-Data-Repo/main/Datasets/Ontario/PM25/"

STATION_RE = re.compile(
    r"Station,\s*([^,]*?)\s*\((\d+)\).*?"
    r"Latitude,\s*([-\d.]+)\s*Longitude,\s*([-\d.]+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_ontario_pm25_text(text: str, year: int) -> pd.DataFrame:
    """Parse the hourly table section from the raw text file."""
    lines = text.splitlines()

    chunks = []
    cols = None
    buf = []
    in_data = False

    for line in lines:
        if line.startswith("Station ID,Pollutant,Date"):
            cols = [c for c in line.split(",") if c != ""]
            in_data = True
            buf = []
            continue

        if not in_data:
            continue

        # data lines begin with numeric station id (>=3 digits)
        if re.match(r"^\d{3,},", line):
            buf.append(line.rstrip().rstrip(","))
            continue

        # end of the data section
        if buf and cols:
            df_chunk = pd.read_csv(StringIO("\n".join(buf)), header=None, names=cols, engine="python")
            chunks.append(df_chunk)
        buf = []
        in_data = False

    # handle file end while still in data section
    if in_data and buf and cols:
        df_chunk = pd.read_csv(StringIO("\n".join(buf)), header=None, names=cols, engine="python")
        chunks.append(df_chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # numeric cleanup for hourly columns
    hour_cols = [c for c in df.columns if re.fullmatch(r"H\d{2}", str(c))]
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce").replace([9999, -999], np.nan)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = int(year)
    return df


def extract_station_meta(text: str) -> pd.DataFrame:
    """Extract station name + lat/lon from the raw text header blocks."""
    records = []
    for m in STATION_RE.finditer(text):
        records.append(
            {
                "Station ID": int(m.group(2)),
                "station_name": m.group(1).strip(),
                "latitude": float(m.group(3)),
                "longitude": float(m.group(4)),
            }
        )
    return pd.DataFrame(records, columns=["Station ID", "station_name", "latitude", "longitude"])


def year_to_filename(year: int) -> str:
    return f"ON_PM25_{year}-01-01_{year}-12-31.csv"


def download_text(url: str, timeout: int) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def build_hourly_wide(
    years: Iterable[int],
    base_raw: str,
    timeout: int,
) -> pd.DataFrame:
    """Download + parse + merge station meta + de-duplicate / merge key duplicates."""
    dfs: List[pd.DataFrame] = []
    meta_dfs: List[pd.DataFrame] = []

    for y in years:
        fname = year_to_filename(y)
        url = base_raw + fname
        raw_text = download_text(url, timeout=timeout)

        df_year = parse_ontario_pm25_text(raw_text, year=y)
        dfs.append(df_year)

        meta_dfs.append(extract_station_meta(raw_text))

    pm25_all = pd.concat(dfs, ignore_index=True)

    station_lookup = (
        pd.concat(meta_dfs, ignore_index=True)
        .sort_values(["Station ID"])
        .drop_duplicates(subset=["Station ID"], keep="first")
        .reset_index(drop=True)
    )

    # merge station meta
    pm25_all["Station ID"] = pd.to_numeric(pm25_all["Station ID"], errors="coerce").astype("Int64")
    station_lookup["Station ID"] = pd.to_numeric(station_lookup["Station ID"], errors="coerce").astype("Int64")

    pm25_hourly_wide = pm25_all.merge(station_lookup, on="Station ID", how="left")

    # remove fully duplicated rows
    df0 = pm25_hourly_wide.drop_duplicates(keep="first").reset_index(drop=True)

    key = ["Station ID", "Pollutant", "Date"]
    hour_cols = [c for c in df0.columns if re.fullmatch(r"H\d{2}", str(c))]
    df0[hour_cols] = df0[hour_cols].replace([9999, -999], np.nan)

    # split duplicated keys
    dup_mask = df0.duplicated(subset=key, keep=False)
    df_dup = df0.loc[dup_mask].copy()
    df_solo = df0.loc[~dup_mask].copy()

    if df_dup.empty:
        return df0.sort_values(key).reset_index(drop=True)

    def is_conflict_group(g: pd.DataFrame) -> bool:
        nunq = g[hour_cols].apply(lambda s: s.dropna().nunique(), axis=0)
        return bool((nunq > 1).any())

    conflict_flag = (
        df_dup.groupby(key, group_keys=False).apply(is_conflict_group).reset_index(name="is_conflict")
    )

    conflict_keys = conflict_flag.loc[conflict_flag["is_conflict"], key].drop_duplicates()
    non_conflict_keys = conflict_flag.loc[~conflict_flag["is_conflict"], key].drop_duplicates()

    def merge_no_conflict_group(g: pd.DataFrame) -> pd.Series:
        gg = g.copy()
        valid_cnt = gg[hour_cols].notna().sum(axis=1)
        gg = gg.loc[valid_cnt.sort_values(ascending=False).index]

        base = gg.iloc[0].copy()
        for i in range(1, len(gg)):
            row = gg.iloc[i]
            missing = base[hour_cols].isna() & row[hour_cols].notna()
            idx = missing.index[missing]
            base.loc[idx] = row.loc[idx]
        return base

    df_dup_non_conflict = df_dup.merge(non_conflict_keys, on=key, how="inner")
    merged_non_conflict = (
        df_dup_non_conflict.groupby(key, group_keys=False).apply(merge_no_conflict_group).reset_index(drop=True)
    )

    def merge_conflict_group(g: pd.DataFrame) -> pd.Series:
        gg = g.copy()
        valid_cnt = gg[hour_cols].notna().sum(axis=1)
        gg = gg.loc[valid_cnt.sort_values(ascending=False).index]
        out = gg.iloc[0].copy()

        # fill missing meta cols from any row (first non-null)
        meta_cols = [c for c in gg.columns if c not in hour_cols]
        for c in meta_cols:
            if pd.isna(out[c]):
                non_na = gg[c].dropna()
                if len(non_na) > 0:
                    out[c] = non_na.iloc[0]

        # resolve conflicts per hour by mean of unique numeric values
        for h in hour_cols:
            vals = gg[h].dropna()
            if vals.empty:
                out[h] = np.nan
                continue
            uniq = pd.unique(vals)
            if len(uniq) == 1:
                out[h] = uniq[0]
            else:
                uniq_num = pd.to_numeric(pd.Series(uniq), errors="coerce").dropna().values
                out[h] = float(np.mean(uniq_num)) if len(uniq_num) else np.nan

        return out

    df_dup_conflict = df_dup.merge(conflict_keys, on=key, how="inner")
    merged_conflict = (
        df_dup_conflict.groupby(key, group_keys=False).apply(merge_conflict_group).reset_index(drop=True)
    )

    out = (
        pd.concat([df_solo, merged_non_conflict, merged_conflict], ignore_index=True)
        .sort_values(key)
        .reset_index(drop=True)
    )

    # final validation: should be no duplicate keys
    if out.duplicated(subset=key, keep=False).any():
        raise RuntimeError("De-duplication failed: duplicate keys still exist after merging.")

    return out


def build_panel(df_hourly_wide_final: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Create full (Station ID, Pollutant) x Date daily grid and left-join hourly values."""
    df = df_hourly_wide_final.copy()

    key = ["Station ID", "Pollutant", "Date"]
    hour_cols = [c for c in df.columns if re.fullmatch(r"H\d{2}", str(c))]
    meta_cols = ["station_name", "latitude", "longitude"]

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    station_meta = (
        df[["Station ID"] + [c for c in meta_cols if c in df.columns]]
        .drop_duplicates(subset=["Station ID"])
        .reset_index(drop=True)
    )

    all_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")

    pairs = df[["Station ID", "Pollutant"]].drop_duplicates().reset_index(drop=True)
    full = (
        pairs.assign(_k=1).merge(pd.DataFrame({"Date": all_dates, "_k": 1}), on="_k").drop(columns="_k")
    )

    panel = full.merge(df, on=key, how="left")

    panel = (
        panel.drop(columns=[c for c in meta_cols if c in panel.columns], errors="ignore")
        .merge(station_meta, on="Station ID", how="left")
    )

    if "year" in panel.columns:
        panel["year"] = panel["year"].fillna(panel["Date"].dt.year)
    else:
        panel["year"] = panel["Date"].dt.year
    panel["year"] = panel["year"].astype(int)

    ordered_cols = key + ["year"] + [c for c in meta_cols if c in panel.columns] + hour_cols
    ordered_cols = [c for c in ordered_cols if c in panel.columns]
    return panel[ordered_cols]


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--base-raw", type=str, default=DEFAULT_BASE_RAW)
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument(
        "--out-hourly",
        type=str,
        default="data/processed/pm25_hourly_wide_final.csv",
        help="Output path for the cleaned hourly wide CSV.",
    )
    ap.add_argument(
        "--make-panel",
        action="store_true",
        help="If set, also create the complete date panel CSV.",
    )
    ap.add_argument(
        "--out-panel",
        type=str,
        default="data/processed/pm25_hourly_wide_panel.csv",
        help="Output path for the panel CSV (only used if --make-panel is set).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start_year must be <= end_year")

    years = list(range(args.start_year, args.end_year + 1))

    df_hourly = build_hourly_wide(years=years, base_raw=args.base_raw, timeout=args.timeout)

    out_hourly = Path(args.out_hourly)
    ensure_parent_dir(out_hourly)
    df_hourly.to_csv(out_hourly, index=False)

    print(f"[OK] wrote hourly wide: {out_hourly}  rows={len(df_hourly):,}  cols={df_hourly.shape[1]}")

    if args.make_panel:
        df_panel = build_panel(df_hourly, start_year=args.start_year, end_year=args.end_year)
        out_panel = Path(args.out_panel)
        ensure_parent_dir(out_panel)
        df_panel.to_csv(out_panel, index=False)
        print(f"[OK] wrote panel:       {out_panel}  rows={len(df_panel):,}  cols={df_panel.shape[1]}")


if __name__ == "__main__":
    main()
