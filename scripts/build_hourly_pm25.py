#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Ontario hourly PM2.5 wide dataset (parse + station meta + de-dup) reproducibly.

Key behaviors:
- Prefer local repo files under --input-dir (default: Datasets/Ontario/PM25)
- If missing locally, fallback to GitHub raw URL base (--base-raw)
- Parse multi-block raw text files into a single hourly-wide table
- Extract station metadata (Station ID, name, lat, lon) from the same raw text
- Merge metadata into hourly data
- Drop fully duplicated rows
- Merge duplicate keys (Station ID, Pollutant, Date):
    * Non-conflict groups: complementary missingness -> fill missing hours
    * Conflict groups: same hour has different values -> take mean of UNIQUE values
- Robust across pandas versions where groupby.apply may exclude grouping columns:
    * We inject key columns from g.name before returning each merged row.

Usage (GitHub Actions):
  uv run --no-project python scripts/build_hourly_pm25.py \
    --input-dir "Datasets/Ontario/PM25" \
    --out "outputs/pm25_hourly_wide_final.csv" \
    --out-station-lookup "outputs/station_lookup.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -------------------------
# Defaults
# -------------------------
DEFAULT_INPUT_DIR = Path("Datasets/Ontario/PM25")
DEFAULT_BASE_RAW = "https://raw.githubusercontent.com/yikaimaa/Air-Quality-Data-Repo/main/Datasets/Ontario/PM25/"
DEFAULT_FILES = [
    "ON_PM25_2020-01-01_2020-12-31.csv",
    "ON_PM25_2021-01-01_2021-12-31.csv",
    "ON_PM25_2022-01-01_2022-12-31.csv",
    "ON_PM25_2023-01-01_2023-12-31.csv",
    "ON_PM25_2024-01-01_2024-12-31.csv",
]

STATION_RE = re.compile(
    r"Station,\s*([^,]*?)\s*\((\d+)\).*?"
    r"Latitude,\s*([-\d.]+)\s*Longitude,\s*([-\d.]+)",
    flags=re.IGNORECASE | re.DOTALL,
)

HOUR_COL_RE = re.compile(r"^H\d{2}$")


# -------------------------
# Utilities
# -------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def infer_year_from_filename(fname: str) -> int:
    m = re.search(r"ON_PM25_(\d{4})-", fname)
    if not m:
        raise ValueError(f"Cannot infer year from filename: {fname}")
    return int(m.group(1))


def read_text_local_or_remote(
    fname: str,
    input_dir: Path,
    base_raw: str,
    timeout: int,
) -> Tuple[str, str]:
    """Return (text, source_label). Prefer local; else fetch from base_raw."""
    local_path = input_dir / fname
    if local_path.exists():
        text = local_path.read_text(encoding="utf-8", errors="replace")
        return text, f"local:{local_path.as_posix()}"

    url = base_raw.rstrip("/") + "/" + fname
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text, f"remote:{url}"


def hour_cols_sorted(cols: Iterable[str]) -> List[str]:
    hours = [c for c in cols if HOUR_COL_RE.fullmatch(str(c))]
    return sorted(hours, key=lambda x: int(str(x)[1:]))


def groupby_apply_robust(gb, func):
    """
    Pandas compatibility:
    - Newer pandas: apply(..., include_groups=False) avoids deprecation warning and future behavior changes
    - Older pandas: include_groups not supported -> fallback
    """
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)


# -------------------------
# Parsing (hourly blocks)
# -------------------------
def parse_ontario_pm25_text(text: str, year: int) -> pd.DataFrame:
    """
    Parse the raw text file that contains repeated blocks with a header line:
        Station ID,Pollutant,Date,H01,...,H24
    and data lines starting with digits (Station ID).
    """
    lines = text.splitlines()

    chunks: List[pd.DataFrame] = []
    cols: Optional[List[str]] = None
    buf: List[str] = []
    in_data = False

    for line in lines:
        line = line.strip("\n")

        if line.startswith("Station ID,Pollutant,Date"):
            cols = [c.strip() for c in line.split(",") if c.strip() != ""]
            in_data = True
            buf = []
            continue

        if not in_data:
            continue

        if line.strip() == "":
            continue

        if re.match(r"^\d{3,},", line):
            buf.append(line.rstrip().rstrip(","))
            continue

        if buf and cols:
            df_chunk = pd.read_csv(StringIO("\n".join(buf)), header=None, names=cols, engine="python")
            chunks.append(df_chunk)

        buf = []
        in_data = False

    if in_data and buf and cols:
        df_chunk = pd.read_csv(StringIO("\n".join(buf)), header=None, names=cols, engine="python")
        chunks.append(df_chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    hour_cols = hour_cols_sorted(df.columns)
    if hour_cols:
        df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")
        df[hour_cols] = df[hour_cols].replace(9999, np.nan)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["year"] = year
    return df


# -------------------------
# Station meta extraction
# -------------------------
def extract_station_meta(text: str) -> pd.DataFrame:
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


def build_station_lookup(texts: List[str]) -> pd.DataFrame:
    meta_dfs = [extract_station_meta(t) for t in texts]
    station_lookup = (
        pd.concat(meta_dfs, ignore_index=True)
        if meta_dfs
        else pd.DataFrame(columns=["Station ID", "station_name", "latitude", "longitude"])
    )

    if station_lookup.empty:
        return station_lookup

    station_lookup["Station ID"] = pd.to_numeric(station_lookup["Station ID"], errors="coerce").astype("Int64")
    station_lookup = (
        station_lookup.sort_values(["Station ID"])
        .drop_duplicates(subset=["Station ID"], keep="first")
        .reset_index(drop=True)
    )
    return station_lookup


# -------------------------
# De-duplication logic
# -------------------------
def drop_fully_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    extra = df.duplicated(keep="first").sum()
    log(f"[INFO] extra fully duplicated rows BEFORE: {extra:,}")

    out = df.drop_duplicates(keep="first").reset_index(drop=True)

    extra_after = out.duplicated(keep="first").sum()
    log(f"[INFO] extra fully duplicated rows AFTER : {extra_after:,}")
    log(f"[INFO] shape BEFORE: {before:,} | AFTER: {out.shape[0]:,}")
    return out


def merge_key_duplicates(df0: pd.DataFrame, key: List[str]) -> pd.DataFrame:
    """
    Merge duplicates by key:
    - non-conflict groups: complementary missingness -> fill missing hours
    - conflict groups: for each hour col, if >1 unique values -> mean of UNIQUE values

    IMPORTANT robustness fix:
    - groupby.apply may exclude grouping columns in newer pandas.
    - We always inject key values from g.name into the returned Series.
    """
    df = df0.copy()

    hour_cols = hour_cols_sorted(df.columns)
    if not hour_cols:
        raise ValueError("No hour columns found matching H\\d{2} (e.g., H01..H24).")

    df[hour_cols] = df[hour_cols].replace([-999, 9999], np.nan)

    dup_mask = df.duplicated(subset=key, keep=False)
    df_dup = df.loc[dup_mask].copy()
    df_solo = df.loc[~dup_mask].copy()

    if df_dup.empty:
        log("[INFO] No duplicated keys found. Skip merge_key_duplicates.")
        return df.sort_values(key).reset_index(drop=True)

    def _inject_group_key(row: pd.Series, group_name) -> pd.Series:
        # group_name is the group key tuple in the same order as `key`
        if not isinstance(group_name, tuple):
            group_name = (group_name,)
        for k, v in zip(key, group_name):
            row[k] = v
        return row

    def is_conflict_group(g: pd.DataFrame) -> bool:
        nunq = g[hour_cols].apply(lambda s: s.dropna().nunique(), axis=0)
        return (nunq > 1).any()

    gb = df_dup.groupby(key, dropna=False, group_keys=False)
    conflict_flag = groupby_apply_robust(gb, is_conflict_group).reset_index(name="is_conflict")

    conflict_keys = conflict_flag.loc[conflict_flag["is_conflict"], key].drop_duplicates()
    non_conflict_keys = conflict_flag.loc[~conflict_flag["is_conflict"], key].drop_duplicates()

    log(f"[INFO] duplicated keys total: {len(conflict_flag):,}")
    log(f"[INFO] conflict keys: {len(conflict_keys):,}")
    log(f"[INFO] non-conflict keys: {len(non_conflict_keys):,}")

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

        return _inject_group_key(base, g.name)

    def merge_conflict_group(g: pd.DataFrame) -> pd.Series:
        gg = g.copy()
        valid_cnt = gg[hour_cols].notna().sum(axis=1)
        gg = gg.loc[valid_cnt.sort_values(ascending=False).index]
        out = gg.iloc[0].copy()

        meta_cols = [c for c in gg.columns if c not in hour_cols]
        for c in meta_cols:
            if pd.isna(out.get(c, np.nan)):
                non_na = gg[c].dropna()
                if len(non_na) > 0:
                    out[c] = non_na.iloc[0]

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

        return _inject_group_key(out, g.name)

    merged_non_conflict = pd.DataFrame()
    if not non_conflict_keys.empty:
        df_dup_non_conflict = df_dup.merge(non_conflict_keys, on=key, how="inner")
        gb_nc = df_dup_non_conflict.groupby(key, dropna=False, group_keys=False)
        merged_non_conflict = (
            groupby_apply_robust(gb_nc, merge_no_conflict_group)
            .reset_index(drop=True)
        )

    merged_conflict = pd.DataFrame()
    if not conflict_keys.empty:
        df_dup_conflict = df_dup.merge(conflict_keys, on=key, how="inner")
        gb_c = df_dup_conflict.groupby(key, dropna=False, group_keys=False)
        merged_conflict = (
            groupby_apply_robust(gb_c, merge_conflict_group)
            .reset_index(drop=True)
        )

    out = (
        pd.concat([df_solo, merged_non_conflict, merged_conflict], ignore_index=True)
        .sort_values(key)
        .reset_index(drop=True)
    )

    remaining = out.duplicated(subset=key, keep=False).sum()
    if remaining != 0:
        raise RuntimeError(f"De-duplication failed: duplicate keys remain after merge (rows={remaining}).")

    return out


# -------------------------
# Main pipeline
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Ontario hourly PM2.5 wide dataset (parse + station meta + de-dup)."
    )
    ap.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    ap.add_argument("--base-raw", type=str, default=DEFAULT_BASE_RAW)
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out", type=str, default="outputs/pm25_hourly_wide_final.csv")
    ap.add_argument("--out-station-lookup", type=str, default="outputs/station_lookup.csv")
    ap.add_argument("--keep-year", action="store_true")

    # Colab/Jupyter safe parsing: ignore notebook-injected args, but keep strict in CLI
    args, unknown = ap.parse_known_args()
    is_notebook = ("ipykernel" in sys.modules) or ("google.colab" in sys.modules)
    if unknown:
        if is_notebook:
            log(f"[INFO] Ignoring notebook args: {unknown}")
        else:
            ap.error(f"unrecognized arguments: {' '.join(unknown)}")

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)
    out_station = Path(args.out_station_lookup)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_station.parent.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    dfs: List[pd.DataFrame] = []

    for f in args.files:
        year = infer_year_from_filename(f)
        text, src = read_text_local_or_remote(f, input_dir, args.base_raw, args.timeout)
        log(f"[INFO] loaded {f} ({src})")
        texts.append(text)

        df_year = parse_ontario_pm25_text(text, year=year)
        if df_year.empty:
            log(f"[WARN] parsed empty dataframe for {f}")
        dfs.append(df_year)

    pm25_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if pm25_all.empty:
        raise RuntimeError("Parsed pm25_all is empty. Check inputs or parsing logic.")

    log(f"[INFO] pm25_all shape: {pm25_all.shape}")

    station_lookup = build_station_lookup(texts)
    log(f"[INFO] station_lookup rows: {len(station_lookup):,}")

    if "Station ID" in pm25_all.columns:
        pm25_all["Station ID"] = pd.to_numeric(pm25_all["Station ID"], errors="coerce").astype("Int64")
    if not station_lookup.empty and "Station ID" in station_lookup.columns:
        station_lookup["Station ID"] = pd.to_numeric(station_lookup["Station ID"], errors="coerce").astype("Int64")

    if station_lookup.empty:
        log("[WARN] station_lookup is empty -> output will not have station_name/lat/lon")
        pm25_hourly_wide = pm25_all.copy()
    else:
        pm25_hourly_wide = pm25_all.merge(station_lookup, on="Station ID", how="left")

    log(f"[INFO] merged hourly wide shape: {pm25_hourly_wide.shape}")

    pm25_hourly_wide_nofulldup = drop_fully_duplicated_rows(pm25_hourly_wide)

    key = ["Station ID", "Pollutant", "Date"]
    for k in key:
        if k not in pm25_hourly_wide_nofulldup.columns:
            raise KeyError(f"Missing key column: {k}")

    pm25_hourly_wide_final = merge_key_duplicates(pm25_hourly_wide_nofulldup, key=key)

    if not args.keep_year and "year" in pm25_hourly_wide_final.columns:
        pm25_hourly_wide_final = pm25_hourly_wide_final.drop(columns=["year"])

    hour_cols = hour_cols_sorted(pm25_hourly_wide_final.columns)
    meta_cols = [c for c in ["Station ID", "Pollutant", "Date", "station_name", "latitude", "longitude"] if c in pm25_hourly_wide_final.columns]
    other_cols = [c for c in pm25_hourly_wide_final.columns if c not in set(meta_cols + hour_cols)]
    pm25_hourly_wide_final = pm25_hourly_wide_final[meta_cols + other_cols + hour_cols]

    remaining = pm25_hourly_wide_final.duplicated(subset=key, keep=False).sum()
    if remaining != 0:
        raise RuntimeError(f"Unexpected duplicate keys remain at end (rows={remaining}).")

    log(f"[INFO] final shape: {pm25_hourly_wide_final.shape}")
    log("[INFO] writing outputs...")

    pm25_hourly_wide_final.to_csv(out_path, index=False)
    log(f"[INFO] wrote: {out_path.as_posix()}")

    if not station_lookup.empty:
        station_lookup.to_csv(out_station, index=False)
        log(f"[INFO] wrote: {out_station.as_posix()}")

    log("[OK] Done.")


if __name__ == "__main__":
    main()
