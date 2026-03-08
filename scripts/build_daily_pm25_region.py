#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build region-day PM2.5 + wildfire features from an hourly-wide PM2.5 dataset.

Input (default):
  outputs/pm25_hourly_wide_final.csv   (produced by scripts/build_hourly_pm25.py)

Wildfire input (default):
  Datasets/wildfire/wildfire_clean_2020_2024.csv.gz

Output (default):
  outputs/pm25_daily_region.csv

Definition:
- First compute station-day daily mean from H01..H24:
    pm25_daily_avg = mean(H01..H24, skip NaN)
    n_hours        = count of non-missing hourly values
    if n_hours == 0 -> pm25_daily_avg = NaN
    if min_hours > 0 and n_hours < min_hours -> pm25_daily_avg = NaN

- Then construct station-day wildfire features using same-day fire spots:
    fire_count_50km
    fire_count_100km
    frp_sum_100km
    min_fire_distance_km

- Then aggregate to region-day:
    pm25_region_daily_avg = mean(pm25_daily_avg across stations in the region, skip NaN)
    fire_count_50km_avg   = mean(fire_count_50km across stations)
    fire_count_100km_avg  = mean(fire_count_100km across stations)
    frp_sum_100km_avg     = mean(frp_sum_100km across stations)
    min_fire_distance_km  = min(min_fire_distance_km across stations)

Region mapping:
- Requires a 'region' column. If missing / partially missing, it can be filled via a
  Station ID -> region lookup CSV (default: Datasets/Ontario/pm25_station_region_lookup.csv)
  Expected columns: Station ID, region
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
    ap = argparse.ArgumentParser(description="Build region-day PM2.5 + wildfire features from hourly-wide CSV.")

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
        default="Datasets/Ontario/intermediate_tables/pm25_station_region_lookup.csv",
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
    ap.add_argument(
        "--wildfire",
        dest="wildfire_path",
        type=str,
        default="Datasets/wildfire/wildfire_clean_2020_2024.csv.gz",
        help="Path to cleaned wildfire CSV(.gz) with columns latitude, longitude, acq_date, frp.",
    )
    ap.add_argument(
        "--no-wildfire",
        action="store_true",
        help="Skip wildfire feature construction and only build PM2.5 region-day output.",
    )

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


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")
    return None


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

    keep_cols = [c for c in out.columns if c not in hour_cols]
    return out[keep_cols]


def haversine_km_vec(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_wildfire(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Wildfire file not found: {path.as_posix()}")

    wf = pd.read_csv(path)
    required = ["latitude", "longitude", "acq_date", "frp"]
    missing = [c for c in required if c not in wf.columns]
    if missing:
        raise KeyError(f"Wildfire file missing required columns: {missing}")

    wf = wf[required].copy()
    wf["latitude"] = pd.to_numeric(wf["latitude"], errors="coerce")
    wf["longitude"] = pd.to_numeric(wf["longitude"], errors="coerce")
    wf["frp"] = pd.to_numeric(wf["frp"], errors="coerce")
    wf["acq_date"] = pd.to_datetime(wf["acq_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wf = wf.dropna(subset=["latitude", "longitude", "frp", "acq_date"]).reset_index(drop=True)
    return wf


def add_station_day_wildfire_features(
    df_station_day: pd.DataFrame,
    date_col: str,
    wildfire_path: Path,
) -> pd.DataFrame:
    lat_col = pick_col(df_station_day, ["latitude", "lat", "Latitude", "LAT"])
    lon_col = pick_col(df_station_day, ["longitude", "lon", "Longitude", "LON", "long"])

    out = df_station_day.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    wf = load_wildfire(wildfire_path)
    wf_groups = {d: g for d, g in wf.groupby("acq_date", sort=False)}

    out["fire_count_50km"] = 0.0
    out["fire_count_100km"] = 0.0
    out["frp_sum_100km"] = 0.0
    out["min_fire_distance_km"] = np.nan

    valid_station_mask = out[lat_col].notna() & out[lon_col].notna() & out[date_col].notna()
    if valid_station_mask.sum() == 0:
        log("[WARN] No station-day rows with valid lat/lon/date for wildfire feature construction.")
        return out

    for dt, idx in out.loc[valid_station_mask].groupby(date_col).groups.items():
        wf_day = wf_groups.get(dt)
        if wf_day is None or wf_day.empty:
            continue

        fire_lat = wf_day["latitude"].to_numpy()
        fire_lon = wf_day["longitude"].to_numpy()
        fire_frp = wf_day["frp"].to_numpy()

        day_idx = list(idx)
        for i in day_idx:
            s_lat = float(out.at[i, lat_col])
            s_lon = float(out.at[i, lon_col])

            dists = haversine_km_vec(s_lat, s_lon, fire_lat, fire_lon)

            within_50 = dists <= 50.0
            within_100 = dists <= 100.0

            out.at[i, "fire_count_50km"] = float(within_50.sum())
            out.at[i, "fire_count_100km"] = float(within_100.sum())
            out.at[i, "frp_sum_100km"] = float(fire_frp[within_100].sum()) if within_100.any() else 0.0
            out.at[i, "min_fire_distance_km"] = float(dists.min()) if len(dists) > 0 else np.nan

    return out


def build_region_day(df_station_day: pd.DataFrame, date_col: str) -> pd.DataFrame:
    required = {date_col, "region", "Station ID", "pm25_daily_avg"}
    missing = [c for c in required if c not in df_station_day.columns]
    if missing:
        raise KeyError(f"Missing required columns for aggregation: {missing}")

    df = df_station_day.copy()
    df["Station ID"] = pd.to_numeric(df["Station ID"], errors="coerce").astype("Int64")
    df["region"] = df["region"].astype("string")

    agg_map = {
        "pm25_region_daily_avg": ("pm25_daily_avg", "mean"),
        "n_stations_total": ("Station ID", "nunique"),
        "n_stations_used": ("pm25_daily_avg", lambda s: int(s.notna().sum())),
    }

    if "fire_count_50km" in df.columns:
        agg_map["fire_count_50km_avg"] = ("fire_count_50km", "mean")
    if "fire_count_100km" in df.columns:
        agg_map["fire_count_100km_avg"] = ("fire_count_100km", "mean")
    if "frp_sum_100km" in df.columns:
        agg_map["frp_sum_100km_avg"] = ("frp_sum_100km", "mean")
    if "min_fire_distance_km" in df.columns:
        agg_map["min_fire_distance_km"] = ("min_fire_distance_km", "min")

    out = (
        df.groupby([date_col, "region"], dropna=False)
        .agg(**agg_map)
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

    log(f"[INFO] reading PM2.5 hourly-wide input: {in_path.as_posix()}")
    df = pd.read_csv(in_path)

    df = attach_region(df, lookup_path=Path(args.station_region_lookup), require_region=bool(args.require_region))

    date_col = args.date_col.strip() or detect_date_col(df)
    if date_col not in df.columns:
        raise KeyError(f"--date-col='{date_col}' not found in input columns.")

    df_station_day = compute_station_day(df, min_hours=args.min_hours)

    n_total = len(df_station_day)
    n_nan = int(df_station_day["pm25_daily_avg"].isna().sum())
    log(f"[INFO] station-day computed rows: {n_total:,}; NaN daily avg: {n_nan:,}")

    if not args.no_wildfire:
        log(f"[INFO] reading wildfire input: {args.wildfire_path}")
        df_station_day = add_station_day_wildfire_features(
            df_station_day=df_station_day,
            date_col=date_col,
            wildfire_path=Path(args.wildfire_path),
        )

        log(
            "[INFO] wildfire features added: "
            "fire_count_50km, fire_count_100km, frp_sum_100km, min_fire_distance_km"
        )

    df_region_day = build_region_day(df_station_day, date_col=date_col)

    n_total_r = len(df_region_day)
    n_nan_r = int(df_region_day["pm25_region_daily_avg"].isna().sum())
    log(f"[INFO] region-day rows: {n_total_r:,}; NaN region daily avg: {n_nan_r:,}")

    log(f"[INFO] writing region-day output: {out_path.as_posix()}")
    df_region_day.to_csv(out_path, index=False)

    log("[OK] Done.")


if __name__ == "__main__":
    main()
