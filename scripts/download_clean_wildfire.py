#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load an already-cleaned wildfire dataset from local repo storage
and write it to a specified output path.

Expected input:
    Datasets/wildfire/wildfire_clean_2020_2024.csv.gz

Expected columns:
    latitude
    longitude
    acq_date
    frp
    confidence
    type

This script does NOT download from Google Drive and does NOT redo the
full cleaning pipeline. It simply:
    1. reads the cleaned local wildfire file
    2. validates required columns
    3. optionally standardizes acq_date format
    4. writes the result to the requested output path

Example:
    python scripts/download_clean_wildfire.py \
      --in-clean "Datasets/wildfire/wildfire_clean_2020_2024.csv.gz" \
      --out-clean "outputs/wildfire_clean_2020_2024.csv.gz"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load locally stored cleaned wildfire CSV and write to output path."
    )
    parser.add_argument(
        "--in-clean",
        required=False,
        default="Datasets/wildfire/wildfire_clean_2020_2024.csv.gz",
        help="Path to already-cleaned wildfire file (.csv or .csv.gz).",
    )
    parser.add_argument(
        "--out-clean",
        required=True,
        help="Output path for cleaned wildfire file.",
    )
    return parser.parse_args()


def load_clean_wildfire(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input wildfire file not found: {path.as_posix()}")

    df = pd.read_csv(path)

    required_cols = ["latitude", "longitude", "acq_date", "frp", "confidence", "type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in cleaned wildfire file: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    # Standardize types just to be safe
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["type"] = pd.to_numeric(df["type"], errors="coerce")
    df["confidence"] = df["confidence"].astype("string").str.strip().str.lower()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    # Drop rows with critical missing values, just as a safety check
    df = df.dropna(subset=["latitude", "longitude", "frp", "acq_date"])

    # Standardize date format
    df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")

    df = df.reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    in_clean = Path(args.in_clean)
    out_clean = Path(args.out_clean)
    out_clean.parent.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Reading cleaned wildfire file: {in_clean.as_posix()}")
    df_clean = load_clean_wildfire(in_clean)

    log(f"[INFO] Rows loaded: {len(df_clean):,}")
    log(f"[INFO] Writing cleaned wildfire file to: {out_clean.as_posix()}")

    if out_clean.suffix == ".gz":
        df_clean.to_csv(out_clean, index=False, compression="gzip")
    else:
        df_clean.to_csv(out_clean, index=False)

    log("[OK] Done.")


if __name__ == "__main__":
    main()
