#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download wildfire CSV from Google Drive, keep only selected columns,
apply light cleaning, and output ONE cleaned CSV.

Keeps only:
    latitude
    longitude
    acq_date
    frp
    confidence
    type

Filters:
    - type == 0
    - confidence in {"nominal", "high"}
    - 2020-01-01 <= acq_date <= 2024-12-31
    - drop rows missing latitude / longitude / frp

Example:
    python download_clean_wildfire.py \
      --gdrive-link "https://drive.google.com/file/d/1wf9_Rgeu4xZ1xqTdWuhq1j6QD4jXB5dm/view?usp=sharing" \
      --out-clean "outputs/wildfire_clean.csv"
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import requests


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and lightly clean wildfire CSV from Google Drive."
    )
    parser.add_argument(
        "--gdrive-link",
        required=True,
        help="Google Drive share link for the wildfire CSV.",
    )
    parser.add_argument(
        "--out-clean",
        required=True,
        help="Output path for cleaned CSV.",
    )
    return parser.parse_args()


def extract_gdrive_file_id(link: str) -> str:
    """
    Extract Google Drive file ID from a standard share link.

    Example:
    https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    """
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract Google Drive file ID from link: {link}")


def build_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url: str, out_path: Path) -> None:
    """
    Download file from URL to out_path.
    """
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def load_and_clean_wildfire(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["latitude", "longitude", "acq_date", "frp", "confidence", "type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in wildfire CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only required columns
    df = df[required_cols].copy()

    # Parse numeric columns
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["type"] = pd.to_numeric(df["type"], errors="coerce")

    # Normalize confidence
    df["confidence"] = df["confidence"].astype("string").str.strip().str.lower()

    # Parse date
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.date

    # Apply filters
    df = df[df["type"] == 0]
    df = df[df["confidence"].isin({"nominal", "high"})]
    df = df[
        (pd.to_datetime(df["acq_date"], errors="coerce") >= pd.Timestamp("2020-01-01"))
        & (pd.to_datetime(df["acq_date"], errors="coerce") <= pd.Timestamp("2024-12-31"))
    ]
    df = df.dropna(subset=["latitude", "longitude", "frp", "acq_date"])

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main() -> None:
    args = parse_args()

    out_clean = Path(args.out_clean)
    out_clean.parent.mkdir(parents=True, exist_ok=True)

    file_id = extract_gdrive_file_id(args.gdrive_link)
    download_url = build_direct_download_url(file_id)

    log(f"[INFO] Google Drive file id: {file_id}")
    log("[INFO] Downloading wildfire CSV...")

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "wildfire_raw.csv"
        download_file(download_url, raw_path)

        log(f"[INFO] Temporary raw file downloaded to: {raw_path}")
        log("[INFO] Reading and cleaning wildfire data...")

        df_clean = load_and_clean_wildfire(raw_path)

    log(f"[INFO] Cleaned rows: {len(df_clean):,}")
    log(f"[INFO] Writing cleaned CSV to: {out_clean}")
    df_clean.to_csv(out_clean, index=False)

    log("[OK] Done.")


if __name__ == "__main__":
    main()
