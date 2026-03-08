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
    python scripts/download_clean_wildfire.py \
      --gdrive-link "https://drive.google.com/file/d/1wf9_Rgeu4xZ1xqTdWuhq1j6QD4jXB5dm/view?usp=sharing" \
      --out-clean "outputs/wildfire_clean.csv"
"""

from __future__ import annotations

import argparse
import html
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin

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
    Extract Google Drive file ID from common share-link formats.

    Supported examples:
      https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
      https://drive.google.com/open?id=<FILE_ID>
      https://drive.google.com/uc?id=<FILE_ID>&export=download
    """
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract Google Drive file ID from link: {link}")


def build_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _save_response_stream(resp: requests.Response, out_path: Path) -> None:
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def download_file(url: str, out_path: Path) -> None:
    """
    Download a file from Google Drive, including the large-file confirm page flow.
    """
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    # First request
    r = session.get(url, stream=True, timeout=120, headers=headers)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()

    # Case 1: already got the real file
    if "text/html" not in content_type:
        _save_response_stream(r, out_path)
        return

    # Case 2: got an HTML warning / confirm page
    html_text = r.text

    # Try several patterns used by Google Drive pages
    patterns = [
        r'href="(/uc\?export=download[^"]+)"',
        r'href="(https://drive\.google\.com/uc\?export=download[^"]+)"',
        r'action="(https://drive\.google\.com/uc\?export=download[^"]+)"',
        r'confirm=([0-9A-Za-z_]+).*?id=([0-9A-Za-z_-]+)',
    ]

    confirm_url = None

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, html_text, flags=re.DOTALL)
        if not match:
            continue

        # Pattern with explicit confirm token and file id
        if i == 3:
            confirm_token = match.group(1)
            file_id = match.group(2)
            confirm_url = (
                f"https://drive.google.com/uc?export=download"
                f"&confirm={confirm_token}&id={file_id}"
            )
        else:
            confirm_url = html.unescape(match.group(1))
            if confirm_url.startswith("/"):
                confirm_url = urljoin("https://drive.google.com", confirm_url)
        break

    if not confirm_url:
        raise ValueError(
            "Could not find Google Drive confirm download link in the warning page."
        )

    log("[INFO] Google Drive returned a warning page; following confirm download link...")

    # Second request: try to fetch actual file
    r2 = session.get(confirm_url, stream=True, timeout=120, headers=headers)
    r2.raise_for_status()

    content_type2 = r2.headers.get("Content-Type", "").lower()
    if "text/html" in content_type2:
        # Still HTML means confirm step failed
        preview = r2.text[:500].replace("\n", " ")
        raise ValueError(
            "Google Drive still returned HTML instead of the CSV file after confirm step. "
            f"Page preview: {preview}"
        )

    _save_response_stream(r2, out_path)


def load_and_clean_wildfire(csv_path: Path) -> pd.DataFrame:
    # Guard against accidentally downloading HTML instead of CSV
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(1000)

    head_lower = head.lower()
    if "<html" in head_lower or "<!doctype html" in head_lower:
        raise ValueError(
            "Downloaded file is HTML, not CSV. Google Drive confirmation step likely failed."
        )

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

    # Convert numeric columns
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["type"] = pd.to_numeric(df["type"], errors="coerce")

    # Normalize confidence
    df["confidence"] = df["confidence"].astype("string").str.strip().str.lower()

    # Parse date
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    # Apply filters
    df = df[df["type"] == 0]
    df = df[df["confidence"].isin({"nominal", "high"})]
    df = df[
        (df["acq_date"] >= pd.Timestamp("2020-01-01")) &
        (df["acq_date"] <= pd.Timestamp("2024-12-31"))
    ]
    df = df.dropna(subset=["latitude", "longitude", "frp", "acq_date"])

    # Store date as YYYY-MM-DD string for cleaner downstream CSV behavior
    df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")

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
