from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

from .ingest import fetch_text


STATION_RE = re.compile(
    r"Station,\s*([^,]*?)\s*\((\d+)\).*?"
    r"Latitude,\s*([-\d.]+)\s*Longitude,\s*([-\d.]+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_station_meta(text: str) -> pd.DataFrame:
    """
    Extract station name/lat/lon records embedded in the raw text.
    """
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


def build_station_lookup(base_raw: str, files: Iterable[str]) -> pd.DataFrame:
    """
    Download station metadata from each file, then dedupe to one row per Station ID.
    """
    meta_dfs: List[pd.DataFrame] = []
    for f in files:
        url = base_raw.rstrip("/") + "/" + f
        text = fetch_text(url)
        meta_dfs.append(extract_station_meta(text))

    station_lookup = pd.concat(meta_dfs, ignore_index=True)
    station_lookup["Station ID"] = pd.to_numeric(station_lookup["Station ID"], errors="coerce").astype("Int64")

    station_lookup = (
        station_lookup.sort_values(["Station ID"])
        .drop_duplicates(subset=["Station ID"], keep="first")
        .reset_index(drop=True)
    )
    return station_lookup


def attach_station_meta(df: pd.DataFrame, station_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join station name/lat/lon onto the hourly dataset.
    """
    out = df.copy()
    out["Station ID"] = pd.to_numeric(out["Station ID"], errors="coerce").astype("Int64")
    lookup = station_lookup.copy()
    lookup["Station ID"] = pd.to_numeric(lookup["Station ID"], errors="coerce").astype("Int64")
    return out.merge(lookup, on="Station ID", how="left")
