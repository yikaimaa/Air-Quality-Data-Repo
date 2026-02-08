from __future__ import annotations

import re
from io import StringIO
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests


def parse_ontario_pm25_text(text: str, year: int) -> pd.DataFrame:
    """
    Parse the Ontario PM2.5 raw text file (metadata + one CSV block).
    Returns a wide hourly table with H01..H24 columns.
    """
    lines = text.splitlines()

    chunks: List[pd.DataFrame] = []
    cols: Optional[List[str]] = None
    buf: List[str] = []
    in_data = False

    for line in lines:
        if line.startswith("Station ID,Pol'utant,Date"):
            cols = [c for c in line.split(",") if c != ""]
            in_data = True
            buf = []
            continue

        if not in_data:
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

    hour_cols = [c for c in df.columns if re.fullmatch(r"H\d{2}", str(c))]
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce").replace(9999, np.nan)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = int(year)
    return df


def fetch_text(url: str, timeout: int = 60) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def load_years(base_raw: str, files: Iterable[str]) -> pd.DataFrame:
    """
    Download + parse multiple annual files, concatenate into one dataframe.
    """
    dfs: List[pd.DataFrame] = []
    for f in files:
        m = re.search(r"ON_PM25_(\d{4})-", f)
        if not m:
            raise ValueError(f"Cannot infer year from filename: {f}")
        year = int(m.group(1))
        url = base_raw.rstrip("/") + "/" + f
        text = fetch_text(url)
        dfs.append(parse_ontario_pm25_text(text, year=year))
    return pd.concat(dfs, ignore_index=True)
