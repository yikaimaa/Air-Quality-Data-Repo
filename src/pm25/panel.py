from __future__ import annotations

from typing import Sequence

import pandas as pd

from .utils import hour_columns


def make_balanced_panel(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    key: Sequence[str] = ("Station ID", "Pollutant", "Date"),
    meta_cols: Sequence[str] = ("station_name", "latitude", "longitude"),
) -> pd.DataFrame:
    """
    Create a balanced panel over (Station ID, Pollutant, Date) by expanding all dates,
    then left-joining observed hourly values.

    Keeps station meta columns by re-merging a station_meta table.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()

    hour_cols = hour_columns(out.columns)

    station_meta = (
        out[["Station ID", *meta_cols]]
        .drop_duplicates(subset=["Station ID"])
        .reset_index(drop=True)
    )

    all_dates = pd.date_range(start_date, end_date, freq="D")
    pairs = out[["Station ID", "Pollutant"]].drop_duplicates().reset_index(drop=True)

    full = (
        pairs.assign(_k=1)
        .merge(pd.DataFrame({"Date": all_dates, "_k": 1}), on="_k")
        .drop(columns="_k")
    )

    panel = full.merge(out, on=list(key), how="left")

    panel = (
        panel.drop(columns=list(meta_cols), errors="ignore")
        .merge(station_meta, on="Station ID", how="left")
    )

    if "year" in panel.columns:
        panel["year"] = panel["year"].fillna(panel["Date"].dt.year)
    else:
        panel["year"] = panel["Date"].dt.year
    panel["year"] = panel["year"].astype(int)

    ordered_cols = list(key) + ["year"] + list(meta_cols) + hour_cols
    ordered_cols = [c for c in ordered_cols if c in panel.columns]
    return panel[ordered_cols]
