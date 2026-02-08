from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .utils import require_hour_columns


def add_daily_average(
    df_hourly: pd.DataFrame,
    min_hours: int = 15,
    hour_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-row daily average PM2.5 from hourly columns.

    Adds:
      - n_hours
      - pm25_daily_avg
      - is_low_quality (n_hours < min_hours)

    Drops hour columns (returns a daily-level table).
    """
    df = df_hourly.copy()
    if hour_cols is None:
        hour_cols = require_hour_columns(df, expected=24)

    df["n_hours"] = df[hour_cols].notna().sum(axis=1)
    df["pm25_daily_avg"] = df[hour_cols].mean(axis=1, skipna=True)
    df.loc[df["n_hours"] == 0, "pm25_daily_avg"] = np.nan
    df["is_low_quality"] = df["n_hours"] < int(min_hours)

    return df.drop(columns=hour_cols)
