from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


_HOUR_RE = re.compile(r"^H\d{2}$")


def hour_columns(columns: Iterable[str]) -> List[str]:
    """
    Return sorted hour columns (H01..H24) present in the input.
    """
    cols = [str(c) for c in columns if _HOUR_RE.fullmatch(str(c))]
    cols = sorted(cols, key=lambda x: int(x[1:]))
    return cols


def require_hour_columns(df: pd.DataFrame, expected: int = 24) -> List[str]:
    """
    Detect and validate hour columns. Raises if none found.
    """
    cols = hour_columns(df.columns)
    if not cols:
        raise ValueError("No hour columns found (expected H01..H24).")
    if expected is not None and len(cols) != expected:
        raise ValueError(f"Detected {len(cols)} hour columns; expected {expected}. Found: {cols}")
    return cols
