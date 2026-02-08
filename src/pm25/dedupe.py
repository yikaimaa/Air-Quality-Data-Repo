from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from .utils import hour_columns


@dataclass(frozen=True)
class DedupeReport:
    n_dup_keys: int
    n_dup_rows: int
    n_dup_extra: int
    n_conflict_keys: int
    n_non_conflict_keys: int


def drop_fully_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicated rows (exact duplicates across all columns).
    """
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def _is_conflict_group(g: pd.DataFrame, hour_cols: List[str]) -> bool:
    nunq = g[hour_cols].apply(lambda s: s.dropna().nunique(), axis=0)
    return (nunq > 1).any()


def _merge_no_conflict_group(g: pd.DataFrame, hour_cols: List[str]) -> pd.Series:
    """
    Complement-only duplicates: fill missing hour values using other rows.
    Choose row with most valid hours as base.
    """
    gg = g.copy()
    valid_cnt = gg[hour_cols].notna().sum(axis=1)
    gg = gg.loc[valid_cnt.sort_values(ascending=False).index]

    base = gg.iloc[0].copy()
    for i in range(1, len(gg)):
        row = gg.iloc[i]
        missing = base[hour_cols].isna() & row[hour_cols].notna()
        idx = missing.index[missing]
        base.loc[idx] = row.loc[idx]
    return base


def _merge_conflict_group(g: pd.DataFrame, hour_cols: List[str]) -> pd.Series:
    """
    True conflicts: for each hour, take mean of distinct non-missing values.
    Also fill meta columns from the first non-null seen.
    """
    gg = g.copy()
    valid_cnt = gg[hour_cols].notna().sum(axis=1)
    gg = gg.loc[valid_cnt.sort_values(ascending=False).index]
    out = gg.iloc[0].copy()

    meta_cols = [c for c in gg.columns if c not in hour_cols]
    for c in meta_cols:
        if pd.isna(out[c]):
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

    return out


def merge_key_duplicates(
    df: pd.DataFrame,
    key: Sequence[str] = ("Station ID", "Pollutant", "Date"),
    replace_values: Sequence[float] = (-999, 9999),
) -> tuple[pd.DataFrame, DedupeReport]:
    """
    Merge rows that share the same key. Two cases:
      1) non-conflict: duplicates complement each other => fill missing
      2) conflict: at least one hour has multiple distinct values => average per hour
    """
    out = df.copy()
    hour_cols = hour_columns(out.columns)
    if not hour_cols:
        raise ValueError("No hour columns found; cannot merge duplicates.")

    out[hour_cols] = out[hour_cols].replace(list(replace_values), np.nan)

    dup_mask = out.duplicated(subset=list(key), keep=False)
    df_dup = out.loc[dup_mask].copy()
    df_solo = out.loc[~dup_mask].copy()

    n_dup_keys = int((out.groupby(list(key)).size() > 1).sum())
    n_dup_rows = int(out.duplicated(subset=list(key), keep=False).sum())
    n_dup_extra = int(out.duplicated(subset=list(key), keep="first").sum())

    conflict_flag = (
        df_dup.groupby(list(key), group_keys=False)
        .apply(lambda g: _is_conflict_group(g, hour_cols))
        .reset_index(name="is_conflict")
    )

    conflict_keys = conflict_flag.loc[conflict_flag["is_conflict"], list(key)].drop_duplicates()
    non_conflict_keys = conflict_flag.loc[~conflict_flag["is_conflict"], list(key)].drop_duplicates()

    df_dup_non_conflict = df_dup.merge(non_conflict_keys, on=list(key), how="inner")
    merged_non_conflict = (
        df_dup_non_conflict.groupby(list(key), group_keys=False)
        .apply(lambda g: _merge_no_conflict_group(g, hour_cols))
        .reset_index(drop=True)
    )

    df_dup_conflict = df_dup.merge(conflict_keys, on=list(key), how="inner")
    merged_conflict = (
        df_dup_conflict.groupby(list(key), group_keys=False)
        .apply(lambda g: _merge_conflict_group(g, hour_cols))
        .reset_index(drop=True)
    )

    final = (
        pd.concat([df_solo, merged_non_conflict, merged_conflict], ignore_index=True)
        .sort_values(list(key))
        .reset_index(drop=True)
    )

    report = DedupeReport(
        n_dup_keys=n_dup_keys,
        n_dup_rows=n_dup_rows,
        n_dup_extra=n_dup_extra,
        n_conflict_keys=len(conflict_keys),
        n_non_conflict_keys=len(non_conflict_keys),
    )
    return final, report
