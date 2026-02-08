#!/usr/bin/env python3
"""
Run EDA as a reproducible script and save figures/tables under paper/eda/.

Default inputs:
- data/processed/pm25_hourly_wide_panel.csv (preferred)
- data/processed/pm25_hourly_wide_final.csv (fallback)

Outputs:
- paper/eda/figures/*.png (+ optional pdf)
- paper/eda/tables/*.csv and *.md

This intentionally contains NO interactive display (no plt.show()).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HOUR_RE = re.compile(r"^H\d{2}$")


def repo_root() -> Path:
    # scripts/run_eda_paper.py -> repo/
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_hour_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if HOUR_RE.fullmatch(str(c))]
    # Sort by hour number (H01..H24)
    return sorted(cols, key=lambda x: int(str(x)[1:]))


def save_table(df: pd.DataFrame, out_base: Path, index: bool = False) -> None:
    """
    Save a table as CSV + (if available) Markdown.
    out_base should NOT include extension.
    """
    df.to_csv(out_base.with_suffix(".csv"), index=index)
    try:
        df.to_markdown(out_base.with_suffix(".md"), index=index)
    except Exception:
        # Markdown is optional
        pass


def save_fig(out_path: Path, formats: List[str]) -> None:
    for fmt in formats:
        p = out_path.with_suffix("." + fmt)
        plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()


def load_hourly(in_path: Path | None) -> pd.DataFrame:
    root = repo_root()
    if in_path is None:
        candidates = [
            root / "data/processed/pm25_hourly_wide_panel.csv",
            root / "data/processed/pm25_hourly_wide_final.csv",
        ]
        for c in candidates:
            if c.exists():
                in_path = c
                break

    if in_path is None or not in_path.exists():
        raise FileNotFoundError(
            "Could not find hourly input. Provide --in-hourly, or create "
            "data/processed/pm25_hourly_wide_panel.csv (preferred) / pm25_hourly_wide_final.csv."
        )

    df = pd.read_csv(in_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hourly", type=str, default=None, help="Path to hourly-wide CSV (panel or final).")
    ap.add_argument("--outdir", type=str, default="paper/eda", help="Output directory (default: paper/eda).")
    ap.add_argument("--formats", nargs="+", default=["png"], help="Figure formats, e.g. png pdf")
    ap.add_argument("--event-thresh", type=float, default=25.0, help="Event day threshold for daily mean PM2.5.")
    ap.add_argument("--event-min-hours", type=int, default=18, help="Min valid hours for an event day.")
    ap.add_argument("--station-bar-topn", type=int, default=30, help="Top-N stations to show in station bar plot.")
    args = ap.parse_args()

    root = repo_root()
    outdir = root / args.outdir
    fig_dir = outdir / "figures"
    tbl_dir = outdir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(tbl_dir)

    df = load_hourly(Path(args.in_hourly) if args.in_hourly else None)
    hour_cols = find_hour_cols(df)

    if len(hour_cols) < 12:
        raise ValueError(f"Expected many hour columns like H01..H24; found only {len(hour_cols)}: {hour_cols}")

    # Coerce hour columns numeric
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")

    # ---------- 1) Missingness over time (all-day missing) ----------
    all_day_missing = df[hour_cols].isna().all(axis=1)
    ts_all_missing = (
        df.assign(all_day_missing=all_day_missing)
          .groupby("Date", as_index=False)["all_day_missing"]
          .sum()
          .rename(columns={"all_day_missing": "all_day_missing_stationdays"})
          .sort_values("Date")
          .reset_index(drop=True)
    )
    save_table(ts_all_missing, tbl_dir / "missing_all_day_over_time", index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(ts_all_missing["Date"], ts_all_missing["all_day_missing_stationdays"])
    plt.title("Ontario PM2.5: All-day-missing station-days over time")
    plt.xlabel("Date")
    plt.ylabel("Count of station-days with all 24 hours missing")
    save_fig(fig_dir / "missing_all_day_over_time", args.formats)

    # ---------- 2) Daily mean distribution (province-level) ----------
    df["pm25_24h_mean_station"] = df[hour_cols].mean(axis=1, skipna=True)
    ontario_daily = (
        df.groupby("Date")["pm25_24h_mean_station"]
          .mean()
          .rename("ontario_daily_mean")
          .reset_index()
          .sort_values("Date")
          .reset_index(drop=True)
    )
    s = ontario_daily["ontario_daily_mean"].dropna()
    summary = pd.DataFrame({
        "metric": ["n_days", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"],
        "value": [
            int(len(s)),
            float(s.mean()),
            float(s.std()),
            float(s.min()),
            float(s.quantile(0.50)),
            float(s.quantile(0.90)),
            float(s.quantile(0.95)),
            float(s.quantile(0.99)),
            float(s.max()),
        ],
    })
    save_table(summary, tbl_dir / "ontario_daily_mean_summary", index=False)
    save_table(ontario_daily, tbl_dir / "ontario_daily_mean_timeseries", index=False)

    plt.figure(figsize=(10, 4))
    plt.hist(s, bins=120)
    plt.yscale("log")
    plt.xlabel("Ontario daily mean PM2.5 (µg/m³)")
    plt.ylabel("Count (log scale)")
    plt.title("Distribution of Ontario Daily Mean PM2.5")
    save_fig(fig_dir / "ontario_daily_mean_hist", args.formats)

    # ---------- 3) Monthly station-weighted mean ----------
    df["month"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    station_month = (
        df.groupby(["Station ID", "month"])["pm25_24h_mean_station"]
          .mean()
          .reset_index()
    )
    monthly_station_weighted = (
        station_month.groupby("month")["pm25_24h_mean_station"]
          .mean()
          .reset_index()
          .rename(columns={"pm25_24h_mean_station": "station_weighted_mean"})
          .sort_values("month")
          .reset_index(drop=True)
    )
    save_table(monthly_station_weighted, tbl_dir / "monthly_station_weighted_mean", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly_station_weighted["month"],
        monthly_station_weighted["station_weighted_mean"],
        marker="o",
    )
    plt.title("Ontario PM2.5 by Month (Station-weighted mean)")
    plt.xlabel("Month")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.xticks(rotation=45)
    save_fig(fig_dir / "pm25_by_month_station_weighted", args.formats)

    # ---------- 4) Mean PM2.5 by hour ----------
    hourly_mean = df[hour_cols].mean(axis=0, skipna=True)
    hours = [int(c[1:]) for c in hour_cols]

    hourly_table = pd.DataFrame({"hour": hours, "mean_pm25": hourly_mean.values})
    save_table(hourly_table, tbl_dir / "mean_pm25_by_hour", index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(hours, hourly_mean.values, marker="o")
    plt.xticks(hours)
    plt.xlabel("Hour (1–24)")
    plt.ylabel("Mean PM2.5 (µg/m³)")
    plt.title("Mean PM2.5 by Hour (H01–H24)")
    plt.grid(True, alpha=0.3)
    save_fig(fig_dir / "mean_pm25_by_hour", args.formats)

    # ---------- 5) Mean PM2.5 by station (top-N bar) ----------
    df["pm25_daily_mean"] = df[hour_cols].mean(axis=1, skipna=True)
    group_cols = ["Station ID"]
    if "station_name" in df.columns:
        group_cols.append("station_name")

    station_mean = (
        df.groupby(group_cols, as_index=False)
          .agg(
              mean_pm25=("pm25_daily_mean", "mean"),
              n_days_non_missing=("pm25_daily_mean", "count"),
          )
          .sort_values("mean_pm25", ascending=False)
          .reset_index(drop=True)
    )
    save_table(station_mean, tbl_dir / "mean_pm25_by_station_all", index=False)

    topn = max(5, int(args.station_bar_topn))
    plot_df = station_mean.head(topn).copy()
    if "station_name" in plot_df.columns:
        plot_df["station_label"] = plot_df["station_name"].fillna("Unknown").astype(str) + " (" + plot_df["Station ID"].astype(str) + ")"
    else:
        plot_df["station_label"] = plot_df["Station ID"].astype(str)

    plt.figure(figsize=(14, 6))
    plt.bar(plot_df["station_label"], plot_df["mean_pm25"])
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Station")
    plt.ylabel("Mean daily PM2.5 (µg/m³)")
    plt.title(f"Mean PM2.5 by Station (Top {topn})")
    save_fig(fig_dir / "mean_pm25_by_station_topn", args.formats)

    # ---------- 6) Seasonality: month-of-year station-weighted ----------
    df["month_num"] = pd.to_datetime(df["Date"]).dt.month
    station_moy = (
        df.groupby(["Station ID", "month_num"])["pm25_24h_mean_station"]
          .mean()
          .reset_index()
    )
    moy = (
        station_moy.groupby("month_num")["pm25_24h_mean_station"]
          .mean()
          .reset_index()
          .rename(columns={"pm25_24h_mean_station": "pm25_month_mean_station_weighted"})
          .sort_values("month_num")
          .reset_index(drop=True)
    )
    save_table(moy, tbl_dir / "seasonality_month_of_year", index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(moy["month_num"], moy["pm25_month_mean_station_weighted"])
    plt.xticks(range(1, 13))
    plt.title("Ontario PM2.5 Seasonality (Month-of-Year, Station-weighted)")
    plt.xlabel("Month (1–12)")
    plt.ylabel("PM2.5 (µg/m³)")
    save_fig(fig_dir / "seasonality_month_of_year", args.formats)

    # ---------- 7) Simple station-level event episodes (table) ----------
    df["valid_hours"] = df[hour_cols].notna().sum(axis=1)
    df["event_day"] = (df["pm25_24h_mean_station"] >= float(args.event_thresh)) & (df["valid_hours"] >= int(args.event_min_hours))
    df = df.sort_values(["Station ID", "Date"]).reset_index(drop=True)

    # Define episode starts when event_day is True and previous day for same station is not event_day or not consecutive date
    prev_date = df.groupby("Station ID")["Date"].shift(1)
    prev_event = df.groupby("Station ID")["event_day"].shift(1)
    is_consecutive = (df["Date"] - prev_date) == pd.Timedelta(days=1)

    new_episode = df["event_day"] & (~(prev_event.fillna(False) & is_consecutive))
    df["episode_id"] = new_episode.groupby(df["Station ID"]).cumsum()
    # Keep only event days for episode aggregation
    ev = df[df["event_day"]].copy()

    if len(ev) > 0:
        ep = (
            ev.groupby(["Station ID", "episode_id"], as_index=False)
              .agg(
                  start_date=("Date", "min"),
                  end_date=("Date", "max"),
                  n_days=("Date", "count"),
                  mean_pm25=("pm25_24h_mean_station", "mean"),
                  max_pm25=("pm25_24h_mean_station", "max"),
              )
              .sort_values(["n_days", "max_pm25"], ascending=[False, False])
              .reset_index(drop=True)
        )
        save_table(ep, tbl_dir / "event_episodes", index=False)
    else:
        # Save an empty table for reproducibility
        ep = pd.DataFrame(columns=["Station ID", "episode_id", "start_date", "end_date", "n_days", "mean_pm25", "max_pm25"])
        save_table(ep, tbl_dir / "event_episodes", index=False)

    print("EDA outputs written to:", outdir)
    print("Figures:", fig_dir)
    print("Tables:", tbl_dir)


if __name__ == "__main__":
    main()
