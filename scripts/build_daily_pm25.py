from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pm25.features import add_daily_average


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hourly", required=True, help="Hourly cleaned dataset (parquet or csv)")
    p.add_argument("--out", required=True, help="Daily cleaned output path (csv)")
    p.add_argument("--min-hours", type=int, default=15)
    args = p.parse_args()

    hourly_path = Path(args.hourly)
    if hourly_path.suffix.lower() in [".parquet", ".pq"]:
        df_hourly = pd.read_parquet(hourly_path)
    else:
        df_hourly = pd.read_csv(hourly_path)

    df_daily = add_daily_average(df_hourly, min_hours=args.min_hours)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
