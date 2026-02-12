# scripts/build_daily_weather_region.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if "region" not in df.columns:
        raise ValueError("Input must contain 'region' column")

    if "date" not in df.columns:
        raise ValueError("Input must contain 'date' column")

    # Drop rows without region
    df = df.dropna(subset=["region"])

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Remove station id related columns from aggregation
    numeric_cols = [c for c in numeric_cols if "station" not in c.lower()]

    agg_dict = {col: "mean" for col in numeric_cols}

    weather_region = (
        df.groupby(["region", "date"], as_index=False)
          .agg(agg_dict)
    )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    weather_region.to_csv(args.out_path, index=False)

    print("Wrote:", args.out_path)
    print("Rows:", len(weather_region))


if __name__ == "__main__":
    main()
