from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    weather = pd.read_csv(args.weather, low_memory=False)
    lookup = pd.read_csv(args.lookup, low_memory=False)

    weather.columns = weather.columns.str.strip()
    lookup.columns = lookup.columns.str.strip()

    weather["station_id"] = weather["station_id"].astype(int).astype(str)
    lookup["weather_station_id"] = lookup["weather_station_id"].astype(float).astype(int).astype(str)

    merged = weather.merge(
        lookup[["weather_station_id", "assigned_region"]],
        left_on="station_id",
        right_on="weather_station_id",
        how="left",
    )

    merged = merged.rename(columns={"assigned_region": "region"})

    print("Missing region rows:", merged["region"].isna().sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
