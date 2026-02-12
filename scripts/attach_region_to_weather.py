# scripts/attach_region_to_weather.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach region to weather daily data using nearest-neighbour lookup."
    )

    parser.add_argument("--weather", required=True, help="Clean weather daily CSV")
    parser.add_argument("--lookup", required=True, help="Weather station-region lookup CSV")
    parser.add_argument("--out", required=True, help="Output CSV path")

    args = parser.parse_args()

    weather_path = Path(args.weather)
    lookup_path = Path(args.lookup)
    out_path = Path(args.out)

    if not weather_path.exists():
        raise FileNotFoundError(f"Weather file not found: {weather_path}")

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup file not found: {lookup_path}")

    print(f"[INFO] Reading weather: {weather_path}")
    weather = pd.read_csv(weather_path)

    print(f"[INFO] Reading lookup: {lookup_path}")
    lookup = pd.read_csv(lookup_path)

    # Standardize column names
    weather.columns = weather.columns.str.strip()
    lookup.columns = lookup.columns.str.strip()

    # Expect station id column to exist
    if "station_id" not in weather.columns:
        raise ValueError("weather CSV must contain column 'station_id'")

    if "station_id" not in lookup.columns:
        raise ValueError("lookup CSV must contain column 'station_id'")

    if "assigned_region" not in lookup.columns:
        raise ValueError("lookup CSV must contain column 'assigned_region'")

    # Merge
    merged = weather.merge(
        lookup[["station_id", "assigned_region"]],
        on="station_id",
        how="left",
    )

    merged = merged.rename(columns={"assigned_region": "region"})

    # Check missing regions
    missing = merged["region"].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} rows have missing region")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
