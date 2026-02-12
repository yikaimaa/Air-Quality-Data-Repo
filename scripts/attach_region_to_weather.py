# scripts/attach_region_to_weather.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach region to weather daily data using nearest-neighbour lookup."
    )

    parser.add_argument("--weather", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    weather_path = Path(args.weather)
    lookup_path = Path(args.lookup)
    out_path = Path(args.out)

    print(f"[INFO] Reading weather: {weather_path}")
    weather = pd.read_csv(weather_path, low_memory=False)

    print(f"[INFO] Reading lookup: {lookup_path}")
    lookup = pd.read_csv(lookup_path, low_memory=False)

    weather.columns = weather.columns.str.strip()
    lookup.columns = lookup.columns.str.strip()

    # 明确使用 weather_station_id
    if "weather_station_id" not in weather.columns:
        raise ValueError("weather file must contain column 'weather_station_id'")

    if "weather_station_id" not in lookup.columns:
        raise ValueError("lookup file must contain column 'weather_station_id'")

    if "assigned_region" not in lookup.columns:
        raise ValueError("lookup file must contain column 'assigned_region'")

    merged = weather.merge(
        lookup[["weather_station_id", "assigned_region"]],
        on="weather_station_id",
        how="left",
    )

    merged = merged.rename(columns={"assigned_region": "region"})

    missing = merged["region"].isna().sum()
    print(f"[INFO] Missing region rows: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
