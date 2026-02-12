# scripts/attach_region_to_weather.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", required=True)
    parser.add_argument("--lookup", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    weather = pd.read_csv(args.weather, low_memory=False)
    lookup = pd.read_csv(args.lookup, low_memory=False)

    weather.columns = weather.columns.str.strip()
    lookup.columns = lookup.columns.str.strip()

    print("[INFO] Weather columns:", weather.columns.tolist())
    print("[INFO] Lookup columns:", lookup.columns.tolist())

    # 明确列名
    if "station_id" not in weather.columns:
        raise ValueError("Weather file must contain 'station_id'")

    if "weather_station_id" not in lookup.columns:
        raise ValueError("Lookup file must contain 'weather_station_id'")

    if "assigned_region" not in lookup.columns:
        raise ValueError("Lookup file must contain 'assigned_region'")

    # 强制类型一致（非常重要）
    weather["station_id"] = weather["station_id"].astype(str)
    lookup["weather_station_id"] = lookup["weather_station_id"].astype(str)

    merged = weather.merge(
        lookup[["weather_station_id", "assigned_region"]],
        left_on="station_id",
        right_on="weather_station_id",
        how="left",
    )

    merged = merged.rename(columns={"assigned_region": "region"})

    print("[INFO] Missing region rows:", merged["region"].isna().sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
