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

    # 自动找共同的 station 列
    possible_cols = set(weather.columns).intersection(set(lookup.columns))

    station_col = None
    for col in possible_cols:
        if "station" in col.lower():
            station_col = col
            break

    if station_col is None:
        raise ValueError(
            f"Could not detect common station column. "
            f"Weather columns={weather.columns.tolist()} "
            f"Lookup columns={lookup.columns.tolist()}"
        )

    print(f"[INFO] Using station column: {station_col}")

    if "assigned_region" not in lookup.columns:
        raise ValueError("lookup file must contain column 'assigned_region'")

    merged = weather.merge(
        lookup[[station_col, "assigned_region"]],
        on=station_col,
        how="left",
    )

    merged = merged.rename(columns={"assigned_region": "region"})

    print("[INFO] Missing region rows:", merged["region"].isna().sum())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
