#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import argparse



def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def find_station_id_col(df: pd.DataFrame) -> str | None:
    # Common names across weather datasets
    candidates = [
        "station_id", "stationId", "station", "station_name",
        "climate_id", "climateId", "wmo", "wmo_id", "id"
    ]
    # Heuristic: any column containing "station" and "id"
    for c in df.columns:
        lc = c.lower()
        if "station" in lc and "id" in lc:
            return c
    return pick_first_existing_col(df, candidates)


def main():
    # Repo-relative paths (script lives in scripts/)
    # BASE_DIR = Path(__file__).resolve().parent.parent
    # DATA_DIR = BASE_DIR / "Datasets" / "Ontario"

    # WEATHER_PATH = DATA_DIR / "ON_weather_daily_merged_2020-2025_clean.csv"
    # OUT_PATH = DATA_DIR / "weather_stations_latlon.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    WEATHER_PATH = Path(args.weather)
    OUT_PATH = Path(args.out)  

    df = pd.read_csv(WEATHER_PATH)

    print(f"Loaded weather rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Required columns per your dataset
    if "longitude_x" not in df.columns or "latitude_y" not in df.columns:
        raise ValueError(
            "Expected columns 'longitude_x' and 'latitude_y' not found.\n"
            f"Found: {list(df.columns)}"
        )

    # Rename to standard names
    df = df.rename(columns={"longitude_x": "lon", "latitude_y": "lat"})

    # Coerce to numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    missing_coords = df["lat"].isna().sum() + df["lon"].isna().sum()
    print(f"Missing lat or lon cells (not rows): {missing_coords:,}")

    # Keep rows with coords
    df_coords = df.dropna(subset=["lat", "lon"]).copy()
    print(f"Rows with valid coords: {len(df_coords):,}")

    # Try to find a station id column
    station_id_col = find_station_id_col(df_coords)
    if station_id_col:
        print(f"Detected station id column: {station_id_col}")

        stations = (
            df_coords[[station_id_col, "lat", "lon"]]
            .drop_duplicates(subset=[station_id_col])
            .rename(columns={station_id_col: "weather_station_id"})
            .reset_index(drop=True)
        )

    else:
        print("No station id column detected. Creating station ids from rounded lat/lon.")
        # Create a stable-ish key (round to 5 decimals ~ 1m precision-ish)
        tmp = df_coords[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        tmp["weather_station_id"] = (
            "lat" + tmp["lat"].round(5).astype(str) + "_lon" + tmp["lon"].round(5).astype(str)
        )
        stations = tmp[["weather_station_id", "lat", "lon"]]

    print(f"Unique weather stations: {len(stations):,}")

    # Sort for determinism
    stations = stations.sort_values(["weather_station_id"]).reset_index(drop=True)

    stations.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()