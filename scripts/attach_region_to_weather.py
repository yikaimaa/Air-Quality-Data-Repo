"""
Attach region to the daily weather table using the weather_station_region_lookup_nn table.

Inputs:
- ../Datasets/Ontario/ON_weather_daily_merged_2020-2025_clean.csv
- ../Datasets/Ontario/weather_station_region_lookup_nn.csv   (your NN lookup output)

Output:
- ../Datasets/Ontario/ON_weather_daily_merged_2020-2025_with_region.csv

Notes:
- We attach `assigned_region` -> `region`
- We optionally drop stations that failed the NN threshold (within_threshold == False)
  (default: keep only mapped stations; set KEEP_UNMAPPED=True to keep all with region=NA)
"""

import pandas as pd

WEATHER_PATH = "../Datasets/Ontario/ON_weather_daily_merged_2020-2025_clean.csv"
LOOKUP_PATH = "../Datasets/Ontario/weather_station_region_lookup_nn.csv"
OUT_PATH = "../Datasets/Ontario/ON_weather_daily_merged_2020-2025_with_region.csv"

KEEP_UNMAPPED = False  # set True if you want to keep stations with no region assignment


def main():
    weather = pd.read_csv(WEATHER_PATH)
    lookup = pd.read_csv(LOOKUP_PATH)

    # --- normalize keys ---
    # Weather uses: station_id (int)
    # Lookup uses: weather_station_id (string or int)
    if "station_id" not in weather.columns:
        raise ValueError(f"Expected `station_id` in weather table. Found: {list(weather.columns)}")

    if "weather_station_id" not in lookup.columns:
        raise ValueError(
            f"Expected `weather_station_id` in lookup table. Found: {list(lookup.columns)}"
        )

    # Ensure compatible merge dtype
    weather["station_id"] = pd.to_numeric(weather["station_id"], errors="coerce").astype("Int64")
    lookup["weather_station_id"] = pd.to_numeric(lookup["weather_station_id"], errors="coerce").astype("Int64")

    # Validate presence of region column in lookup
    if "assigned_region" not in lookup.columns:
        raise ValueError(
            "Expected `assigned_region` in lookup table (weather_station_region_lookup_nn). "
            f"Found: {list(lookup.columns)}"
        )

    # Optionally require threshold pass
    if "within_threshold" in lookup.columns and not KEEP_UNMAPPED:
        lookup_use = lookup.loc[lookup["within_threshold"] == True, ["weather_station_id", "assigned_region"]].copy()
    else:
        lookup_use = lookup[["weather_station_id", "assigned_region"]].copy()

    lookup_use = lookup_use.rename(columns={"weather_station_id": "station_id", "assigned_region": "region"})

    # --- merge ---
    out = weather.merge(lookup_use, on="station_id", how="left")

    if not KEEP_UNMAPPED:
        # drop rows where no region could be assigned
        out = out.dropna(subset=["region"]).copy()

    # --- sanity checks ---
    n_total = len(weather)
    n_out = len(out)
    n_unmapped = out["region"].isna().sum() if "region" in out.columns else None

    print(f"Weather rows (input): {n_total:,}")
    print(f"Weather rows (output): {n_out:,}")
    if KEEP_UNMAPPED:
        print(f"Unmapped rows kept (region is NA): {n_unmapped:,}")
    else:
        print("Unmapped rows dropped.")

    # Save
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print("Example regions:", out["region"].astype(str).value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
