import pandas as pd

WEATHER_PATH = "../Datasets/Ontario/processed_datasets/ON_weather_daily_merged_2020-2025_clean.csv"
PM25_DAILY_PATH = "../Datasets/Ontario/processed_datasets/pm25_daily_cleaned.csv"
JOINED_TABLE_PATH = "../Datasets/Ontario/processed_datasets/pm25_weather_daily_region_joined.csv"

def print_schema(path: str, name: str):
    print("\n" + "=" * 80)
    print(f"{name}")
    print(f"Path: {path}")
    print("=" * 80)

    df = pd.read_csv(path, nrows=5)  # only read a few rows; we just need headers
    cols = list(df.columns)

    print(f"Number of columns: {len(cols)}\n")
    print("Columns (in order):")
    for i, c in enumerate(cols, 1):
        print(f"{i:3d}. {c}")

    print("\nPreview (first 3 rows):")
    print(df.head(3).to_string(index=False))

    print("\nDtypes inferred from preview:")
    print(df.dtypes.to_string())

if __name__ == "__main__":
    print_schema(WEATHER_PATH, "ON_weather_daily_merged_2020-2025_clean")
    print_schema(PM25_DAILY_PATH, "pm25_daily_cleaned")
    print_schema(JOINED_TABLE_PATH, "pm25_weather_daily_region_joined")
