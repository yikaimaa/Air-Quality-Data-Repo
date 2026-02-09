import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Union


# -----------------------------
# 1) Utility functions
# -----------------------------
def to_snake_case(s: str) -> str:
    s = s.strip()
    s = s.replace("°", "deg")
    s = s.replace("/", "_")
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def flag_to_indicators(df: pd.DataFrame, value_col: str, flag_col: str) -> pd.DataFrame:
    if flag_col not in df.columns or value_col not in df.columns:
        return df

    f = df[flag_col].astype("object").fillna("")

    df[f"{value_col}_is_trace"] = (f == "T").astype("int8")
    df[f"{value_col}_is_estimated"] = (f == "E").astype("int8")
    df[f"{value_col}_is_missingflag"] = (f == "M").astype("int8")
    df[f"{value_col}_is_lowquality"] = (f == "L").astype("int8")

    # For precipitation-related variables, treat trace ("T") as 0.0
    if any(k in value_col for k in ["total_rain", "total_snow", "total_precip", "snow_on_grnd"]):
        df.loc[df[f"{value_col}_is_trace"] == 1, value_col] = 0.0

    return df


def apply_range_check(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.DataFrame:
    if col not in df.columns:
        return df
    x = df[col]
    is_out = (x.notna()) & ((x < lo) | (x > hi))
    df[f"{col}_is_outlier"] = is_out.astype("int8")
    df.loc[is_out, col] = np.nan
    return df


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicate column names:
    - If duplicate names exist (e.g., repeated 'year'), keep the *last* occurrence
      (often the later computed / more reliable one).
    - This is a robust approach that minimizes accidental data loss.
    """
    if df.columns.duplicated().any():
        # Drop columns where duplicated(keep="last") is True, i.e., keep the last duplicate
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


def safe_to_parquet(df: pd.DataFrame, path: Path) -> str:
    """
    Safely write parquet: return the error message if it fails (do not raise).
    """
    try:
        df.to_parquet(path, index=False)
        return ""
    except Exception as e:
        return str(e)


def compute_station_missing_rates(
    df: pd.DataFrame,
    station_col: str,
    cols_to_check: List[str],
    extra_id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if station_col not in df.columns:
        raise ValueError(f"Missing station column: {station_col}")

    cols_to_check = [c for c in cols_to_check if c in df.columns]
    if not cols_to_check:
        raise ValueError("No valid columns to check for missing rate.")

    grp = df.groupby(station_col, dropna=False)
    n_rows = grp.size().rename("n_rows")

    miss_rate = grp[cols_to_check].apply(lambda x: x.isna().mean()).astype(float)
    out = miss_rate.join(n_rows, how="left")

    if extra_id_cols:
        for c in [c for c in extra_id_cols if c in df.columns]:
            s = grp[c].apply(lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else pd.NA).rename(c)
            out = out.join(s, how="left")

    out = out.reset_index()

    front = [station_col]
    if extra_id_cols:
        front += [c for c in extra_id_cols if c in out.columns]
    front += ["n_rows"]
    rate_cols = [c for c in cols_to_check if c in out.columns]
    out = out[front + rate_cols]

    out = out.sort_values(["n_rows", station_col], ascending=[False, True]).reset_index(drop=True)
    return out


# -----------------------------
# 2) Main cleaning pipeline
# -----------------------------
def clean_weather_daily(
    input_csv: Union[str, Path],
    output_dir: Union[str, Path],
    drop_flag_columns: bool = True,
) -> Dict:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)

    # Standardize column names
    rename_map = {c: to_snake_case(c) for c in df.columns}
    df = df.rename(columns=rename_map)

    # Dedupe once early (prevents Year/year collisions turning into duplicated columns)
    df = dedupe_columns(df)

    # Clean station_name / climate_id
    if "station_name" in df.columns:
        df["station_name"] = df["station_name"].astype("string").str.strip()

    if "climate_id" in df.columns:
        df["climate_id"] = (
            df["climate_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        )

    # Date fields: if date_time exists, use it to rebuild year/month/day
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df["date"] = df["date_time"].dt.date

        # Use assign to avoid weird duplicate-column behavior
        df = df.assign(
            year=df["date_time"].dt.year,
            month=df["date_time"].dt.month,
            day=df["date_time"].dt.day,
        )

    # Dedupe again to ensure 'year' is unique
    df = dedupe_columns(df)

    # Coerce numeric columns
    numeric_cols = [
        "longitude_x", "latitude_y",
        "max_temp_degc", "min_temp_degc", "mean_temp_degc",
        "heat_deg_days_degc", "cool_deg_days_degc",
        "total_rain_mm", "total_snow_cm", "total_precip_mm",
        "snow_on_grnd_cm",
        "dir_of_max_gust_10s_deg", "spd_of_max_gust_km_h",
        "station_id",
    ]
    df = coerce_numeric(df, numeric_cols)

    # Convert max gust direction from 10s-of-degrees to degrees
    if "dir_of_max_gust_10s_deg" in df.columns:
        df["dir_of_max_gust_deg"] = (df["dir_of_max_gust_10s_deg"] * 10.0) % 360

    # Convert flag columns into indicator variables
    flag_pairs = [
        ("max_temp_degc", "max_temp_flag"),
        ("min_temp_degc", "min_temp_flag"),
        ("mean_temp_degc", "mean_temp_flag"),
        ("heat_deg_days_degc", "heat_deg_days_flag"),
        ("cool_deg_days_degc", "cool_deg_days_flag"),
        ("total_rain_mm", "total_rain_flag"),
        ("total_snow_cm", "total_snow_flag"),
        ("total_precip_mm", "total_precip_flag"),
        ("snow_on_grnd_cm", "snow_on_grnd_flag"),
        ("dir_of_max_gust_10s_deg", "dir_of_max_gust_flag"),
        ("spd_of_max_gust_km_h", "spd_of_max_gust_flag"),
    ]
    for val_col, flg_col in flag_pairs:
        df = flag_to_indicators(df, val_col, flg_col)

    # Simplify data_quality into a binary marker
    if "data_quality" in df.columns:
        df["data_quality_has_mark"] = df["data_quality"].notna().astype("int8")

    # Physical range checks (outliers -> NaN + outlier flag)
    df = apply_range_check(df, "max_temp_degc", lo=-60, hi=50)
    df = apply_range_check(df, "min_temp_degc", lo=-60, hi=50)
    df = apply_range_check(df, "mean_temp_degc", lo=-60, hi=50)

    df = apply_range_check(df, "total_rain_mm", lo=0, hi=500)
    df = apply_range_check(df, "total_precip_mm", lo=0, hi=500)
    df = apply_range_check(df, "total_snow_cm", lo=0, hi=200)
    df = apply_range_check(df, "snow_on_grnd_cm", lo=0, hi=300)

    df = apply_range_check(df, "spd_of_max_gust_km_h", lo=0, hi=250)
    df = apply_range_check(df, "dir_of_max_gust_deg", lo=0, hi=359.999)

    # Deduplicate by primary key
    deduped = 0
    if "station_id" in df.columns and "date" in df.columns:
        before = len(df)
        df = df.sort_values(["station_id", "date"])
        df = df.drop_duplicates(subset=["station_id", "date"], keep="first")
        deduped = before - len(df)

    # Drop original flag columns if requested
    if drop_flag_columns:
        df = df.drop(columns=[c for c in df.columns if c.endswith("_flag")], errors="ignore")

    # Final dedupe of columns (parquet safety)
    df = dedupe_columns(df)

    # -----------------------------
    # Station-level missing rate summary
    # -----------------------------
    cols_for_missing = [
        "date_time",
        "longitude_x", "latitude_y",
        "max_temp_degc", "min_temp_degc", "mean_temp_degc",
        "heat_deg_days_degc", "cool_deg_days_degc",
        "total_rain_mm", "total_snow_cm", "total_precip_mm",
        "snow_on_grnd_cm",
        "dir_of_max_gust_deg", "spd_of_max_gust_km_h",
    ]
    extra_id_cols = ["station_name", "climate_id"]

    missing_summary = compute_station_missing_rates(
        df=df,
        station_col="station_id",
        cols_to_check=cols_for_missing,
        extra_id_cols=extra_id_cols,
    )

    # Outputs
    out_clean_csv = output_dir / "ON_weather_daily_merged_2020—2025_clean.csv"
    out_clean_parquet = output_dir / "ON_weather_daily_merged_2020—2025_clean.parquet"
    out_miss_csv = output_dir / "station_missing_rate_summary.csv"
    out_miss_parquet = output_dir / "station_missing_rate_summary.parquet"

    df.to_csv(out_clean_csv, index=False)
    missing_summary.to_csv(out_miss_csv, index=False)

    # Parquet write: do not abort on failure
    err1 = safe_to_parquet(df, out_clean_parquet)
    err2 = safe_to_parquet(missing_summary, out_miss_parquet)

    summary = {
        "output_rows": int(df.shape[0]),
        "deduped_rows": int(deduped),
        "n_cols_clean": int(df.shape[1]),
        "n_stations": int(missing_summary["station_id"].nunique()) if "station_id" in missing_summary.columns else None,
        "output_clean_csv": str(out_clean_csv),
        "output_missing_csv": str(out_miss_csv),
        "output_clean_parquet": str(out_clean_parquet),
        "output_missing_parquet": str(out_miss_parquet),
        "parquet_clean_error": err1 if err1 else None,
        "parquet_missing_error": err2 if err2 else None,
    }
    return summary


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    INPUT = "/Users/zhangrilong/Downloads/clean data/ON_weather_daily_merged_2020_2025.csv"  # change to your file path
    OUTPUT_DIR = "cleaned_weather"  # change to your desired output directory

    summary = clean_weather_daily(
        input_csv=INPUT,
        output_dir=OUTPUT_DIR,
        drop_flag_columns=True,
    )

    print("✅ Cleaning done. Summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")
