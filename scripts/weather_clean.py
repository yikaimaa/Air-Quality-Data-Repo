import argparse
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd


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
    """Keep the last occurrence if duplicate column names exist (parquet-safe)."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df


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


def zip_single_file(src: Path, zip_path: Path) -> None:
    """Create a zip containing only the file (no parent directories)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=src.name)


# -----------------------------
# 2) Main cleaning pipeline
# -----------------------------
def clean_weather_daily(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    output_missing_csv: Optional[Union[str, Path]] = None,
    drop_flag_columns: bool = True,
) -> Dict:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_missing_csv is not None:
        output_missing_csv = Path(output_missing_csv)
        output_missing_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)

    # Standardize column names
    df = df.rename(columns={c: to_snake_case(c) for c in df.columns})
    df = dedupe_columns(df)

    # Clean station_name / climate_id
    if "station_name" in df.columns:
        df["station_name"] = df["station_name"].astype("string").str.strip()

    if "climate_id" in df.columns:
        df["climate_id"] = df["climate_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

    # Date fields
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
        df["date"] = df["date_time"].dt.date
        df = df.assign(
            year=df["date_time"].dt.year,
            month=df["date_time"].dt.month,
            day=df["date_time"].dt.day,
        )

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

    # Gust direction conversion
    if "dir_of_max_gust_10s_deg" in df.columns:
        df["dir_of_max_gust_deg"] = (df["dir_of_max_gust_10s_deg"] * 10.0) % 360

    # Flag -> indicators
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

    # data_quality marker
    if "data_quality" in df.columns:
        df["data_quality_has_mark"] = df["data_quality"].notna().astype("int8")

    # Range checks
    df = apply_range_check(df, "max_temp_degc", lo=-60, hi=50)
    df = apply_range_check(df, "min_temp_degc", lo=-60, hi=50)
    df = apply_range_check(df, "mean_temp_degc", lo=-60, hi=50)

    df = apply_range_check(df, "total_rain_mm", lo=0, hi=500)
    df = apply_range_check(df, "total_precip_mm", lo=0, hi=500)
    df = apply_range_check(df, "total_snow_cm", lo=0, hi=200)
    df = apply_range_check(df, "snow_on_grnd_cm", lo=0, hi=300)

    df = apply_range_check(df, "spd_of_max_gust_km_h", lo=0, hi=250)
    df = apply_range_check(df, "dir_of_max_gust_deg", lo=0, hi=359.999)

    # Deduplicate by station_id + date
    deduped = 0
    if "station_id" in df.columns and "date" in df.columns:
        before = len(df)
        df = df.sort_values(["station_id", "date"])
        df = df.drop_duplicates(subset=["station_id", "date"], keep="first")
        deduped = before - len(df)

    # Drop raw flag columns
    if drop_flag_columns:
        df = df.drop(columns=[c for c in df.columns if c.endswith("_flag")], errors="ignore")

    df = dedupe_columns(df)

    # Station-level missing summary (optional output)
    missing_summary = None
    if output_missing_csv is not None:
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

        missing_summary.to_csv(output_missing_csv, index=False)

    # Write clean csv
    df.to_csv(output_csv, index=False)

    summary = {
        "input_rows": int(pd.read_csv(input_csv, low_memory=False).shape[0]),
        "output_rows": int(df.shape[0]),
        "deduped_rows_station_date": int(deduped),
        "n_cols_clean": int(df.shape[1]),
        "output_clean_csv": str(output_csv),
        "output_missing_csv": str(output_missing_csv) if output_missing_csv is not None else None,
        "n_stations": int(missing_summary["station_id"].nunique()) if missing_summary is not None else None,
    }
    return summary


# -----------------------------
# 3) CLI entry point (workflow-ready)
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean Ontario daily weather dataset (station-day)")

    p.add_argument("--in", dest="input_csv", required=True, help="Input CSV path")
    p.add_argument("--out", dest="output_csv", required=True, help="Output clean CSV path")

    p.add_argument(
        "--out-missing",
        dest="output_missing_csv",
        default=None,
        help="(Optional) Output station-level missing rate summary CSV path",
    )

    p.add_argument(
        "--out-zip",
        dest="output_zip",
        default=None,
        help="(Optional) Also write a .zip containing only the clean CSV",
    )

    p.add_argument(
        "--keep-flag-columns",
        action="store_true",
        help="Keep original *_flag columns (default drops them)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    summary = clean_weather_daily(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_missing_csv=args.output_missing_csv,
        drop_flag_columns=not args.keep_flag_columns,
    )

    # If requested, zip the clean CSV
    if args.output_zip:
        zip_single_file(Path(args.output_csv), Path(args.output_zip))

    print("✅ Weather cleaning done. Summary:")
    for k, v in summary.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
