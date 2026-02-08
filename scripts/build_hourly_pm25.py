from __future__ import annotations

import argparse
import json
from pathlib import Path

from pm25.ingest import load_years
from pm25.stations import build_station_lookup, attach_station_meta
from pm25.dedupe import drop_fully_duplicated_rows, merge_key_duplicates


DEFAULT_BASE_RAW = "https://raw.githubusercontent.com/yikaimaa/Air-Quality-Data-Repo/main/Datasets/Ontario/PM25/"
DEFAULT_FILES = [
    "ON_PM25_2020-01-01_2020-12-31.csv",
    "ON_PM25_2021-01-01_2021-12-31.csv",
    "ON_PM25_2022-01-01_2022-12-31.csv",
    "ON_PM25_2023-01-01_2023-12-31.csv",
    "ON_PM25_2024-01-01_2024-12-31.csv",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-raw", default=DEFAULT_BASE_RAW)
    p.add_argument("--out", required=True, help="Output parquet/csv path for cleaned hourly dataset")
    p.add_argument("--report-out", default=None, help="Optional JSON report path")
    args = p.parse_args()

    df = load_years(args.base_raw, DEFAULT_FILES)
    station_lookup = build_station_lookup(args.base_raw, DEFAULT_FILES)
    df = attach_station_meta(df, station_lookup)

    df = drop_fully_duplicated_rows(df)
    df, report = merge_key_duplicates(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in [".parquet", ".pq"]:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
