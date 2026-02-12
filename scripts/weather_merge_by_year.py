# scripts/weather_merge_by_year.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def parse_years(years: str) -> list[int]:
    """Accept: '2020-2025' or '2020,2021' or '2020'."""
    s = years.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [int(s)]


def read_station_ids(ids_file: Path) -> list[str]:
    if not ids_file.exists():
        raise FileNotFoundError(f"Station id list not found: {ids_file}")
    station_ids: list[str] = []
    for line in ids_file.read_text(encoding="utf-8").splitlines():
        sid = line.strip()
        if sid:
            station_ids.append(sid)
    if not station_ids:
        raise ValueError(f"No station ids found in {ids_file}")
    return station_ids


def station_year_path(base_dir: Path, station_id: str, year: int, pattern: str) -> Path:
    rel = pattern.format(id=station_id, year=year)
    return base_dir / rel


def iter_existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists() and p.is_file()]


def merge_one_year(
    base_dir: Path,
    station_ids: list[str],
    year: int,
    pattern: str,
    add_cols: bool,
) -> tuple[Optional[pd.DataFrame], dict]:
    paths = [station_year_path(base_dir, sid, year, pattern) for sid in station_ids]
    existing = iter_existing(paths)

    info = {
        "year": year,
        "stations_total": len(station_ids),
        "stations_found": len(existing),
        "stations_missing": len(station_ids) - len(existing),
        "rows_merged": 0,
    }

    if not existing:
        return None, info

    dfs: list[pd.DataFrame] = []
    for p in existing:
        df = pd.read_csv(p, low_memory=False)
        if add_cols:
            sid = p.parent.name
            if sid.startswith("station_"):
                sid = sid.replace("station_", "", 1)
            else:
                sid = "unknown"
            df["station_id"] = sid
            df["year"] = year
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    info["rows_merged"] = int(len(merged))
    return merged, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Ontario daily weather CSVs by year (workflow-ready).")
    ap.add_argument("--base-dir", type=Path, required=True, help="Base directory containing station folders/files.")
    ap.add_argument("--ids-file", type=Path, required=True, help="Text file with one station id per line.")
    ap.add_argument("--years", type=str, default="2020-2025", help='Years: "2020-2025" or "2020,2021".')
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (within workflow workspace).")
    ap.add_argument("--out-all", type=Path, default=None, help="Optional: write a merged-all-years CSV to this path.")
    ap.add_argument(
        "--pattern",
        type=str,
        default="station_{id}/station_{id}_daily_{year}.csv",
        help="Relative path pattern under base-dir. Use {id} and {year}.",
    )
    ap.add_argument("--no-extra-cols", action="store_true", help="Do NOT add station_id/year columns.")
    args = ap.parse_args()

    base_dir: Path = args.base_dir
    ids_file: Path = args.ids_file
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    years = parse_years(args.years)
    station_ids = read_station_ids(ids_file)

    print(f"[INFO] base_dir={base_dir}")
    print(f"[INFO] ids_file={ids_file} (n_station_ids={len(station_ids)})")
    print(f"[INFO] years={years}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] pattern={args.pattern}")
    print(f"[INFO] add_extra_cols={not args.no_extra_cols}")

    merged_year_dfs: list[pd.DataFrame] = []
    yearly_infos: list[dict] = []

    for y in years:
        merged, info = merge_one_year(
            base_dir=base_dir,
            station_ids=station_ids,
            year=y,
            pattern=args.pattern,
            add_cols=(not args.no_extra_cols),
        )
        yearly_infos.append(info)

        if merged is None:
            print(f"[{y}] No files found. (missing={info['stations_missing']})")
            continue

        out_path = out_dir / f"ON_weather_daily_merged_{y}.csv"
        merged.to_csv(out_path, index=False)
        merged_year_dfs.append(merged)

        print(
            f"[{y}] rows={info['rows_merged']:,} | stations_found={info['stations_found']:,} "
            f"| missing={info['stations_missing']:,} | saved -> {out_path}"
        )

    if args.out_all is not None:
        if merged_year_dfs:
            all_df = pd.concat(merged_year_dfs, ignore_index=True)
            args.out_all.parent.mkdir(parents=True, exist_ok=True)
            all_df.to_csv(args.out_all, index=False)
            print(f"[ALL] rows={len(all_df):,} saved -> {args.out_all}")
        else:
            raise RuntimeError("No yearly merges succeeded, cannot write --out-all.")

    log_path = out_dir / "merge_log.json"
    pd.DataFrame(yearly_infos).to_json(log_path, orient="records", indent=2)
    print(f"[OK] Wrote merge log -> {log_path}")


if __name__ == "__main__":
    main()
