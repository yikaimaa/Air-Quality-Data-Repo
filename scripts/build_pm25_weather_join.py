# scripts/build_pm25_weather_join.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm25", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pm = pd.read_csv(args.pm25)
    wx = pd.read_csv(args.weather)

    pm.columns = pm.columns.str.strip().str.lower()
    wx.columns = wx.columns.str.strip().str.lower()

    print("PM columns:", pm.columns.tolist())
    print("Weather columns:", wx.columns.tolist())

    if "region" not in pm.columns or "region" not in wx.columns:
        raise ValueError("Both inputs must contain 'region'")

    if "date" not in pm.columns or "date" not in wx.columns:
        raise ValueError("Both inputs must contain 'date'")

    merged = pm.merge(wx, on=["region", "date"], how="inner")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print("Wrote:", args.out)
    print("Rows:", len(merged))


if __name__ == "__main__":
    main()
