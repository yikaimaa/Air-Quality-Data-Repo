
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_html", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df.columns = df.columns.str.lower()

    print("===== BASIC INFO =====")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\n===== MISSING RATE BY COLUMN =====")
    missing_rate = df.isna().mean().sort_values(ascending=False)
    print(missing_rate)

    print("\n===== ROW-WISE MISSING COUNT DISTRIBUTION =====")
    row_missing = df.isna().sum(axis=1)
    row_missing_dist = row_missing.value_counts().sort_index()
    print(row_missing_dist)

    print("\n===== MISSING RATE OVER REGION =====")
    missing_region = df.groupby("region").apply(lambda x: x.isna().mean().mean())
    print(missing_region)

    print("\n===== MISSING RATE OVER DATE =====")
    missing_date = df.groupby("date").apply(lambda x: x.isna().mean().mean())
    print(missing_date.head())

    print("\n===== DUPLICATE CHECK =====")
    dup_count = df.duplicated().sum()
    dup_region_date = df.duplicated(subset=["region", "date"]).sum()
    print("Total duplicate rows:", dup_count)
    print("Duplicate (region,date):", dup_region_date)

    print("\n===== DATE COMPLETENESS CHECK =====")
    df["date"] = pd.to_datetime(df["date"])
    start = datetime(2020,1,1)
    end = datetime(2024,12,31)
    full_range = pd.date_range(start, end, freq="D")

    region_date_missing = {}
    for region in df["region"].unique():
        sub = df[df["region"] == region]
        existing = set(sub["date"])
        missing_dates = set(full_range) - existing
        region_date_missing[region] = len(missing_dates)

    print("Missing dates per region:")
    print(region_date_missing)

    fig1, ax1 = plt.subplots()
    missing_region.plot(kind="bar", ax=ax1)
    ax1.set_title("Missing Rate over Region")
    ax1.set_ylabel("Missing Rate")
    img_region = fig_to_base64(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    missing_date.plot(ax=ax2)
    ax2.set_title("Missing Rate over Time")
    ax2.set_ylabel("Missing Rate")
    img_time = fig_to_base64(fig2)
    plt.close(fig2)

    html = f"""
    <html>
    <head>
        <title>EDA Quality Report</title>
    </head>
    <body>
        <h1>EDA Quality Report</h1>

        <h2>Basic Info</h2>
        <p>Rows: {len(df)}</p>
        <p>Columns: {len(df.columns)}</p>

        <h2>Missing Rate by Column</h2>
        {missing_rate.to_frame("missing_rate").to_html()}

        <h2>Row Missing Count Distribution</h2>
        {row_missing_dist.to_frame("count").to_html()}

        <h2>Missing Rate over Region</h2>
        <img src="data:image/png;base64,{img_region}" />

        <h2>Missing Rate over Time</h2>
        <img src="data:image/png;base64,{img_time}" />

        <h2>Duplicate Summary</h2>
        <p>Total duplicate rows: {dup_count}</p>
        <p>Duplicate (region,date): {dup_region_date}</p>

        <h2>Date Completeness (2020-01-01 to 2024-12-31)</h2>
        <pre>{region_date_missing}</pre>
    </body>
    </html>
    """

    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_html, "w") as f:
        f.write(html)

    print("\nEDA HTML report written to:", args.out_html)

if __name__ == "__main__":
    main()
