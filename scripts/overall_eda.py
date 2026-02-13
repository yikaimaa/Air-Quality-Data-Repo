from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
import seaborn as sns
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out-html", dest="out_html", required=True)
    parser.add_argument("--out-dir", dest="out_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("EDA STARTED")
    print("==============================")

    # ==========================================================
    # 1️⃣ Missing Profiling
    # ==========================================================

    print("\n===== MISSING RATE BY COLUMN =====")
    missing_rate = df.isna().mean().sort_values(ascending=False)
    print(missing_rate)

    print("\n===== ROW-WISE MISSING COUNT DISTRIBUTION =====")
    row_missing = df.isna().sum(axis=1)
    print(row_missing.value_counts().sort_index())

    print("\n===== MISSING RATE OVER REGION =====")
    missing_region = df.groupby("region").apply(lambda x: x.isna().mean().mean())
    print(missing_region)

    print("\n===== MISSING RATE OVER DATE =====")
    missing_date = df.groupby("date").apply(lambda x: x.isna().mean().mean())
    print(missing_date.head())

    print("\n===== DUPLICATE CHECK =====")
    print("Total duplicate rows:", df.duplicated().sum())
    print("Duplicate (region,date):", df.duplicated(subset=["region","date"]).sum())

    print("\n===== ALL-ZERO COLUMN CHECK =====")
    numeric_df = df.select_dtypes(include="number")
    zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()
    print("All-zero columns:", zero_cols if zero_cols else "None")

    # Date completeness
    print("\n===== DATE COMPLETENESS CHECK =====")
    start = datetime(2020,1,1)
    end = datetime(2024,12,31)
    full_range = pd.date_range(start, end, freq="D")

    region_date_missing = {}
    for region in df["region"].unique():
        sub = df[df["region"] == region]
        region_date_missing[region] = len(set(full_range) - set(sub["date"]))
    print(region_date_missing)

    # --- Save Missing Plots ---
    plt.figure()
    missing_region.plot(kind="bar")
    plt.title("Missing Rate by Region")
    plt.ylabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(fig_dir/"missing_rate_region.png", dpi=300)
    plt.close()

    plt.figure()
    missing_date.plot()
    plt.title("Missing Rate over Time")
    plt.ylabel("Missing Rate")
    plt.tight_layout()
    plt.savefig(fig_dir/"missing_rate_time.png", dpi=300)
    plt.close()

    # ==========================================================
    # 2️⃣ Target Behavior Analysis
    # ==========================================================

    pm_col = "pm25_region_daily_avg"
    pm = df[pm_col]

    print("\n===== PM25 BASIC STATS =====")
    print(pm.describe())
    print("Variance:", pm.var())
    print("Skewness:", pm.skew())
    print("Kurtosis:", pm.kurt())

    # Distribution
    plt.figure()
    pm.hist(bins=50)
    plt.title("PM25 Distribution")
    plt.xlabel("PM25")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir/"pm25_distribution.png", dpi=300)
    plt.close()

    # Ontario average trend
    ontario_avg = df.groupby("date")[pm_col].mean()

    plt.figure()
    ontario_avg.plot()
    plt.title("Ontario Average PM25 Over Time")
    plt.ylabel("PM25")
    plt.tight_layout()
    plt.savefig(fig_dir/"pm25_trend.png", dpi=300)
    plt.close()

    # ADF Test
    adf_result = adfuller(ontario_avg.dropna())
    print("\n===== ADF TEST (Ontario Avg) =====")
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])

    # Region comparison
    region_mean = df.groupby("region")[pm_col].mean()

    plt.figure(figsize=(10,5))
    region_mean.sort_values().plot(kind="bar")
    plt.title("Average PM25 by Region")
    plt.tight_layout()
    plt.savefig(fig_dir/"pm25_region_mean.png", dpi=300)
    plt.close()

    # Variance decomposition
    overall_var = pm.var()
    between_var = region_mean.var()
    within_var = df.groupby("region")[pm_col].var().mean()

    print("\n===== VARIANCE DECOMPOSITION =====")
    print("Overall variance:", overall_var)
    print("Between-region variance:", between_var)
    print("Within-region variance:", within_var)

    # Heavy tail risk
    threshold = 35
    extreme_prob = (pm > threshold).mean()
    print("\n===== HEAVY TAIL RISK =====")
    print(f"P(PM25 > {threshold}) =", extreme_prob)

    # ==========================================================
    # 3️⃣ Autocorrelation
    # ==========================================================

    lag_corr = [ontario_avg.autocorr(lag=i) for i in range(1,15)]
    print("\n===== LAG 1–14 CORRELATION =====")
    print(lag_corr)

    acf_vals = acf(ontario_avg.dropna(), nlags=20)

    plt.figure()
    plt.stem(acf_vals)
    plt.title("ACF (Ontario Avg)")
    plt.tight_layout()
    plt.savefig(fig_dir/"acf.png", dpi=300)
    plt.close()

    # ==========================================================
    # 4️⃣ Correlation Heatmap
    # ==========================================================

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir/"correlation_heatmap.png", dpi=300)
    plt.close()

    # ==========================================================
    # HTML Summary
    # ==========================================================

    html = f"""
    <html>
    <body>
    <h1>EDA Summary</h1>
    <h2>PM25 Summary Stats</h2>
    {pm.describe().to_frame().to_html()}
    <h2>ADF Test</h2>
    <p>ADF Statistic: {adf_result[0]}</p>
    <p>p-value: {adf_result[1]}</p>
    <h2>Heavy Tail Risk</h2>
    <p>P(PM25 > {threshold}) = {extreme_prob}</p>
    <h2>Variance Decomposition</h2>
    <p>Overall: {overall_var}</p>
    <p>Between: {between_var}</p>
    <p>Within: {within_var}</p>
    </body>
    </html>
    """

    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_html, "w") as f:
        f.write(html)

    print("\nEDA COMPLETED")
    print("==============================")
    print("Figures saved to:", fig_dir)
    print("HTML saved to:", args.out_html)


if __name__ == "__main__":
    main()
