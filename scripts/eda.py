from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf


def save_plot(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out-html", dest="out_html", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])

    html_path = Path(args.out_html)
    out_dir = html_path.parent
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # BASIC INFO
    # ==========================================================
    n_rows = len(df)
    n_cols = len(df.columns)
    dup_total = df.duplicated().sum()
    dup_key = df.duplicated(subset=["region", "date"]).sum()
    overall_missing = df.isna().mean().mean()

    # ==========================================================
    # MISSING ANALYSIS
    # ==========================================================
    missing_rate = df.isna().mean().sort_values(ascending=False)
    row_missing_dist = df.isna().sum(axis=1).value_counts().sort_index()

    missing_region = df.groupby("region").apply(lambda x: x.isna().mean().mean())
    missing_date = df.groupby("date").apply(lambda x: x.isna().mean().mean())

    fig = plt.figure()
    missing_region.plot(kind="bar")
    plt.title("Missing Rate by Region")
    save_plot(fig, fig_dir / "missing_rate_region.png")

    fig = plt.figure()
    missing_date.plot()
    plt.title("Missing Rate Over Time")
    save_plot(fig, fig_dir / "missing_rate_time.png")

    # ==========================================================
    # TARGET BEHAVIOR
    # ==========================================================
    pm_col = "pm25_region_daily_avg"
    pm = df[pm_col]

    pm_stats = pm.describe()
    pm_var = pm.var()
    pm_skew = pm.skew()
    pm_kurt = pm.kurt()

    fig = plt.figure()
    pm.hist(bins=50)
    plt.title("PM25 Distribution")
    save_plot(fig, fig_dir / "pm25_distribution.png")

    ontario_avg = df.groupby("date")[pm_col].mean()

    fig = plt.figure()
    ontario_avg.plot()
    plt.title("Ontario Average PM25 Over Time")
    save_plot(fig, fig_dir / "pm25_trend.png")

    adf_stat, adf_p = adfuller(ontario_avg.dropna())[0:2]

    region_mean = df.groupby("region")[pm_col].mean().sort_values()

    fig = plt.figure(figsize=(10, 5))
    region_mean.plot(kind="bar")
    plt.title("Average PM25 by Region")
    save_plot(fig, fig_dir / "pm25_region_mean.png")

    overall_var = pm.var()
    between_var = region_mean.var()
    within_var = df.groupby("region")[pm_col].var().mean()

    threshold = 35
    extreme_prob = (pm > threshold).mean()

    # ==========================================================
    # AUTOCORRELATION
    # ==========================================================
    lag_corr = [ontario_avg.autocorr(lag=i) for i in range(1, 15)]
    lag_corr_df = pd.DataFrame({
        "lag": range(1, 15),
        "correlation": lag_corr
    })

    acf_vals = acf(ontario_avg.dropna(), nlags=20)
    acf_df = pd.DataFrame({
        "lag": range(len(acf_vals)),
        "acf_value": acf_vals
    })

    fig = plt.figure()
    plt.stem(acf_vals)
    plt.title("ACF (Ontario Avg)")
    save_plot(fig, fig_dir / "acf.png")

    # ==========================================================
    # CORRELATION HEATMAP
    # ==========================================================
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_plot(fig, fig_dir / "correlation_heatmap.png")

    # ==========================================================
    # WORKFLOW PRINT
    # ==========================================================
    print("\n===== EDA SUMMARY =====")
    print("Rows:", n_rows)
    print("Duplicates:", dup_total)
    print("Overall missing rate:", round(overall_missing, 4))
    print("ADF p-value:", round(adf_p, 6))
    print("Lag1 correlation:", round(lag_corr[0], 4))
    print("=======================\n")

    # ==========================================================
    # HTML
    # ==========================================================
    html = f"""
    <html>
    <body>
    <h1>EDA Summary</h1>

    <h2>Basic Info</h2>
    <p>Rows: {n_rows}</p>
    <p>Columns: {n_cols}</p>
    <p>Total duplicates: {dup_total}</p>

    <h2>Missing Rate by Column</h2>
    {missing_rate.to_frame("missing_rate").to_html()}

    <h2>Row-wise Missing Distribution</h2>
    {row_missing_dist.to_frame("count").to_html()}

    <h2>Missing by Region</h2>
    {missing_region.to_frame("missing_rate").to_html()}
    <img src="figures/missing_rate_region.png">

    <h2>Missing over Time</h2>
    {missing_date.head(30).to_frame("missing_rate (first 30 days)").to_html()}
    <img src="figures/missing_rate_time.png">

    <h2>PM25 Distribution</h2>
    {pm_stats.to_frame().to_html()}
    <p>Variance: {pm_var}</p>
    <p>Skewness: {pm_skew}</p>
    <p>Kurtosis: {pm_kurt}</p>
    <img src="figures/pm25_distribution.png">

    <h2>Ontario Trend</h2>
    {ontario_avg.tail(30).to_frame("pm25_avg_last_30_days").to_html()}
    <img src="figures/pm25_trend.png">

    <h2>PM25 by Region</h2>
    {region_mean.to_frame("mean_pm25").to_html()}
    <img src="figures/pm25_region_mean.png">

    <h2>Stationarity (ADF)</h2>
    <p>ADF Statistic: {adf_stat}</p>
    <p>p-value: {adf_p}</p>

    <h2>Variance Decomposition</h2>
    <p>Overall: {overall_var}</p>
    <p>Between: {between_var}</p>
    <p>Within: {within_var}</p>

    <h2>Autocorrelation</h2>
    {lag_corr_df.to_html(index=False)}
    {acf_df.to_html(index=False)}
    <img src="figures/acf.png">

    <h2>Correlation Matrix</h2>
    {corr.to_html()}
    <img src="figures/correlation_heatmap.png">

    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print("EDA completed. HTML + PNG saved to:", html_path.parent)


if __name__ == "__main__":
    main()
