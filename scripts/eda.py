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
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    overall_missing = df.isna().mean().mean()

    pm_col = "pm25_region_daily_avg"

    wildfire_cols = [
        "fire_count_50km_avg",
        "fire_count_100km_avg",
        "frp_sum_100km_avg",
        "min_fire_distance_km",
        "min_fire_distance_missing",
    ]
    wildfire_cols = [c for c in wildfire_cols if c in df.columns]

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

    wildfire_missing = None
    if wildfire_cols:
        wildfire_missing = df[wildfire_cols].isna().mean().sort_values(ascending=False)

        fig = plt.figure(figsize=(8, 4))
        wildfire_missing.plot(kind="bar")
        plt.title("Wildfire Feature Missing Rate")
        plt.ylabel("Missing Rate")
        save_plot(fig, fig_dir / "wildfire_missing_rate.png")

    # ==========================================================
    # TARGET BEHAVIOR
    # ==========================================================
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

    # ==========================================================
    # WILDFIRE FEATURE DISTRIBUTIONS
    # Small multiple plots
    # ==========================================================
    wildfire_stats_html = "<p>No wildfire columns found.</p>"
    if wildfire_cols:
        wildfire_stats = df[wildfire_cols].describe().T
        wildfire_stats_html = wildfire_stats.to_html()

        n_feat = len(wildfire_cols)
        ncols = 2
        nrows = int(np.ceil(n_feat / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, wildfire_cols):
            series = df[col].dropna()
            ax.hist(series, bins=40)
            ax.set_title(col)

        for ax in axes[len(wildfire_cols):]:
            ax.axis("off")

        save_plot(fig, fig_dir / "wildfire_feature_distributions.png")

    # ==========================================================
    # WILDFIRE TIME TREND VS PM2.5
    # ==========================================================
    wildfire_time_html = "<p>No wildfire columns found.</p>"
    if wildfire_cols:
        wildfire_time_cols = [c for c in ["fire_count_100km_avg", "frp_sum_100km_avg"] if c in df.columns]
        if wildfire_time_cols:
            ontario_wildfire = df.groupby("date")[wildfire_time_cols].mean()
            wildfire_time_html = ontario_wildfire.tail(30).to_html()

            fig, axes = plt.subplots(len(wildfire_time_cols) + 1, 1, figsize=(10, 4 * (len(wildfire_time_cols) + 1)))

            axes[0].plot(ontario_avg.index, ontario_avg.values)
            axes[0].set_title("Ontario Average PM25 Over Time")

            for i, col in enumerate(wildfire_time_cols, start=1):
                axes[i].plot(ontario_wildfire.index, ontario_wildfire[col].values)
                axes[i].set_title(f"Ontario Average {col} Over Time")

            save_plot(fig, fig_dir / "wildfire_pm25_time_trend.png")

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
    # EXTREME PM2.5 DAYS VS NORMAL DAYS
    # ==========================================================
    extreme_days_html = "<p>No wildfire columns found.</p>"
    if wildfire_cols:
        threshold = df[pm_col].quantile(0.9)
        df["pm25_extreme_flag"] = (df[pm_col] >= threshold).astype(int)

        compare_cols = [c for c in ["fire_count_100km_avg", "frp_sum_100km_avg", "min_fire_distance_km"] if c in df.columns]
        if compare_cols:
            extreme_summary = df.groupby("pm25_extreme_flag")[compare_cols].mean().T
            extreme_days_html = extreme_summary.to_html()

            fig, axes = plt.subplots(1, len(compare_cols), figsize=(5 * len(compare_cols), 4))
            if len(compare_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, compare_cols):
                sns.boxplot(data=df, x="pm25_extreme_flag", y=col, ax=ax)
                ax.set_title(col)
                ax.set_xlabel("Extreme PM25 Day (0=No, 1=Yes)")

            save_plot(fig, fig_dir / "wildfire_extreme_pm25_comparison.png")

    # ==========================================================
    # SIMPLIFIED CORRELATION ANALYSIS
    # Corr(X_t , PM_{t+1})
    # ==========================================================
    df["pm25_next_day"] = (
        df.groupby("region")[pm_col]
        .shift(-1)
    )

    df_corr = df.dropna(subset=["pm25_next_day"]).copy()

    numeric_df = df_corr.select_dtypes(include="number")

    # Remove all-zero columns except wildfire missing indicator if present
    zero_cols = numeric_df.columns[(numeric_df == 0).all()].tolist()
    zero_cols = [c for c in zero_cols if c != "min_fire_distance_missing"]

    if zero_cols:
        print("Removing all-zero columns from correlation:")
        print(zero_cols)

    numeric_df = numeric_df.drop(columns=zero_cols, errors="ignore")

    if "pm25_next_day" not in numeric_df.columns:
        numeric_df["pm25_next_day"] = df_corr["pm25_next_day"]

    corr_matrix = numeric_df.corr()
    target_corr = corr_matrix["pm25_next_day"].sort_values(ascending=False)
    target_corr = target_corr.drop("pm25_next_day")

    corr_table = target_corr.to_frame("corr_with_next_day_pm25")

    fig = plt.figure(figsize=(8, 10))
    target_corr.sort_values().plot(kind="barh")
    plt.title("Correlation with Next-Day PM25")
    plt.xlabel("Correlation")
    save_plot(fig, fig_dir / "correlation_next_day_pm25.png")

    # wildfire-only correlation table
    wildfire_corr_html = "<p>No wildfire columns found.</p>"
    wildfire_corr_cols = [c for c in wildfire_cols if c in target_corr.index]
    if wildfire_corr_cols:
        wildfire_corr_html = target_corr.loc[wildfire_corr_cols].to_frame("corr_with_next_day_pm25").to_html()

    # ==========================================================
    # WORKFLOW PRINT (concise)
    # ==========================================================
    print("\n===== EDA SUMMARY =====")
    print("Rows:", n_rows)
    print("Duplicates:", dup_total)
    print("Overall missing rate:", round(overall_missing, 4))
    print("ADF p-value:", round(adf_p, 6))
    print("Top correlation:", round(target_corr.iloc[0], 4))
    if wildfire_cols:
        print("Wildfire columns:", wildfire_cols)
    print("=======================\n")

    # ==========================================================
    # HTML REPORT
    # ==========================================================
    html = f"""
    <html>
    <body>
    <h1>EDA Summary</h1>

    <h2>Basic Info</h2>
    <p>Rows: {n_rows}</p>
    <p>Columns: {n_cols}</p>
    <p>Duplicates: {dup_total}</p>
    <p>Overall missing rate: {overall_missing:.4f}</p>

    <h2>Missing Analysis</h2>
    {missing_rate.to_frame("missing_rate").to_html()}
    {row_missing_dist.to_frame("count").to_html()}
    {missing_region.to_frame("missing_rate").to_html()}
    <img src="figures/missing_rate_region.png">
    <img src="figures/missing_rate_time.png">

    <h3>Wildfire Missing Analysis</h3>
    {wildfire_missing.to_frame("missing_rate").to_html() if wildfire_missing is not None else "<p>No wildfire columns found.</p>"}
    <img src="figures/wildfire_missing_rate.png">

    <h2>PM25 Distribution</h2>
    {pm_stats.to_frame().to_html()}
    <p>Variance: {pm_var}</p>
    <p>Skewness: {pm_skew}</p>
    <p>Kurtosis: {pm_kurt}</p>
    <img src="figures/pm25_distribution.png">

    <h2>Ontario Trend</h2>
    {ontario_avg.tail(30).to_frame("pm25_last_30_days").to_html()}
    <img src="figures/pm25_trend.png">

    <h2>PM25 by Region</h2>
    {region_mean.to_frame("mean_pm25").to_html()}
    <img src="figures/pm25_region_mean.png">

    <h2>Wildfire Feature Distributions</h2>
    {wildfire_stats_html}
    <img src="figures/wildfire_feature_distributions.png">

    <h2>Wildfire Time Trend vs PM25</h2>
    {wildfire_time_html}
    <img src="figures/wildfire_pm25_time_trend.png">

    <h2>Extreme PM25 Days vs Normal Days</h2>
    {extreme_days_html}
    <img src="figures/wildfire_extreme_pm25_comparison.png">

    <h2>Stationarity (ADF)</h2>
    <p>ADF Statistic: {adf_stat}</p>
    <p>p-value: {adf_p}</p>

    <h2>Autocorrelation</h2>
    {lag_corr_df.to_html(index=False)}
    {acf_df.to_html(index=False)}
    <img src="figures/acf.png">

    <h2>Correlation with Next-Day PM25</h2>
    {corr_table.to_html()}
    <img src="figures/correlation_next_day_pm25.png">

    <h3>Wildfire Correlation with Next-Day PM25</h3>
    {wildfire_corr_html}

    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)

    print("EDA completed. HTML + PNG saved to:", html_path.parent)


if __name__ == "__main__":
    main()
