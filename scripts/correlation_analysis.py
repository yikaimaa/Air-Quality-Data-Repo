import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


INPUT_PATH = "../Datasets/pm25_weather_daily_region_joined.csv"
OUTPUT_DIR = "../Datasets/correlation_results"



TARGET_COL = "pm25_region_daily_avg"
DATE_COL = "date"

FEATURE_COLS = [
    "max_temp_degc",
    "min_temp_degc",
    "mean_temp_degc",
    "heat_deg_days_degc",
    "cool_deg_days_degc",
    "total_rain_mm",
    "total_snow_cm",
    "total_precip_mm",
    "snow_on_grnd_cm",
    "spd_of_max_gust_km_h",
    "dir_of_max_gust_deg",
    "n_weather_stations"
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

 
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # -----------------------------
    # remove missing
    # -----------------------------
    cols = [TARGET_COL] + FEATURE_COLS
    df_corr = df[cols].copy()
    df_corr = df_corr.dropna()
    print(f"Data shape after dropna: {df_corr.shape}")

    # -----------------------------
    # 1️⃣ Pearson correlation
    # -----------------------------
    pearson_corr = df_corr.corr(method="pearson")
    pearson_target = pearson_corr[[TARGET_COL]].sort_values(by=TARGET_COL, ascending=False)
    pearson_target.to_csv(os.path.join(OUTPUT_DIR, "pearson_correlation.csv"))

    # -----------------------------
    # 2️⃣ Spearman correlation
    # -----------------------------
    spearman_corr = df_corr.corr(method="spearman")
    spearman_target = spearman_corr[[TARGET_COL]].sort_values(by=TARGET_COL, ascending=False)
    spearman_target.to_csv(os.path.join(OUTPUT_DIR, "spearman_correlation.csv"))

    # -----------------------------
    # 3️⃣ Lag1 correlation
    # -----------------------------
    lag_df = df_corr.copy()
    for col in FEATURE_COLS:
        lag_df[f"{col}_lag1"] = lag_df[col].shift(1)
    
    lag_df = lag_df.dropna()
    lag_cols = [f"{col}_lag1" for col in FEATURE_COLS]
    lag_corr = lag_df.corr(method="pearson")
    lag_target = lag_corr.loc[lag_cols, TARGET_COL].sort_values(ascending=False)
    lag_target.to_csv(os.path.join(OUTPUT_DIR, "lag1_correlation.csv"))

    # -----------------------------
    # 4️⃣ Heatmap
    # -----------------------------
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_pearson.png"))
    plt.close()

    # -----------------------------
    # 5️⃣ Scatter plots
    # -----------------------------
    for col in FEATURE_COLS:
        plt.figure()
        sns.scatterplot(x=df_corr[col], y=df_corr[TARGET_COL], alpha=0.5)
        plt.xlabel(col)
        plt.ylabel(TARGET_COL)
        plt.title(f"{TARGET_COL} vs {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{col}.png"))
        plt.close()

    print("Correlation analysis completed.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
