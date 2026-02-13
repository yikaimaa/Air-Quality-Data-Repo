#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-aware LASSO baseline (aligned with LSTM pipeline)

- Strict time-based 80/20 split
- Metrics computed on original scale
- No data leakage
- Region-wise modeling
- Outputs full test metrics
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

from scipy.stats import pearsonr


# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input CSV file")
parser.add_argument("--output_dir", required=True, help="Output directory")
parser.add_argument("--train_frac", type=float, default=0.8)

args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir
TRAIN_FRAC = args.train_frac

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

RESPONSE = "pm25_label"

DROP_BASE = [
    "date",
    "region",
    "longitude_x",
    "latitude_x",
    "longitude_y",
    "latitude_y",
    RESPONSE
]

results = {}
selected_features = {}
all_coefficients = []


# -------------------------
# Region-wise modeling
# -------------------------
for region, g in df.groupby("region"):

    g = g.sort_values("date").dropna(subset=[RESPONSE])

    if len(g) < 50:
        print(f"[skip] {region}: too few rows")
        continue

    y = g[RESPONSE]

    X = g.drop(columns=DROP_BASE, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    # Remove rows with NA in predictors
    idx = X.dropna().index
    X = X.loc[idx]
    y = y.loc[idx]
    dates = g.loc[idx, "date"]

    if len(X) < 50:
        print(f"[skip] {region}: too few valid rows after NA removal")
        continue

    # -------------------------
    # Time-aware split
    # -------------------------
    order = np.argsort(dates.values)
    split = int(len(order) * TRAIN_FRAC)

    train_idx = order[:split]
    test_idx = order[split:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # -------------------------
    # LASSO model
    # -------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(
            cv=5,
            max_iter=20000,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # -------------------------
    # Test metrics (original scale)
    # -------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    mape = np.mean(
        np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))
    ) * 100

    smape = np.mean(
        2 * np.abs(y_pred - y_test) /
        np.maximum(np.abs(y_test) + np.abs(y_pred), 1e-6)
    ) * 100

    pearson_r = pearsonr(y_test, y_pred)[0]

    lasso = pipe.named_steps["lasso"]
    coef = pd.Series(lasso.coef_, index=X.columns)

    sel = coef[coef != 0].sort_values(key=np.abs, ascending=False)

    # Save full coefficients
    for feature, value in coef.items():
        all_coefficients.append({
            "region": region,
            "feature": feature,
            "coef": float(value)
        })

    # Save model
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"lasso_model_{region}.pkl")
    joblib.dump(pipe, model_path)

    results[region] = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_pct": float(mape),
        "smape_pct": float(smape),
        "pearson_r": float(pearson_r),
        "alpha": float(lasso.alpha_),
        "intercept": float(lasso.intercept_),
        "n_selected_features": int(len(sel))
    }

    selected_features[region] = sel

    print(f"[{region}] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")


# -------------------------
# Save region metrics
# -------------------------
results_df = pd.DataFrame(results).T.sort_values("r2", ascending=False)

results_path = os.path.join(OUTPUT_DIR, "region_model_results.csv")
results_df.to_csv(results_path, index=True)

print(f"Saved: {results_path}")


# -------------------------
# Save selected features
# -------------------------
if selected_features:
    all_selected = []

    for region, coef_series in selected_features.items():
        temp = coef_series.to_frame("coef").reset_index()
        temp.columns = ["feature", "coef"]
        temp["region"] = region
        all_selected.append(temp)

    all_selected_df = pd.concat(all_selected, ignore_index=True)

    features_path = os.path.join(OUTPUT_DIR, "all_selected_features.csv")
    all_selected_df.to_csv(features_path, index=False)

    print(f"Saved: {features_path}")


# -------------------------
# Save ALL coefficients
# -------------------------
all_coef_df = pd.DataFrame(all_coefficients)

coef_path = os.path.join(OUTPUT_DIR, "all_coefficients.csv")
all_coef_df.to_csv(coef_path, index=False)

print(f"Saved: {coef_path}")


# -------------------------
# Save JSON summary (like LSTM)
# -------------------------
summary = {
    "created_at": pd.Timestamp.now().isoformat(),
    "train_frac": TRAIN_FRAC,
    "results": results
}

summary_path = os.path.join(OUTPUT_DIR, "lasso_summary.json")

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved: {summary_path}")
print("Done.")
