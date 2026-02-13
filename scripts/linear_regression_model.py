import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import os
import argparse
import joblib




# -------------------------
# Load data
# -------------------------
# -------------------------
# Argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input CSV file")
parser.add_argument("--output_dir", required=True, help="Output directory")

args = parser.parse_args()

INPUT_PATH = args.input
OUTPUT_DIR = args.output_dir

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")



# -------------------------
# Config
# -------------------------
RESPONSE = "pm25_label"

# Base columns to drop
DROP_BASE = [
    "date",
    "region",
    "longitude_x",
    "latitude_x",
    "longitude_y",
    "latitude_y",
    "pm25_label",
    RESPONSE
]

TRAIN_FRAC = 0.8

results = {}
selected_features = {}
all_coefficients = []  

# -------------------------
# Region-wise modeling
# -------------------------
for region, g in df.groupby("region"):
    g = g.sort_values("date").dropna(subset=[RESPONSE])

    y = g[RESPONSE]

    # Initial predictor set
    X = g.drop(columns=DROP_BASE, errors="ignore")

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Drop NA rows (leakage-safe)
    idx = X.dropna().index
    X, y = X.loc[idx], y.loc[idx]

    if len(X) < 50:
        print(f"[skip] {region}: too few rows")
        continue

    # Time-based split
    split = int(len(X) * TRAIN_FRAC)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Standardize + LASSO
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

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    lasso = pipe.named_steps["lasso"]
    coef = pd.Series(lasso.coef_, index=X.columns)
    sel = coef[coef != 0].sort_values(key=np.abs, ascending=False)
    # 保存完整系数（包含0）
    for feature, value in coef.items():
        all_coefficients.append({
            "region": region,
            "feature": feature,
            "coef": value
        })

    # 保存模型（包含 scaler + lasso）
    model_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"lasso_model_{region}.pkl")
    joblib.dump(pipe, model_path)


    results[region] = {
    "n_train": len(X_train),
    "n_test": len(X_test),
    "rmse": rmse,
    "r2": r2,
    "alpha": lasso.alpha_,
    "intercept": lasso.intercept_, 
    "n_selected_features": len(sel)
    }



    selected_features[region] = sel

# -------------------------
# Results table
# -------------------------

results_df = pd.DataFrame(results).T.sort_values("r2", ascending=False)

results_path = os.path.join(OUTPUT_DIR, "region_model_results.csv")
results_df.to_csv(results_path, index=True)

print(f"Saved: {results_path}")



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
# Save ALL coefficients (including zeros)
# -------------------------
all_coef_df = pd.DataFrame(all_coefficients)

coef_path = os.path.join(OUTPUT_DIR, "all_coefficients.csv")
all_coef_df.to_csv(coef_path, index=False)

print(f"Saved: {coef_path}")

