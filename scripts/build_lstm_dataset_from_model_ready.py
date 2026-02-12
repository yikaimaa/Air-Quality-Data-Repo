"""
Build LSTM-ready dataset from model_ready_dataset.csv.

Each row in model_ready_dataset:
(region, date=t)
pm25_label = PM2.5 at t+1

We build sliding windows per region:
- X[i] = features from [t-seq_len+1 ... t]
- y[i] = pm25_label at t
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = "outputs/model_ready_dataset.csv"
DEFAULT_OUTPUT_PATH = "outputs/lstm_dataset_seq14.npz"
DEFAULT_SEQ_LEN = 14  # you can change to 7, 21, etc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=DEFAULT_INPUT_PATH,
                        help="Input CSV path (model_ready_dataset.csv).")
    parser.add_argument("--out", dest="out_path", default=DEFAULT_OUTPUT_PATH,
                        help="Output NPZ path.")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="Sequence length (e.g., 7, 14, 21).")
    args = parser.parse_args()

    input_path = args.in_path
    output_path = args.out_path
    seq_len = args.seq_len

    df = pd.read_csv(input_path)

    # -----------------------------
    # Basic validation
    # -----------------------------
    required_cols = ["region", "date", "pm25_label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    # -----------------------------
    # Define feature columns
    # -----------------------------
    feature_cols = [c for c in df.columns if c not in ["region", "date", "pm25_label"]]

    print("Using feature columns:")
    for c in feature_cols:
        print("  -", c)

    # convert all features + target to numeric
    for c in feature_cols + ["pm25_label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with any NA in features/target
    df = df.dropna(subset=feature_cols + ["pm25_label"]).copy()

    # -----------------------------
    # Build sequences per region
    # -----------------------------
    X_all = []
    y_all = []
    region_all = []
    end_date_all = []

    for region, g in df.groupby("region"):

        g = g.sort_values("date").reset_index(drop=True)

        if len(g) < seq_len:
            continue

        features = g[feature_cols].values.astype(np.float32)
        targets = g["pm25_label"].values.astype(np.float32)
        dates = g["date"].values

        for end_idx in range(seq_len - 1, len(g)):
            start_idx = end_idx - seq_len + 1

            X_window = features[start_idx:end_idx + 1]
            y_value = targets[end_idx]

            X_all.append(X_window)
            y_all.append(y_value)
            region_all.append(region)
            end_date_all.append(dates[end_idx])

    if len(X_all) == 0:
        raise ValueError("No sequences created. Try reducing --seq-len.")

    X = np.stack(X_all)
    y = np.array(y_all)

    # -----------------------------
    # Save dataset
    # -----------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        feature_names=np.array(feature_cols, dtype=object),
        region=np.array(region_all, dtype=object),
        end_date=np.array([str(d) for d in end_date_all], dtype=object),
    )

    print("\nSaved LSTM dataset to:", output_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Sequence length:", seq_len)
    print("Number of features:", len(feature_cols))


if __name__ == "__main__":
    main()
