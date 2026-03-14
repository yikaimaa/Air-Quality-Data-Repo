from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


TARGET_COL = "pm25_label"
DATE_COL = "date"
GROUP_COL = "region"
DEFAULT_EXCLUDE_COLS = {
    TARGET_COL,
    DATE_COL,
    GROUP_COL,
    "longitude_x",
    "latitude_y",
    "month",
    "day",
}
SPARSE_FIRE_COLS = ["fire_count_50km_avg", "fire_count_100km_avg", "frp_sum_100km_avg"]


@dataclass
class SequenceSplit:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    regions: np.ndarray


class PM25SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class TargetStandardizer:
    def __init__(self):
        self.mean_: float | None = None
        self.scale_: float | None = None

    def fit(self, y: np.ndarray) -> "TargetStandardizer":
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        scale = float(y.std())
        self.scale_ = scale if scale > 1e-8 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float32) * self.scale_ + self.mean_

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean_, "scale": self.scale_}



def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df



def get_feature_columns(df: pd.DataFrame, exclude_cols: Sequence[str] | None = None) -> List[str]:
    exclude = set(DEFAULT_EXCLUDE_COLS)
    if exclude_cols:
        exclude.update(exclude_cols)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols



def maybe_log_transform(df: pd.DataFrame, feature_cols: Sequence[str], use_log_fire_features: bool) -> pd.DataFrame:
    out = df.copy()
    if use_log_fire_features:
        for col in SPARSE_FIRE_COLS:
            if col in feature_cols:
                out[col] = np.log1p(out[col].clip(lower=0))
    return out



def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    seq_len: int,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    date_col: str = DATE_COL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, date_list, region_list = [], [], [], []

    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    for region, g in df.groupby(group_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)
        feats = g[list(feature_cols)].to_numpy(dtype=np.float32)
        target = g[target_col].to_numpy(dtype=np.float32)
        dates = g[date_col].to_numpy()

        if len(g) < seq_len:
            continue

        for end_idx in range(seq_len - 1, len(g)):
            start_idx = end_idx - seq_len + 1
            X_list.append(feats[start_idx : end_idx + 1])
            y_list.append(target[end_idx])
            date_list.append(dates[end_idx])
            region_list.append(region)

    X = np.stack(X_list)
    y = np.asarray(y_list, dtype=np.float32)
    dates = np.asarray(date_list)
    regions = np.asarray(region_list)
    return X, y, dates, regions



def split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    regions: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, SequenceSplit]:
    unique_dates = np.array(sorted(pd.to_datetime(pd.Series(dates)).unique()))
    n_dates = len(unique_dates)
    train_end = max(1, int(n_dates * train_ratio))
    val_end = max(train_end + 1, int(n_dates * (train_ratio + val_ratio)))
    train_cut = unique_dates[min(train_end - 1, n_dates - 1)]
    val_cut = unique_dates[min(val_end - 1, n_dates - 1)]

    dt = pd.to_datetime(pd.Series(dates))
    train_mask = dt <= train_cut
    val_mask = (dt > train_cut) & (dt <= val_cut)
    test_mask = dt > val_cut

    if val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Validation or test split is empty. Adjust ratios or check dataset length.")

    return {
        "train": SequenceSplit(X[train_mask], y[train_mask], dates[train_mask], regions[train_mask]),
        "val": SequenceSplit(X[val_mask], y[val_mask], dates[val_mask], regions[val_mask]),
        "test": SequenceSplit(X[test_mask], y[test_mask], dates[test_mask], regions[test_mask]),
    }



def fit_feature_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n_train, seq_len, n_feat = X_train.shape
    scaler.fit(X_train.reshape(n_train * seq_len, n_feat))
    return scaler



def transform_features(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, seq_len, n_feat = X.shape
    X2 = scaler.transform(X.reshape(n * seq_len, n_feat)).reshape(n, seq_len, n_feat)
    return X2.astype(np.float32)



def prepare_datasets(
    csv_path: str | Path,
    seq_len: int,
    exclude_cols: Sequence[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_log_fire_features: bool = False,
):
    df = load_dataframe(csv_path)
    feature_cols = get_feature_columns(df, exclude_cols=exclude_cols)
    df = maybe_log_transform(df, feature_cols, use_log_fire_features)

    X, y, dates, regions = build_sequences(df=df, feature_cols=feature_cols, seq_len=seq_len)
    splits = split_by_time(X, y, dates, regions, train_ratio=train_ratio, val_ratio=val_ratio)

    x_scaler = fit_feature_scaler(splits["train"].X)
    y_scaler = TargetStandardizer().fit(splits["train"].y)

    prepared = {}
    for split_name, split in splits.items():
        prepared[split_name] = {
            "X": transform_features(split.X, x_scaler),
            "y": y_scaler.transform(split.y),
            "y_raw": split.y.astype(np.float32),
            "dates": split.dates,
            "regions": split.regions,
        }

    metadata = {
        "feature_columns": feature_cols,
        "seq_len": seq_len,
        "target_col": TARGET_COL,
        "excluded_columns": sorted(set(exclude_cols or []).union(DEFAULT_EXCLUDE_COLS)),
        "use_log_fire_features": use_log_fire_features,
        "n_features": len(feature_cols),
        "split_sizes": {k: int(v["X"].shape[0]) for k, v in prepared.items()},
        "target_scaler": y_scaler.state_dict(),
    }
    return prepared, x_scaler, y_scaler, metadata



def save_metadata(out_dir: str | Path, metadata: Dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
