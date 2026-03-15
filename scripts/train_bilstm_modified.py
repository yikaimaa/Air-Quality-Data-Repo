#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CSV_PATH = "/Users/zhangrilong/Documents/GitHub/Air-Quality-Data-Repo/Datasets/Ontario/processed_datasets/case2_model_ready_dataset.csv"
TARGET_COL = "pm25_label"
DATE_COL = "date"
GROUP_COL = "region"
SPARSE_FIRE_COLS = ["fire_count_50km_avg", "fire_count_100km_avg", "frp_sum_100km_avg"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum())
    return float(0.0 if denom == 0 else (yt * yp).sum() / denom)


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    exclude = {TARGET_COL, DATE_COL, GROUP_COL}
    if exclude_cols:
        exclude.update(exclude_cols)
    return [c for c in df.columns if c not in exclude]


def maybe_log_transform_fire_features(df: pd.DataFrame, feature_cols: List[str], use_log_fire_features: bool) -> pd.DataFrame:
    out = df.copy()
    if use_log_fire_features:
        for col in SPARSE_FIRE_COLS:
            if col in feature_cols:
                out[col] = np.log1p(out[col].clip(lower=0))
    return out


def build_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, date_list, region_list = [], [], [], []
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    for region, g in df.groupby(GROUP_COL, sort=False):
        g = g.sort_values(DATE_COL).reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        target = g[TARGET_COL].to_numpy(dtype=np.float32)
        dates = g[DATE_COL].to_numpy()

        if len(g) < seq_len:
            continue

        for end_idx in range(seq_len - 1, len(g)):
            start_idx = end_idx - seq_len + 1
            X_list.append(feats[start_idx:end_idx + 1])
            y_list.append(target[end_idx])
            date_list.append(dates[end_idx])
            region_list.append(region)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    dates = pd.to_datetime(pd.Series(np.asarray(date_list)))
    regions = np.asarray(region_list)
    return X, y, dates, regions


def time_split_indices_by_date(dates: pd.Series, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(dates.values)
    split = int(len(order) * train_ratio)
    return order[:split], order[split:]


def compute_feature_fill_values(X_train: np.ndarray) -> np.ndarray:
    flat = X_train.reshape(-1, X_train.shape[-1])
    fill_values = np.nanmedian(flat, axis=0)
    fill_values = np.where(np.isnan(fill_values), 0.0, fill_values).astype(np.float32)
    return fill_values


def impute_features(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.where(np.isinf(X), np.nan, X)
    fill_3d = fill_values.reshape(1, 1, -1)
    X = np.where(np.isnan(X), fill_3d, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X


@dataclass
class Config:
    seq_len: int = 14
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.25
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 10
    device: str = "auto"
    loss: str = "huber"
    huber_delta: float = 1.0
    weight_alpha: float = 2.0
    use_log1p: bool = True
    use_log_fire_features: bool = False
    pooling: str = "hn"  # hn | last | mean
    bidirectional: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-5


def train_bilstm_csv(csv_path: str, out_dir: str, cfg: Config, seed: int = 42) -> Dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    seed_everything(seed)
    ensure_dir(out_dir)

    df = load_dataframe(csv_path)
    feature_cols = get_feature_columns(df)
    df = maybe_log_transform_fire_features(df, feature_cols, cfg.use_log_fire_features)

    X_raw, y_raw, dates, regions = build_sequences(df, feature_cols, cfg.seq_len)
    train_idx, test_idx = time_split_indices_by_date(dates, train_ratio=0.8)

    X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test_raw = X_raw[test_idx], y_raw[test_idx]
    dates_train, dates_test = dates.iloc[train_idx], dates.iloc[test_idx]
    regions_train, regions_test = regions[train_idx], regions[test_idx]

    fill_values = compute_feature_fill_values(X_train_raw)
    X_train_imp = impute_features(X_train_raw, fill_values)
    X_test_imp = impute_features(X_test_raw, fill_values)

    feat_mean = X_train_imp.reshape(-1, X_train_imp.shape[-1]).mean(axis=0)
    feat_std = X_train_imp.reshape(-1, X_train_imp.shape[-1]).std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)

    def standardize(x: np.ndarray) -> np.ndarray:
        z = (x - feat_mean) / feat_std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X_train = standardize(X_train_imp)
    X_test = standardize(X_test_imp)

    y_train_raw = np.nan_to_num(y_train_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    y_test_raw = np.nan_to_num(y_test_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if cfg.use_log1p:
        y_train_t = np.log1p(np.maximum(y_train_raw, 0)).astype(np.float32)
        y_test_t = np.log1p(np.maximum(y_test_raw, 0)).astype(np.float32)
    else:
        y_train_t = y_train_raw.copy()
        y_test_t = y_test_raw.copy()

    y_train_t = y_train_t.reshape(-1, 1)
    y_test_t = y_test_t.reshape(-1, 1)

    order_train = np.argsort(dates_train.values)
    n_tr = len(order_train)
    n_val = max(int(0.1 * n_tr), 1)
    val_ids = order_train[-n_val:]
    tr_ids = order_train[:-n_val] if n_tr > n_val else order_train

    class SeqDataset(Dataset):
        def __init__(self, X_, y_t_, y_orig_):
            self.X = torch.from_numpy(np.asarray(X_, dtype=np.float32))
            self.y_t = torch.from_numpy(np.asarray(y_t_, dtype=np.float32))
            self.y_orig = torch.from_numpy(np.asarray(y_orig_, dtype=np.float32).reshape(-1, 1))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.y_t[idx], self.y_orig[idx]

    tr_ds = SeqDataset(X_train[tr_ids], y_train_t[tr_ids], y_train_raw[tr_ids])
    val_ds = SeqDataset(X_train[val_ids], y_train_t[val_ids], y_train_raw[val_ids])
    test_ds = SeqDataset(X_test, y_test_t, y_test_raw)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    class BiLSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, pooling: str = "hn", bidirectional: bool = True):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.pooling = pooling
            self.num_directions = 2 if bidirectional else 1
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
            head_in = hidden_size * self.num_directions
            self.head = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            out, (h_n, _) = self.lstm(x)
            if self.bidirectional and self.pooling == "hn":
                h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_size)
                last_layer = h_n[-1]  # (2, B, H)
                pooled = torch.cat([last_layer[0], last_layer[1]], dim=1)
            elif self.pooling == "mean":
                pooled = out.mean(dim=1)
            else:
                pooled = out[:, -1, :]
            return self.head(pooled)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BiLSTMRegressor(
        input_size=X_train.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        bidirectional=cfg.bidirectional,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_lr,
    )

    l1_loss = torch.nn.L1Loss(reduction="mean")
    huber_loss = torch.nn.SmoothL1Loss(beta=cfg.huber_delta, reduction="mean")
    p90 = float(np.percentile(y_train_raw, 90))
    p90 = max(p90, 1e-6)

    def compute_loss(pred_t, y_t, y_orig):
        if cfg.loss == "l1":
            return l1_loss(pred_t, y_t)
        if cfg.loss == "huber":
            return huber_loss(pred_t, y_t)
        if cfg.loss == "weighted_l1":
            w = 1.0 + cfg.weight_alpha * (y_orig / p90)
            w = torch.clamp(w, 1.0, 1.0 + 2.0 * cfg.weight_alpha)
            return torch.mean(torch.abs(pred_t - y_t) * w)
        raise ValueError(f"Unknown loss: {cfg.loss}")

    def inv_transform(y_t_np: np.ndarray) -> np.ndarray:
        if cfg.use_log1p:
            return np.expm1(y_t_np)
        return y_t_np

    @torch.no_grad()
    def eval_loader(loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        preds_t, trues_orig = [], []
        total, count = 0.0, 0
        for xb, ytb, yorigb in loader:
            xb = xb.to(device)
            ytb = ytb.to(device)
            yorigb = yorigb.to(device)
            pb = model(xb)
            loss = compute_loss(pb, ytb, yorigb)
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)
            preds_t.append(pb.detach().cpu().numpy().reshape(-1))
            trues_orig.append(yorigb.detach().cpu().numpy().reshape(-1))
        preds_t = np.concatenate(preds_t, axis=0)
        y_true_orig = np.concatenate(trues_orig, axis=0)
        y_pred_orig = inv_transform(preds_t)
        y_pred_orig = np.clip(y_pred_orig, 0.0, None)
        return total / max(count, 1), y_true_orig, y_pred_orig

    best_val_rmse = float("inf")
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_total, train_count = 0.0, 0
        for xb, ytb, yorigb in tr_loader:
            xb = xb.to(device)
            ytb = ytb.to(device)
            yorigb = yorigb.to(device)

            opt.zero_grad(set_to_none=True)
            pb = model(xb)
            loss = compute_loss(pb, ytb, yorigb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_total += float(loss.item()) * xb.size(0)
            train_count += xb.size(0)

        tr_loss = train_total / max(train_count, 1)
        val_loss, yv, pv = eval_loader(val_loader)
        val_metrics = {
            "mae": mae(yv, pv),
            "rmse": rmse(yv, pv),
            "r2": r2_score_np(yv, pv),
        }
        scheduler.step(val_metrics["rmse"])
        current_lr = float(opt.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss_train_space": tr_loss,
            "val_loss_train_space": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)

    val_loss, yv, pv = eval_loader(val_loader)
    test_loss, yt, pt = eval_loader(test_loader)

    metrics = {
        "mae": mae(yt, pt),
        "rmse": rmse(yt, pt),
        "r2": r2_score_np(yt, pt),
        "mape_pct": mape(yt, pt),
        "smape_pct": smape(yt, pt),
        "pearson_r": pearsonr_np(yt, pt),
        "test_loss_train_space": float(test_loss),
        "n_train": int(len(tr_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_idx)),
        "seq_len_used": int(cfg.seq_len),
        "n_features_numeric": int(X_train.shape[-1]),
        "use_log1p": bool(cfg.use_log1p),
        "loss": cfg.loss,
        "device": str(device),
        "pooling": cfg.pooling,
    }

    pm_feature_name = "pm25_region_daily_avg"
    baseline_metrics = {}
    if pm_feature_name in feature_cols:
        pm_idx = feature_cols.index(pm_feature_name)
        persistence_pred = np.clip(X_test_imp[:, -1, pm_idx], 0.0, None)
        mean_seq_pred = np.clip(np.mean(X_test_imp[:, :, pm_idx], axis=1), 0.0, None)
        baseline_metrics["persistence"] = {
            "mae": mae(y_test_raw, persistence_pred),
            "rmse": rmse(y_test_raw, persistence_pred),
            "r2": r2_score_np(y_test_raw, persistence_pred),
        }
        baseline_metrics["mean_seq"] = {
            "mae": mae(y_test_raw, mean_seq_pred),
            "rmse": rmse(y_test_raw, mean_seq_pred),
            "r2": r2_score_np(y_test_raw, mean_seq_pred),
        }
        metrics["baselines"] = baseline_metrics

    rng = np.random.default_rng(seed)
    pi_n = min(2000, len(test_ds))
    pi_idx = np.arange(len(test_ds))
    if len(pi_idx) > pi_n:
        pi_idx = rng.choice(pi_idx, size=pi_n, replace=False)

    Xpi = X_test[pi_idx]
    ypi_true = y_test_raw[pi_idx]

    @torch.no_grad()
    def predict_np(X_np: np.ndarray) -> np.ndarray:
        model.eval()
        outs = []
        bs = 512
        for i in range(0, len(X_np), bs):
            xb = torch.from_numpy(X_np[i:i+bs].astype(np.float32)).to(device)
            pb = model(xb).detach().cpu().numpy().reshape(-1)
            outs.append(pb)
        pred_t = np.concatenate(outs, axis=0)
        pred = inv_transform(pred_t)
        return np.clip(pred, 0.0, None)

    base_pred = predict_np(Xpi)
    base_mae = mae(ypi_true, base_pred)
    importances = []
    for j, fn in enumerate(feature_cols):
        X_shuf = Xpi.copy()
        perm = rng.permutation(X_shuf.shape[0])
        X_shuf[:, :, j] = X_shuf[perm, :, j]
        pred_shuf = predict_np(X_shuf)
        m = mae(ypi_true, pred_shuf)
        importances.append((fn, m - base_mae))

    imp_df = pd.DataFrame(importances, columns=["feature", "mae_increase"]).sort_values("mae_increase", ascending=False).reset_index(drop=True)

    import joblib
    model_path = os.path.join(out_dir, "best_bilstm_model.pt")
    scaler_path = os.path.join(out_dir, "feature_scaler.joblib")
    metrics_path = os.path.join(out_dir, "metrics_bilstm.json")
    imp_path = os.path.join(out_dir, "feature_importance_bilstm_permutation.csv")
    preds_path = os.path.join(out_dir, "predictions_bilstm.npz")
    history_path = os.path.join(out_dir, "history_bilstm.json")
    summary_path = os.path.join(out_dir, "bilstm_summary.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump({"feat_mean": feat_mean, "feat_std": feat_std, "fill_values": fill_values, "feature_names": feature_cols}, scaler_path)
    imp_df.to_csv(imp_path, index=False)
    np.savez_compressed(preds_path, y_true=yt, y_pred=pt, test_dates=np.asarray(dates_test.astype(str)), test_regions=regions_test)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "config": asdict(cfg), "model_path": model_path, "importance_path": imp_path}, f, ensure_ascii=False, indent=2)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": csv_path,
        "result": {
            "metrics": metrics,
            "model_path": model_path,
            "importance_path": imp_path,
            "history_path": history_path,
            "predictions_path": preds_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n[BiLSTM] Test metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    if baseline_metrics:
        print("\n[BiLSTM] Baselines on test:")
        for k, v in baseline_metrics.items():
            print(f"  - {k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R2={v['r2']:.4f}")
    print("\n[BiLSTM] Top 15 features by permutation MAE increase:")
    print(imp_df.head(15).to_string(index=False))
    print("\nSaved summary to:", summary_path)

    return {
        "metrics": metrics,
        "importance_path": imp_path,
        "model_path": model_path,
        "history_path": history_path,
        "predictions_path": preds_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM on case2_model_ready_dataset.csv with fixes.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to case2_model_ready_dataset.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/bilstm_csv_aligned_fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--loss", choices=["l1", "huber", "weighted_l1"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--weight-alpha", type=float, default=2.0)
    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p target transform")
    parser.add_argument("--use-log-fire-features", action="store_true", help="Apply log1p to sparse wildfire count/FRP features")
    parser.add_argument("--pooling", choices=["hn", "last", "mean"], default="hn")
    args = parser.parse_args()

    cfg = Config(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        loss=args.loss,
        huber_delta=args.huber_delta,
        weight_alpha=args.weight_alpha,
        use_log1p=(not args.no_log1p),
        use_log_fire_features=args.use_log_fire_features,
        pooling=args.pooling,
    )

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    train_bilstm_csv(csv_path=args.csv, out_dir=args.out_dir, cfg=cfg, seed=args.seed)


if __name__ == "__main__":
    main()
