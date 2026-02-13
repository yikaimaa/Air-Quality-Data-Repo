#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train models to predict next-day PM2.5 using past N days data (default N=7).

Inputs (repo relative paths):
- LSTM dataset (.npz): Datasets/Ontario/processed_datasets/lstm_dataset_seq14.npz
  expected keys: X (N, 14, F), y (N,), feature_names (F,), end_date (N,)

This script improvements:
- Strong baselines: persistence (next=last day), mean of past N days
- Loss options: L1 / Huber (SmoothL1) / Weighted L1 (emphasize high pollution)
- Larger default model/training: hidden=128, epochs=50, early stopping kept
- Region categorical embedding if a region-like feature exists in feature_names
- log1p(y) transform (default ON) for stability; metrics computed on original scale
- Feature importance: permutation (MAE increase) + region importance (if embedded)

Run from repo root:
  python scripts/lstm_train_pm25_nextday.py --run lstm --seq-len 14
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
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


def ensure_dir(path: str) -> None:
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


def parse_dates_like(arr: np.ndarray) -> pd.DatetimeIndex:
    if isinstance(arr, pd.DatetimeIndex):
        return arr
    s = pd.Series(arr)
    if s.dtype == object:
        s = s.apply(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
    return pd.to_datetime(s, errors="coerce")


def time_split_indices_by_date(dates: pd.DatetimeIndex, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(dates)
    if dates.isna().any():
        split = int(n * train_ratio)
        idx = np.arange(n)
        return idx[:split], idx[split:]
    order = np.argsort(dates.values)
    split = int(n * train_ratio)
    return order[:split], order[split:]


def find_region_feature(feature_names: List[str]) -> Optional[str]:
    # 你可以按你实际列名继续加关键词
    candidates = [
        "region", "region_id", "region_code", "region_idx", "Region", "REGION",
        "area", "area_id", "zone", "zone_id"
    ]
    lower_map = {f.lower(): f for f in feature_names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def default_repo_paths() -> str:
    return os.path.join("Datasets", "Ontario", "processed_datasets", "lstm_dataset_seq14.npz")


# -----------------------------
# LSTM (PyTorch) with optional Region embedding + log1p target
# -----------------------------
@dataclass
class LSTMConfig:
    seq_len: int = 7
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    device: str = "auto"

    loss: str = "huber"          # "l1" | "huber" | "weighted_l1"
    huber_delta: float = 1.0     # SmoothL1 beta
    weight_alpha: float = 2.0    # weighted_l1 strength

    use_log1p: bool = True
    use_region_embed: bool = True


def train_lstm(npz_path: str, out_dir: str, cfg: LSTMConfig, seed: int = 42) -> Dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    seed_everything(seed)
    ensure_dir(out_dir)

    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"]  # (N, 14, F)
    y_raw = data["y"]  # (N,)
    feature_names = data["feature_names"].tolist() if "feature_names" in data else [f"f{i}" for i in range(X_raw.shape[-1])]
    end_date = data["end_date"] if "end_date" in data else np.arange(len(y_raw))

    # use last cfg.seq_len days
    if X_raw.shape[1] < cfg.seq_len:
        raise ValueError(f"X has seq_len={X_raw.shape[1]} < requested {cfg.seq_len}")
    X_raw = X_raw[:, -cfg.seq_len:, :]

    dates = parse_dates_like(end_date)
    train_idx, test_idx = time_split_indices_by_date(dates, train_ratio=0.8)

    X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test_raw = X_raw[test_idx], y_raw[test_idx]

    # Identify PM2.5 feature for baselines (if present)
    pm_feature_name = "pm25_region_daily_avg"
    pm_idx = feature_names.index(pm_feature_name) if pm_feature_name in feature_names else None

    # Region embedding: if region feature exists, pull it out as categorical
    region_feature = find_region_feature(feature_names) if cfg.use_region_embed else None
    region_idx = feature_names.index(region_feature) if region_feature is not None else None

    if region_idx is not None:
        # assume region constant across timesteps; use first timestep
        region_train_raw = X_train_raw[:, 0, region_idx]
        region_test_raw = X_test_raw[:, 0, region_idx]

        # factorize to 0..R-1 based on full train set
        # robust: round floats if needed
        region_train_raw = np.asarray(region_train_raw)
        region_test_raw = np.asarray(region_test_raw)
        if np.issubdtype(region_train_raw.dtype, np.floating):
            region_train_raw = np.rint(region_train_raw).astype(int)
            region_test_raw = np.rint(region_test_raw).astype(int)
        else:
            region_train_raw = region_train_raw.astype(int)
            region_test_raw = region_test_raw.astype(int)

        uniq = np.unique(region_train_raw)
        mapping = {int(v): i for i, v in enumerate(uniq.tolist())}
        # map unknown regions in test to -1 -> put to 0 (or create "UNK")
        def map_region(arr):
            out = np.array([mapping.get(int(v), -1) for v in arr], dtype=int)
            if (out < 0).any():
                # create UNK at end
                unk_id = len(mapping)
                out[out < 0] = unk_id
                return out, unk_id + 1, mapping, True
            return out, len(mapping), mapping, False

        region_train, n_regions, region_map, has_unk = map_region(region_train_raw)
        region_test, n_regions2, region_map2, has_unk2 = map_region(region_test_raw)
        n_regions = max(n_regions, n_regions2)

        # remove region feature from numeric features
        keep_cols = [i for i in range(X_raw.shape[-1]) if i != region_idx]
        X_train_num = X_train_raw[:, :, keep_cols]
        X_test_num = X_test_raw[:, :, keep_cols]
        feature_names_num = [feature_names[i] for i in keep_cols]
    else:
        region_train = None
        region_test = None
        n_regions = 0
        region_map = None
        X_train_num = X_train_raw
        X_test_num = X_test_raw
        feature_names_num = feature_names

    # Standardize numeric features (exclude region)
    feat_mean = X_train_num.reshape(-1, X_train_num.shape[-1]).mean(axis=0)
    feat_std = X_train_num.reshape(-1, X_train_num.shape[-1]).std(axis=0)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)

    def standardize(x: np.ndarray) -> np.ndarray:
        return (x - feat_mean) / feat_std

    X_train = standardize(X_train_num).astype(np.float32)
    X_test = standardize(X_test_num).astype(np.float32)

    # Target transform
    if cfg.use_log1p:
        y_train_t = np.log1p(np.maximum(y_train_raw, 0)).astype(np.float32)
        y_test_t = np.log1p(np.maximum(y_test_raw, 0)).astype(np.float32)
    else:
        y_train_t = y_train_raw.astype(np.float32)
        y_test_t = y_test_raw.astype(np.float32)

    y_train_t = y_train_t.reshape(-1, 1)
    y_test_t = y_test_t.reshape(-1, 1)

    # For weighted loss: compute scale based on train distribution in ORIGINAL space
    p90 = float(np.percentile(y_train_raw, 90))
    p90 = max(p90, 1e-6)

    class SeqDataset(Dataset):
        def __init__(self, X_, y_t_, y_orig_, region_=None):
            self.X = torch.from_numpy(X_)
            self.y_t = torch.from_numpy(y_t_)
            self.y_orig = torch.from_numpy(y_orig_.astype(np.float32).reshape(-1, 1))
            self.region = None if region_ is None else torch.from_numpy(region_.astype(np.int64))

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            if self.region is None:
                return self.X[idx], self.y_t[idx], self.y_orig[idx]
            return self.X[idx], self.y_t[idx], self.y_orig[idx], self.region[idx]

    train_ds_full = SeqDataset(X_train, y_train_t, y_train_raw, region_train)
    test_ds = SeqDataset(X_test, y_test_t, y_test_raw, region_test)

    # Build a time-aware val split from train: last 10% of train in time
    dates_train = parse_dates_like(end_date)[train_idx]
    if dates_train.isna().any():
        order_train = np.arange(len(train_idx))
    else:
        order_train = np.argsort(dates_train.values)

    n_tr = len(order_train)
    n_val = max(int(0.1 * n_tr), 1)
    val_ids = order_train[-n_val:]
    tr_ids = order_train[:-n_val] if n_tr > n_val else order_train

    def subset_dataset(ds: SeqDataset, ids: np.ndarray) -> SeqDataset:
        if ds.region is None:
            return SeqDataset(ds.X.numpy()[ids], ds.y_t.numpy()[ids], ds.y_orig.numpy()[ids].reshape(-1))
        return SeqDataset(ds.X.numpy()[ids], ds.y_t.numpy()[ids], ds.y_orig.numpy()[ids].reshape(-1), ds.region.numpy()[ids])

    tr_ds = subset_dataset(train_ds_full, tr_ids)
    val_ds = subset_dataset(train_ds_full, val_ids)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last)

    class LSTMRegressorWithRegion(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, n_regions: int):
            super().__init__()
            # embedding dim heuristic
            emb_dim = int(min(16, max(2, round(np.sqrt(max(n_regions, 2))))))
            self.region_emb = nn.Embedding(num_embeddings=n_regions, embedding_dim=emb_dim)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size + emb_dim),
                nn.Linear(hidden_size + emb_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x, region_id):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            emb = self.region_emb(region_id)
            h = torch.cat([last, emb], dim=1)
            return self.head(h)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if region_idx is None:
        model = LSTMRegressor(
            input_size=X_train.shape[-1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
    else:
        model = LSTMRegressorWithRegion(
            input_size=X_train.shape[-1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            n_regions=n_regions,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Loss functions
    l1_loss = torch.nn.L1Loss(reduction="mean")
    huber_loss = torch.nn.SmoothL1Loss(beta=cfg.huber_delta, reduction="mean")

    def compute_loss(pred_t, y_t, y_orig):
        if cfg.loss == "l1":
            return l1_loss(pred_t, y_t)
        if cfg.loss == "huber":
            return huber_loss(pred_t, y_t)
        if cfg.loss == "weighted_l1":
            # emphasize high pollution days using ORIGINAL y
            # weight = 1 + alpha * (y / p90), clipped
            w = 1.0 + cfg.weight_alpha * (y_orig / p90)
            w = torch.clamp(w, 1.0, 1.0 + 2.0 * cfg.weight_alpha)
            return torch.mean(torch.abs(pred_t - y_t) * w)
        raise ValueError(f"Unknown loss: {cfg.loss}")

    def inv_transform(y_t_np: np.ndarray) -> np.ndarray:
        if cfg.use_log1p:
            return np.expm1(y_t_np)
        return y_t_np

    # Eval: compute MAE in ORIGINAL space (for early stopping and reporting)
    @torch.no_grad()
    def eval_loader(loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        preds_t = []
        trues_orig = []
        total = 0.0
        count = 0

        for batch in loader:
            if region_idx is None:
                xb, ytb, yorigb = batch
                xb = xb.to(device)
                ytb = ytb.to(device)
                yorigb = yorigb.to(device)
                pb = model(xb)
            else:
                xb, ytb, yorigb, rb = batch
                xb = xb.to(device)
                ytb = ytb.to(device)
                yorigb = yorigb.to(device)
                rb = rb.to(device)
                pb = model(xb, rb)

            loss = compute_loss(pb, ytb, yorigb)
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)

            preds_t.append(pb.detach().cpu().numpy().reshape(-1))
            trues_orig.append(yorigb.detach().cpu().numpy().reshape(-1))

        preds_t = np.concatenate(preds_t, axis=0)
        y_true_orig = np.concatenate(trues_orig, axis=0)
        y_pred_orig = inv_transform(preds_t)
        # safety: predictions may go negative due to expm1 around small negatives -> clip
        y_pred_orig = np.clip(y_pred_orig, 0.0, None)
        return total / max(count, 1), y_true_orig, y_pred_orig

    best_val_mae = float("inf")
    best_state = None
    bad = 0

    # Training loop
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)

            if region_idx is None:
                xb, ytb, yorigb = batch
                xb = xb.to(device)
                ytb = ytb.to(device)
                yorigb = yorigb.to(device)
                pb = model(xb)
            else:
                xb, ytb, yorigb, rb = batch
                xb = xb.to(device)
                ytb = ytb.to(device)
                yorigb = yorigb.to(device)
                rb = rb.to(device)
                pb = model(xb, rb)

            loss = compute_loss(pb, ytb, yorigb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        val_loss, yv_true, yv_pred = eval_loader(val_loader)
        val_mae = mae(yv_true, yv_pred)

        if val_mae < best_val_mae - 1e-5:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, y_true, y_pred = eval_loader(test_loader)

    # -------------------------
    # Baselines (on ORIGINAL scale)
    # -------------------------
    baseline_metrics = {}

    if pm_idx is not None:
        # persistence: next day = last day's pm25 in window
        y_persist = X_test_raw[:, -1, pm_idx].reshape(-1)
        # mean 7-day: next day = mean pm25 across window
        y_mean7 = X_test_raw[:, :, pm_idx].mean(axis=1).reshape(-1)

        baseline_metrics["persistence"] = {
            "mae": mae(y_test_raw, y_persist),
            "rmse": rmse(y_test_raw, y_persist),
            "r2": r2_score_np(y_test_raw, y_persist),
        }
        baseline_metrics["mean_seq"] = {
            "mae": mae(y_test_raw, y_mean7),
            "rmse": rmse(y_test_raw, y_mean7),
            "r2": r2_score_np(y_test_raw, y_mean7),
        }

    # Model metrics (ORIGINAL scale)
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "smape_pct": smape(y_true, y_pred),
        "pearson_r": pearsonr_np(y_true, y_pred),
        "test_loss_train_space": float(test_loss),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seq_len_used": int(cfg.seq_len),
        "n_features_numeric": int(X_train.shape[-1]),
        "region_feature": region_feature if region_idx is not None else None,
        "n_regions": int(n_regions) if region_idx is not None else 0,
        "use_log1p": bool(cfg.use_log1p),
        "loss": cfg.loss,
        "device": str(device),
        "baselines": baseline_metrics,
    }

    # Save model
    model_path = os.path.join(out_dir, f"lstm_seq{cfg.seq_len}_{cfg.loss}{'_log1p' if cfg.use_log1p else ''}{'_regionemb' if region_idx is not None else ''}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "feature_names_numeric": feature_names_num,
            "feature_names_all": feature_names,
            "region_feature": region_feature,
            "region_map": region_map,
            "n_regions": n_regions,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        model_path,
    )

    # -------------------------
    # Permutation importance (MAE increase) on test
    # - numeric features: shuffle feature j across samples for all timesteps
    # - region: shuffle region ids across samples (if embedded)
    # -------------------------
    max_eval = 8000
    rng = np.random.default_rng(seed)

    # build arrays for importance evaluation
    Xpi = X_test.copy()
    ypi_true = y_test_raw.copy().reshape(-1)

    if Xpi.shape[0] > max_eval:
        sel = rng.choice(Xpi.shape[0], size=max_eval, replace=False)
        Xpi = Xpi[sel]
        ypi_true = ypi_true[sel]
        rpi = region_test[sel] if region_idx is not None else None
    else:
        rpi = region_test if region_idx is not None else None

    @torch.no_grad()
    def predict_np(X_in: np.ndarray, region_in: Optional[np.ndarray], batch: int = 1024) -> np.ndarray:
        model.eval()
        outp_t = []
        for i in range(0, len(X_in), batch):
            xb = torch.from_numpy(X_in[i:i+batch]).to(device)
            if region_idx is None:
                pb = model(xb)
            else:
                rb = torch.from_numpy(region_in[i:i+batch].astype(np.int64)).to(device)
                pb = model(xb, rb)
            outp_t.append(pb.detach().cpu().numpy().reshape(-1))
        outp_t = np.concatenate(outp_t, axis=0)
        y_pred_o = inv_transform(outp_t)
        return np.clip(y_pred_o, 0.0, None)

    base_pred = predict_np(Xpi, rpi)
    base_mae = mae(ypi_true, base_pred)

    importances = []
    for j, fn in enumerate(feature_names_num):
        X_shuf = Xpi.copy()
        perm = rng.permutation(X_shuf.shape[0])
        X_shuf[:, :, j] = X_shuf[perm, :, j]
        pred_shuf = predict_np(X_shuf, rpi)
        m = mae(ypi_true, pred_shuf)
        importances.append((fn, m - base_mae))

    # region importance (if embedded)
    if region_idx is not None and rpi is not None:
        r_shuf = rpi.copy()
        perm = rng.permutation(len(r_shuf))
        r_shuf = r_shuf[perm]
        pred_shuf = predict_np(Xpi, r_shuf)
        m = mae(ypi_true, pred_shuf)
        importances.append((f"[REGION_EMB]{region_feature}", m - base_mae))

    imp_df = pd.DataFrame(importances, columns=["feature", "mae_increase"])
    imp_df = imp_df.sort_values("mae_increase", ascending=False).reset_index(drop=True)

    imp_path = os.path.join(out_dir, "feature_importance_lstm_permutation.csv")
    imp_df.to_csv(imp_path, index=False)

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics_lstm.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "model_path": model_path, "importance_path": imp_path},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Print summary
    print("\n[LSTM] Test metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    if baseline_metrics:
        print("\n[LSTM] Baselines on test:")
        for k, v in baseline_metrics.items():
            print(f"  - {k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R2={v['r2']:.4f}")

    print("\n[LSTM] Top 15 features by permutation MAE increase:")
    print(imp_df.head(15).to_string(index=False))

    return {"metrics": metrics, "importance": imp_df, "model_path": model_path}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", choices=["lstm"], default="lstm", help="Only LSTM in this script version")
    parser.add_argument("--seq-len", type=int, default=7, help="Use last N days from seq14")
    parser.add_argument("--npz-path", type=str, default=None, help="Path to lstm_dataset_seq14.npz")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    # training knobs
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")

    # improvements
    parser.add_argument("--loss", choices=["l1", "huber", "weighted_l1"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--weight-alpha", type=float, default=2.0)

    parser.add_argument("--no-log1p", action="store_true", help="Disable log1p target transform")
    parser.add_argument("--no-region-embed", action="store_true", help="Disable region embedding even if region feature exists")

    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    npz_path = args.npz_path or default_repo_paths()

    cfg = LSTMConfig(
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
        use_region_embed=(not args.no_region_embed),
    )

    results = {}
    results["lstm"] = train_lstm(npz_path=npz_path, out_dir=args.out_dir, cfg=cfg, seed=args.seed)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": {
            "lstm": {
                "metrics": results["lstm"]["metrics"],
                "model_path": results["lstm"]["model_path"],
            }
        },
    }

    with open(os.path.join(args.out_dir, "lstm_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved summary to:", os.path.join(args.out_dir, "lstm_summary.json"))
    print("Done.")


if __name__ == "__main__":
    main()
