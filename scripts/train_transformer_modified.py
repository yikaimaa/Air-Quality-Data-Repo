

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import sequence_data
print("Using sequence_data from:", sequence_data.__file__)
from sequence_data import PM25SequenceDataset, prepare_datasets, save_metadata

DEFAULT_CSV_PATH = "/Users/zhangrilong/Documents/GitHub/Air-Quality-Data-Repo/Datasets/Ontario/processed_datasets/case2_model_ready_dataset.csv"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
        pooling: str = "mean",
        seq_len: int = 14,
        use_cls_token: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.pooling = pooling
        self.use_cls_token = use_cls_token
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        max_len = max(seq_len + (1 if use_cls_token else 0), 32)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max(d_model // 2, 16)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 16), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.pos_encoder(x)
        x = self.encoder(x)

        if self.pooling == "mean":
            pooled = x.mean(dim=1)
        elif self.pooling == "cls":
            pooled = x[:, 0, :]
        else:
            pooled = x[:, -1, :]

        pred = self.head(pooled).squeeze(-1)
        return pred


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_numpy_array(name: str, arr: np.ndarray):
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN.")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf.")


def check_tensor(name: str, t: torch.Tensor):
    if torch.isnan(t).any():
        raise ValueError(f"{name} contains NaN.")
    if torch.isinf(t).any():
        raise ValueError(f"{name} contains inf.")


def sanitize_numpy(name: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
    if bad > 0:
        print(f"[sanitize_numpy] {name}: fixed {bad} non-finite values")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def sanitize_tensor(name: str, t: torch.Tensor) -> torch.Tensor:
    bad = int(torch.isnan(t).sum().item() + torch.isinf(t).sum().item())
    if bad > 0:
        print(f"[sanitize_tensor] {name}: fixed {bad} non-finite values")
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def inverse_scale_feature(x_scaled: np.ndarray, feature_idx: int, x_scaler) -> np.ndarray:
    mean = float(x_scaler.mean_[feature_idx])
    scale = float(x_scaler.scale_[feature_idx])
    return x_scaled * scale + mean


def compute_baselines(prepared: dict, metadata: dict, x_scaler):
    feature_names = metadata.get("feature_columns", [])
    if "pm25_region_daily_avg" not in feature_names:
        return {}

    pm_idx = feature_names.index("pm25_region_daily_avg")
    X_test = np.asarray(prepared["test"]["X"], dtype=np.float32)
    y_test = np.asarray(prepared["test"]["y_raw"], dtype=np.float32).reshape(-1)

    last_pm_scaled = X_test[:, -1, pm_idx]
    last_pm_raw = inverse_scale_feature(last_pm_scaled, pm_idx, x_scaler).reshape(-1)

    mean_pm_scaled = X_test[:, :, pm_idx].mean(axis=1)
    mean_pm_raw = inverse_scale_feature(mean_pm_scaled, pm_idx, x_scaler).reshape(-1)

    baselines = {
        "persistence": {
            "mae": float(mean_absolute_error(y_test, last_pm_raw)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, last_pm_raw))),
            "r2": float(r2_score(y_test, last_pm_raw)),
        },
        "mean_seq": {
            "mae": float(mean_absolute_error(y_test, mean_pm_raw)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, mean_pm_raw))),
            "r2": float(r2_score(y_test, mean_pm_raw)),
        },
    }
    return baselines


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, loss_train_space: float | None = None):
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    eps = 1e-6
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape_pct = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100.0)
    smape_pct = float(np.mean(2.0 * np.abs(y_pred - y_true) / np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)) * 100.0)
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    out = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape_pct,
        "smape_pct": smape_pct,
        "pearson_r": pearson_r,
    }
    if loss_train_space is not None:
        out["test_loss_train_space"] = float(loss_train_space)
    return out


def evaluate(model, loader, device, y_scaler, criterion=None):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = sanitize_tensor("Validation input xb", xb.to(device))
            yb = sanitize_tensor("Validation target yb", yb.to(device)).view(-1)

            pred = model(xb)
            pred = sanitize_tensor("Validation predictions", pred)

            if criterion is not None:
                loss = criterion(pred, yb)
                total_loss += float(loss.item()) * xb.size(0)
                total_n += xb.size(0)

            preds.append(pred.detach().cpu().numpy().reshape(-1, 1))
            targets.append(yb.detach().cpu().numpy().reshape(-1, 1))

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    check_numpy_array("preds (scaled)", preds)
    check_numpy_array("targets (scaled)", targets)

    preds_raw = y_scaler.inverse_transform(preds).reshape(-1)
    targets_raw = y_scaler.inverse_transform(targets).reshape(-1)

    check_numpy_array("preds_raw", preds_raw)
    check_numpy_array("targets_raw", targets_raw)

    avg_loss = (total_loss / max(total_n, 1)) if criterion is not None else None
    metrics = compute_metrics(targets_raw, preds_raw, loss_train_space=avg_loss)
    return metrics, preds_raw, targets_raw


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = sanitize_tensor("Training input xb", xb.to(device))
        yb = sanitize_tensor("Training target yb", yb.to(device)).view(-1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        pred = sanitize_tensor("Training predictions", pred)

        loss = criterion(pred, yb)
        loss = sanitize_tensor("Training loss", loss.unsqueeze(0)).squeeze(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def permutation_importance(model, loader, device, y_scaler, feature_names, criterion=None, random_state: int = 42):
    base_metrics, _, _ = evaluate(model, loader, device, y_scaler, criterion=criterion)
    base_mae = base_metrics["mae"]

    X_test = loader.dataset.X.numpy().copy()
    y_test = loader.dataset.y.numpy().copy()
    rng = np.random.default_rng(random_state)

    rows = []
    for j, feat in enumerate(feature_names):
        X_perm = X_test.copy()
        perm_idx = rng.permutation(X_perm.shape[0])
        X_perm[:, :, j] = X_perm[perm_idx, :, j]

        perm_ds = PM25SequenceDataset(X_perm, y_test)
        perm_loader = DataLoader(perm_ds, batch_size=loader.batch_size, shuffle=False)
        perm_metrics, _, _ = evaluate(model, perm_loader, device, y_scaler, criterion=None)
        rows.append({"feature": feat, "mae_increase": float(perm_metrics["mae"] - base_mae)})

    return pd.DataFrame(rows).sort_values("mae_increase", ascending=False).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer model for PM2.5 forecasting.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to the model-ready CSV.")
    p.add_argument("--out_dir", type=str, default="outputs/transformer_csv_aligned")
    p.add_argument("--seq_len", type=int, default=14)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--ff_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--pooling", choices=["last", "mean", "cls"], default="mean")
    p.add_argument("--use_cls_token", action="store_true")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_log_fire_features", action="store_true")

    p.add_argument("--loss", choices=["l1", "huber"], default="huber")
    p.add_argument("--huber_delta", type=float, default=0.8)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-5)

    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.pooling == "cls" and not args.use_cls_token:
        args.use_cls_token = True

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===== Training Transformer =====")
    print(f"CSV: {csv_path}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Device: {device}")
    print("================================\n")

    prepared, x_scaler, y_scaler, metadata = prepare_datasets(
        csv_path=str(csv_path),
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_log_fire_features=args.use_log_fire_features,
    )

    print("Split sizes:", metadata["split_sizes"])
    print("Number of features:", metadata["n_features"])
    print()

    for split in ["train", "val", "test"]:
        prepared[split]["X"] = sanitize_numpy(f"prepared[{split}][X]", prepared[split]["X"])
        prepared[split]["y"] = sanitize_numpy(f"prepared[{split}][y]", prepared[split]["y"])
        X = prepared[split]["X"]
        y = prepared[split]["y"]
        print(
            f"{split}: X shape={X.shape}, y shape={y.shape}, "
            f"X has NaN={np.isnan(X).any()}, X has inf={np.isinf(X).any()}, "
            f"y has NaN={np.isnan(y).any()}, y has inf={np.isinf(y).any()}"
        )
    print()

    train_ds = PM25SequenceDataset(prepared["train"]["X"], prepared["train"]["y"])
    val_ds = PM25SequenceDataset(prepared["val"]["X"], prepared["val"]["y"])
    test_ds = PM25SequenceDataset(prepared["test"]["X"], prepared["test"]["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TransformerRegressor(
        input_dim=metadata["n_features"],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        pooling=args.pooling,
        seq_len=args.seq_len,
        use_cls_token=args.use_cls_token,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    criterion = nn.HuberLoss(delta=args.huber_delta) if args.loss == "huber" else nn.L1Loss()

    history = []
    best_val_rmse = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device, y_scaler, criterion=criterion)
        scheduler.step(val_metrics["rmse"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss_train_space": train_loss,
            "val_loss_train_space": val_metrics.get("test_loss_train_space"),
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)

    val_metrics, val_preds, val_targets = evaluate(model, val_loader, device, y_scaler, criterion=criterion)
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device, y_scaler, criterion=criterion)
    baseline_metrics = compute_baselines(prepared, metadata, x_scaler)

    test_metrics.update({
        "n_train": int(prepared["train"]["X"].shape[0]),
        "n_val": int(prepared["val"]["X"].shape[0]),
        "n_test": int(prepared["test"]["X"].shape[0]),
        "seq_len_used": int(args.seq_len),
        "n_features_numeric": int(metadata["n_features"]),
        "use_log1p": True,
        "loss": args.loss,
        "device": str(device),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "use_cls_token": bool(args.use_cls_token),
        "baselines": baseline_metrics,
    })

    imp_df = permutation_importance(
        model=model,
        loader=test_loader,
        device=device,
        y_scaler=y_scaler,
        feature_names=metadata.get("feature_columns", [f"f{i}" for i in range(metadata["n_features"])]),
        criterion=None,
        random_state=args.seed,
    )

    model_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.json"
    metrics_path = out_dir / "metrics_transformer.json"
    imp_path = out_dir / "feature_importance_transformer_permutation.csv"
    pred_path = out_dir / "predictions.npz"
    summary_path = out_dir / "transformer_summary.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")

    save_metadata(
        out_dir,
        {
            **metadata,
            "model": "transformer",
            "best_val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )

    np.savez_compressed(
        pred_path,
        val_preds=val_preds,
        val_targets=val_targets,
        test_preds=test_preds,
        test_targets=test_targets,
        test_dates=prepared["test"]["dates"].astype(str),
        test_regions=prepared["test"]["regions"],
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)

    imp_df.to_csv(imp_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": test_metrics, "model_path": str(model_path), "importance_path": str(imp_path)},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "results": {
            "transformer": {
                "metrics": test_metrics,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "history_path": str(history_path),
                "predictions_path": str(pred_path),
                "importance_path": str(imp_path),
                "top_features": imp_df.head(15).to_dict(orient="records"),
            }
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n[Transformer] Test metrics:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    if baseline_metrics:
        print("\n[Transformer] Baselines on test:")
        for k, v in baseline_metrics.items():
            print(f"  - {k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R2={v['r2']:.4f}")

    print("\n[Transformer] Top 15 features by permutation MAE increase:")
    print(imp_df.head(15).to_string(index=False))

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

