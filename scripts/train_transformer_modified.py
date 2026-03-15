

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from copy import deepcopy
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

    return {
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
            xb = sanitize_tensor("Eval input xb", xb.to(device))
            yb = sanitize_tensor("Eval target yb", yb.to(device)).view(-1)

            pred = model(xb)
            pred = sanitize_tensor("Eval predictions", pred)

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


def permutation_importance(model, loader, device, y_scaler, feature_names, random_state: int = 42):
    base_metrics, _, _ = evaluate(model, loader, device, y_scaler, criterion=None)
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


def parse_list_arg(s: str, cast_func):
    return [cast_func(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_args():
    p = argparse.ArgumentParser(description="Train/search Transformer model for PM2.5 forecasting.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH)
    p.add_argument("--out_dir", type=str, default="outputs/transformer_search")
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

    p.add_argument("--search", action="store_true", help="Run hyperparameter search.")
    p.add_argument("--search_max_runs", type=int, default=8)

    p.add_argument("--search_d_model", type=str, default="64,96")
    p.add_argument("--search_nhead", type=str, default="4")
    p.add_argument("--search_num_layers", type=str, default="2,3")
    p.add_argument("--search_ff_dim", type=str, default="128,192")
    p.add_argument("--search_dropout", type=str, default="0.2,0.3")
    p.add_argument("--search_lr", type=str, default="3e-4,5e-4")
    p.add_argument("--search_weight_decay", type=str, default="1e-4")
    p.add_argument("--search_pooling", type=str, default="mean,last")

    return p.parse_args()


def build_search_space(args):
    d_models = parse_list_arg(args.search_d_model, int)
    nheads = parse_list_arg(args.search_nhead, int)
    num_layers = parse_list_arg(args.search_num_layers, int)
    ff_dims = parse_list_arg(args.search_ff_dim, int)
    dropouts = parse_list_arg(args.search_dropout, float)
    lrs = parse_list_arg(args.search_lr, float)
    weight_decays = parse_list_arg(args.search_weight_decay, float)
    poolings = parse_list_arg(args.search_pooling, str)

    combos = []
    for dm, nh, nl, ff, do, lr, wd, pool in itertools.product(
        d_models, nheads, num_layers, ff_dims, dropouts, lrs, weight_decays, poolings
    ):
        if dm % nh != 0:
            continue
        use_cls_token = (pool == "cls")
        combos.append({
            "d_model": dm,
            "nhead": nh,
            "num_layers": nl,
            "ff_dim": ff,
            "dropout": do,
            "lr": lr,
            "weight_decay": wd,
            "pooling": pool,
            "use_cls_token": use_cls_token,
        })

    combos = combos[: max(1, args.search_max_runs)]
    return combos


def run_single_experiment(
    run_name: str,
    cfg: dict,
    prepared: dict,
    metadata: dict,
    x_scaler,
    y_scaler,
    args,
    device,
    out_dir: Path,
):
    train_ds = PM25SequenceDataset(prepared["train"]["X"], prepared["train"]["y"])
    val_ds = PM25SequenceDataset(prepared["val"]["X"], prepared["val"]["y"])
    test_ds = PM25SequenceDataset(prepared["test"]["X"], prepared["test"]["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TransformerRegressor(
        input_dim=metadata["n_features"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["ff_dim"],
        dropout=cfg["dropout"],
        pooling=cfg["pooling"],
        seq_len=args.seq_len,
        use_cls_token=cfg["use_cls_token"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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

    print(f"\n===== Run: {run_name} =====")
    print(json.dumps(cfg, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device, y_scaler, criterion=criterion)
        scheduler.step(val_metrics["rmse"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        row = {
            "run_name": run_name,
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
            best_state = deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"Training failed for run {run_name}")

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
        "baselines": baseline_metrics,
        **cfg,
    })

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics_transformer.json"
    pred_path = run_dir / "predictions.npz"

    torch.save(model.state_dict(), model_path)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(
        pred_path,
        val_preds=val_preds,
        val_targets=val_targets,
        test_preds=test_preds,
        test_targets=test_targets,
        test_dates=prepared["test"]["dates"].astype(str),
        test_regions=prepared["test"]["regions"],
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": test_metrics, "val_metrics": val_metrics}, f, ensure_ascii=False, indent=2, default=str)

    summary_row = {
        "run_name": run_name,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "pearson_r": test_metrics["pearson_r"],
        **cfg,
    }

    return {
        "run_name": run_name,
        "cfg": cfg,
        "model": model,
        "test_loader": test_loader,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "summary_row": summary_row,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(pred_path),
    }


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

    print("\n===== Training/Search Transformer =====")
    print(f"CSV: {csv_path}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Device: {device}")
    print("======================================\n")

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

    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
    save_metadata(out_dir, {**metadata, "model": "transformer_search"})

    if args.search:
        search_space = build_search_space(args)
    else:
        search_space = [{
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pooling": args.pooling,
            "use_cls_token": bool(args.use_cls_token),
        }]

    print(f"Total runs to execute: {len(search_space)}")

    all_results = []
    best_run = None
    best_key = None

    for i, cfg in enumerate(search_space, start=1):
        run_name = f"run_{i:02d}"
        result = run_single_experiment(
            run_name=run_name,
            cfg=cfg,
            prepared=prepared,
            metadata=metadata,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        all_results.append(result)

        key = (result["val_metrics"]["rmse"], result["val_metrics"]["mae"])
        if best_run is None or key < best_key:
            best_run = result
            best_key = key

    if best_run is None:
        raise RuntimeError("No run completed successfully.")

    print("\n===== Best Run =====")
    print("Run name:", best_run["run_name"])
    print("Config:", json.dumps(best_run["cfg"], ensure_ascii=False))
    print("Val metrics:", json.dumps(best_run["val_metrics"], ensure_ascii=False, indent=2))
    print("Test metrics:", json.dumps(best_run["test_metrics"], ensure_ascii=False, indent=2))

    # permutation importance only for best run
    imp_df = permutation_importance(
        model=best_run["model"],
        loader=best_run["test_loader"],
        device=device,
        y_scaler=y_scaler,
        feature_names=metadata.get("feature_columns", [f"f{i}" for i in range(metadata["n_features"])]),
        random_state=args.seed,
    )

    imp_path = out_dir / "feature_importance_transformer_permutation.csv"
    imp_df.to_csv(imp_path, index=False)

    # save ranking table
    results_df = pd.DataFrame([r["summary_row"] for r in all_results]).sort_values(
        ["val_rmse", "val_mae"], ascending=[True, True]
    )
    results_csv = out_dir / "search_results.csv"
    results_df.to_csv(results_csv, index=False)

    # copy best model info to top-level files
    top_metrics_path = out_dir / "metrics_transformer.json"
    top_summary_path = out_dir / "transformer_summary.json"

    final_metrics = deepcopy(best_run["test_metrics"])
    final_metrics["importance_path"] = str(imp_path)
    final_metrics["search_results_path"] = str(results_csv)

    with open(top_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_run": best_run["run_name"],
                "best_config": best_run["cfg"],
                "val_metrics": best_run["val_metrics"],
                "metrics": final_metrics,
                "model_path": best_run["model_path"],
                "importance_path": str(imp_path),
                "search_results_path": str(results_csv),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "best_run": best_run["run_name"],
        "best_config": best_run["cfg"],
        "results": {
            "transformer": {
                "val_metrics": best_run["val_metrics"],
                "metrics": final_metrics,
                "model_path": best_run["model_path"],
                "metrics_path": str(top_metrics_path),
                "history_path": best_run["history_path"],
                "predictions_path": best_run["predictions_path"],
                "importance_path": str(imp_path),
                "search_results_path": str(results_csv),
                "top_features": imp_df.head(15).to_dict(orient="records"),
            }
        },
    }

    with open(top_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n[Transformer] Top 15 features by permutation MAE increase:")
    print(imp_df.head(15).to_string(index=False))

    print("\n[Transformer] Search ranking:")
    print(results_df.to_string(index=False))

    print(f"\nSaved best summary to: {top_summary_path}")
    print(f"Saved search results to: {results_csv}")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
