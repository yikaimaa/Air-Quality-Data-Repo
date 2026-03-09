from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from sequence_data import PM25SequenceDataset, prepare_datasets, save_metadata


DEFAULT_CSV_PATH = "/Users/zhangrilong/Documents/GitHub/Air-Quality-Data-Repo/Datasets/Ontario/processed_datasets/case2_model_ready_dataset.csv"


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.regressor(last).squeeze(-1)
        return pred


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, y_scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    preds_raw = y_scaler.inverse_transform(preds)
    targets_raw = y_scaler.inverse_transform(targets)
    rmse = mean_squared_error(targets_raw, preds_raw, squared=False)
    mae = mean_absolute_error(targets_raw, preds_raw)
    r2 = r2_score(targets_raw, preds_raw)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}, preds_raw, targets_raw


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(total_n, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train a BiLSTM model for PM2.5 forecasting.")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to the model-ready CSV.")
    p.add_argument("--out_dir", type=str, default="outputs/bilstm")
    p.add_argument("--seq_len", type=int, default=14)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_log_fire_features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n===== Training BiLSTM =====")
    print(f"CSV: {csv_path}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Device: {device}")
    print("===========================\n")

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

    train_ds = PM25SequenceDataset(prepared["train"]["X"], prepared["train"]["y"])
    val_ds = PM25SequenceDataset(prepared["val"]["X"], prepared["val"]["y"])
    test_ds = PM25SequenceDataset(prepared["test"]["X"], prepared["test"]["y"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMRegressor(
        input_dim=metadata["n_features"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss()

    history = []
    best_val_rmse = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device, y_scaler)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    val_metrics, val_preds, val_targets = evaluate(model, val_loader, device, y_scaler)
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, device, y_scaler)

    torch.save(model.state_dict(), out_dir / "best_model.pt")
    joblib.dump(x_scaler, out_dir / "x_scaler.joblib")
    save_metadata(out_dir, {**metadata, "model": "bilstm", "best_val_metrics": val_metrics, "test_metrics": test_metrics})

    np.savez_compressed(
        out_dir / "predictions.npz",
        val_preds=val_preds,
        val_targets=val_targets,
        test_preds=test_preds,
        test_targets=test_targets,
        test_dates=prepared["test"]["dates"].astype(str),
        test_regions=prepared["test"]["regions"],
    )
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nFinal validation metrics:", val_metrics)
    print("Final test metrics:", test_metrics)
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
