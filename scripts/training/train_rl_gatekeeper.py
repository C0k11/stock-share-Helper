import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class _TabDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(float(self.y[idx]), dtype=torch.float32)


class LightGatekeeper(torch.nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std <= 1e-12, 1.0, self.std)
        return (x - self.mean) / std


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_val_split(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(x.shape[0])
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(math.ceil(n * float(val_ratio)))) if n > 2 else 1
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size == 0:
        tr_idx = val_idx[:-1]
        val_idx = val_idx[-1:]
    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 19.1: train RL Gatekeeper (contextual bandit reward predictor)")
    p.add_argument("--data", default="data/training/rl_gatekeeper_dataset_v1.csv")
    p.add_argument("--out", default="models/rl_gatekeeper_v1.pt")
    p.add_argument("--target", default="y_reward_allow")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _set_seed(int(args.seed))

    df = pd.read_csv(str(args.data))
    if df.empty:
        raise SystemExit(f"Empty dataset: {args.data}")

    target_col = str(args.target)
    if target_col not in df.columns:
        raise SystemExit(f"Missing target column: {target_col}")

    feature_cols = [c for c in df.columns if c not in {"date", "system", target_col}]
    x_df = df[feature_cols].copy()
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = x_df.to_numpy(dtype=np.float32)

    y_s = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    y = y_s.to_numpy(dtype=np.float32)

    x_tr, y_tr, x_va, y_va = _train_val_split(x, y, float(args.val_ratio), int(args.seed))

    scaler = Scaler(mean=x_tr.mean(axis=0), std=x_tr.std(axis=0))
    x_tr_s = scaler.transform(x_tr)
    x_va_s = scaler.transform(x_va)

    train_loader = DataLoader(_TabDataset(x_tr_s, y_tr), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(_TabDataset(x_va_s, y_va), batch_size=int(args.batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGatekeeper(input_dim=x_tr_s.shape[1], dropout=float(args.dropout)).to(device)

    crit = torch.nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * float(xb.shape[0])

        tr_loss = tr_loss / max(1, len(train_loader.dataset))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                va_loss += float(loss.item()) * float(xb.shape[0])

        va_loss = va_loss / max(1, len(val_loader.dataset))
        print(f"epoch={epoch:03d} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

        if va_loss + 1e-10 < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"Early stopping at epoch={epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": best_state,
        "feature_names": [str(c) for c in feature_cols],
        "scaler_mean": scaler.mean.astype(np.float32).tolist(),
        "scaler_std": scaler.std.astype(np.float32).tolist(),
        "dropout": float(args.dropout),
        "best_val_loss": float(best_val_loss),
        "seed": int(args.seed),
        "target": str(target_col),
    }

    torch.save(payload, str(out_path))
    print(f"Saved model bundle: {out_path}")


if __name__ == "__main__":
    main()
