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
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(int(self.y[idx]), dtype=torch.long)


class LightPlanner(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(input_dim), 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _train_val_split_stratified(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_by_class: Dict[int, List[int]] = {}
    for i, yi in enumerate(y.tolist()):
        idx_by_class.setdefault(int(yi), []).append(int(i))

    train_idx: List[int] = []
    val_idx: List[int] = []
    for k, idxs in idx_by_class.items():
        idxs2 = idxs[:]
        rng.shuffle(idxs2)
        n_val = max(1, int(math.ceil(len(idxs2) * float(val_ratio)))) if len(idxs2) > 2 else 1
        val_idx.extend(idxs2[:n_val])
        train_idx.extend(idxs2[n_val:])

    train_idx = sorted(set(train_idx))
    val_idx = sorted(set(val_idx))

    if not train_idx:
        train_idx = val_idx[:-1]
        val_idx = val_idx[-1:]

    x_tr = x[train_idx]
    y_tr = y[train_idx]
    x_va = x[val_idx]
    y_va = y[val_idx]
    return x_tr, y_tr, x_va, y_va


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 17.3: train Planner SFT (MLP classifier) from tabular dataset")
    p.add_argument("--data", default="data/training/planner_dataset_v1.csv")
    p.add_argument("--out", default="models/planner_sft_v1.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _set_seed(int(args.seed))

    df = pd.read_csv(str(args.data))
    if df.empty:
        raise SystemExit(f"Empty dataset: {args.data}")

    if "y_planner_strategy" not in df.columns:
        raise SystemExit("Missing y_planner_strategy column")

    y_raw = df["y_planner_strategy"].astype(str)
    labels = sorted([x for x in y_raw.unique().tolist() if str(x).strip()])
    if not labels:
        raise SystemExit("No labels found in y_planner_strategy")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {int(i): str(lab) for lab, i in label_to_idx.items()}

    feature_cols = [c for c in df.columns if c not in {"date", "system", "y_planner_strategy"}]
    x_df = df[feature_cols].copy()
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = x_df.to_numpy(dtype=np.float32)
    y = y_raw.map(label_to_idx).to_numpy(dtype=np.int64)

    x_tr, y_tr, x_va, y_va = _train_val_split_stratified(x, y, float(args.val_ratio), int(args.seed))

    scaler = Scaler(mean=x_tr.mean(axis=0), std=x_tr.std(axis=0))
    x_tr_s = scaler.transform(x_tr)
    x_va_s = scaler.transform(x_va)

    train_loader = DataLoader(_TabDataset(x_tr_s, y_tr), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(_TabDataset(x_va_s, y_va), batch_size=int(args.batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightPlanner(input_dim=x_tr_s.shape[1], output_dim=len(labels), dropout=float(args.dropout)).to(device)

    crit = torch.nn.CrossEntropyLoss()
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
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * float(xb.shape[0])

        tr_loss = tr_loss / max(1, len(train_loader.dataset))

        model.eval()
        va_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += float(loss.item()) * float(xb.shape[0])
                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.shape[0])

        va_loss = va_loss / max(1, len(val_loader.dataset))
        va_acc = float(correct) / max(1, total)

        print(f"epoch={epoch:03d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_loss + 1e-8 < best_val_loss:
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
        "idx_to_label": {int(k): str(v) for k, v in idx_to_label.items()},
        "label_to_idx": {str(k): int(v) for k, v in label_to_idx.items()},
        "dropout": float(args.dropout),
        "best_val_loss": float(best_val_loss),
        "seed": int(args.seed),
    }

    torch.save(payload, str(out_path))
    print(f"Saved model bundle: {out_path}")


if __name__ == "__main__":
    main()
