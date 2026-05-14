"""
Intent MLP: maps GLCM feature targets → terrace processing parameters.

Pipeline:
  1. Load sweep_mapping.csv (1440 rows)
  2. Extract 9-dim GLCM input features + 3-dim parameter targets
  3. Train MLP with MSE loss, Adam, ReduceLROnPlateau, early stopping
  4. Save model + scaler + energy→steps thresholds

Usage (from project root):
    python -m src.training.intent_mlp --sweep_csv outputs/feature_analysis/sweep_mapping.csv
    python -m src.training.intent_mlp --sweep_csv ... --epochs 500 --lr 1e-3

Inference:
    model, scaler, meta = load_model("models/intent_mlp")
    params = predict_intent(glcm_vector, model, scaler, meta)
    # → {"gamma": 1.3, "edge_sigma": 2.1, "morph_strength": 0.8, "terrace_steps": 8}
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────
# Feature / parameter column names (must match sweep_mapping.csv)
# ──────────────────────────────────────────────────────────────

# 9 input features (GLCM + geometry) — as listed in user's spec
INPUT_COLS = [
    "contrast", "homogeneity", "energy", "correlation", "entropy",
    "height_mean", "height_std", "hist_entropy", "gradient_mean",
]

# 3 continuous output parameters (sigmoid → scaled to range)
PARAM_RANGES = {
    "gamma":          (0.8, 1.5),
    "edge_sigma":     (0.5, 4.5),
    "morph_strength": (0.5, 2.0),
}
OUTPUT_COLS = list(PARAM_RANGES.keys())

# Energy → terrace_steps mapping
ENERGY_STEPS_THRESHOLDS = [
    (0.33, 4),
    (0.25, 6),
    (0.22, 8),
    (0.21, 10),
    (0.19, 12),
    (0.00, 16),
]


# ──────────────────────────────────────────────────────────────
# MLP Architecture
# ──────────────────────────────────────────────────────────────

class IntentMLP(nn.Module):
    """9-dim GLCM features → 3-dim processing parameters (sigmoid output)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(9),
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_sweep_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load sweep_mapping.csv, return (X, y) arrays.

    X: (N, 9) GLCM features
    y: (N, 3) continuous parameters [gamma, edge_sigma, morph_strength].
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X = np.array([[float(r[c]) for c in INPUT_COLS] for r in rows], dtype=np.float32)
    y = np.array([[float(r[c]) for c in OUTPUT_COLS] for r in rows], dtype=np.float32)
    return X, y


def normalize_params(y: np.ndarray) -> np.ndarray:
    """Scale parameters from physical range to [0, 1] for sigmoid training."""
    y_norm = np.zeros_like(y)
    for i, col in enumerate(OUTPUT_COLS):
        lo, hi = PARAM_RANGES[col]
        y_norm[:, i] = (y[:, i] - lo) / (hi - lo)
    return np.clip(y_norm, 0.0, 1.0)


def denormalize_params(y_norm: np.ndarray) -> np.ndarray:
    """Scale from [0, 1] back to physical range."""
    y = np.zeros_like(y_norm)
    for i, col in enumerate(OUTPUT_COLS):
        lo, hi = PARAM_RANGES[col]
        y[:, i] = y_norm[:, i] * (hi - lo) + lo
    return y


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

class FeatureScaler:
    """Standardise features to zero mean, unit variance."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(np.array(d["mean"]), np.array(d["std"]))


def train(
    sweep_csv: Path,
    output_dir: Path,
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    val_split: float = 0.2,
) -> Path:
    """Train the Intent MLP and save model + metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load & split
    X_raw, y_raw = load_sweep_data(sweep_csv)
    print(f"  Loaded {X_raw.shape[0]} rows, {X_raw.shape[1]} inputs, {y_raw.shape[1]} outputs")

    y_norm = normalize_params(y_raw)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_raw))
    n_val = int(len(idx) * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
    y_train, y_val = y_norm[train_idx], y_norm[val_idx]

    scaler = FeatureScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )

    # Model
    model = IntentMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.5
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = criterion(val_pred, y_val_t.to(device)).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save metadata
    model.load_state_dict(torch.load(output_dir / "best.pt", weights_only=True))

    model.eval()
    with torch.no_grad():
        val_pred_phys = denormalize_params(model(X_val_t.to(device)).cpu().numpy())
        val_true_phys = denormalize_params(y_val)
        mae = np.abs(val_pred_phys - val_true_phys).mean(axis=0)

    meta = {
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
        "param_ranges": PARAM_RANGES,
        "scaler": scaler.to_dict(),
        "energy_steps_thresholds": ENERGY_STEPS_THRESHOLDS,
        "val_mae": {col: float(mae[i]) for i, col in enumerate(OUTPUT_COLS)},
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "best_val_loss": float(best_val_loss),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(output_dir / "loss_curve.json", "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    print(f"\n  Model saved → {output_dir}")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Validation MAE (physical units):")
    for col, v in meta["val_mae"].items():
        lo, hi = PARAM_RANGES[col]
        pct = v / (hi - lo) * 100
        print(f"    {col:16s}  MAE={v:.4f}  ({pct:.1f}% of range)")

    return output_dir


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def load_model(model_dir: Path | str) -> tuple[IntentMLP, FeatureScaler, dict]:
    """Load trained model, scaler, and metadata."""
    model_dir = Path(model_dir)
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)

    scaler = FeatureScaler.from_dict(meta["scaler"])
    model = IntentMLP()
    model.load_state_dict(torch.load(model_dir / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    return model, scaler, meta


def energy_to_steps(energy: float) -> int:
    """Map energy target value to terrace_steps via threshold rules."""
    for threshold, steps in ENERGY_STEPS_THRESHOLDS:
        if energy > threshold:
            return steps
    return 16


def predict_intent(
    glcm_target: list[float] | np.ndarray,
    model: IntentMLP,
    scaler: FeatureScaler,
    meta: dict,
) -> dict:
    """Predict TactileIntent parameters from a 9-dim GLCM target vector.

    Returns dict: gamma, edge_sigma, morph_strength, terrace_steps.
    """
    x = np.array(glcm_target, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_norm = model(x_tensor).numpy()[0]

    params = denormalize_params(y_norm.reshape(1, -1))[0]
    energy_idx = meta["input_cols"].index("energy")
    steps = energy_to_steps(glcm_target[energy_idx])

    return {
        "gamma": float(np.clip(params[0], 0.8, 1.5)),
        "edge_sigma": float(np.clip(params[1], 0.5, 4.5)),
        "morph_strength": float(np.clip(params[2], 0.5, 2.0)),
        "terrace_steps": steps,
    }


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Intent MLP")
    parser.add_argument("--sweep_csv", type=str,
                        default="outputs/feature_analysis/sweep_mapping.csv")
    parser.add_argument("--output_dir", type=str, default="models/intent_mlp")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        sweep_csv=Path(args.sweep_csv),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
