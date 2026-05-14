"""
End-to-end validation: MLP predicted params → terrace processing → GLCM features.

For each validation heightmap:
  1. Get target GLCM features (from sweep CSV)
  2. MLP predicts (gamma, edge_sigma, morph_strength)
  3. energy→steps threshold predicts terrace_steps
  4. Apply terrace processing to original heightmap with predicted params
  5. Compute GLCM features on output
  6. Compare output features vs target features

Pass criterion: MAE < 10% of each feature's range across the val set.

Usage:
    python scripts/validate_intent_e2e.py
    python scripts/validate_intent_e2e.py --model_dir models/intent_mlp --val_split 0.2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk as morph_disk

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from training.intent_mlp import (
    INPUT_COLS, load_model, predict_intent, load_sweep_data,
)

SWEEP_CSV = ROOT / "outputs" / "feature_analysis" / "sweep_mapping.csv"
HEIGHTMAP_DIR = ROOT.parent / "dataset_split" / "heightmap"
IMG_SIZE = 512
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 64


# ──────────────────────────────────────────────────────────────
# GLCM feature extraction (mirrors feature_analysis.py)
# ──────────────────────────────────────────────────────────────

def compute_glcm_features(gray_u8: np.ndarray) -> dict:
    glcm = graycomatrix(
        gray_u8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    features = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        features[prop] = float(vals.mean())
    entropies = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a].astype(np.float64)
            p = p / (p.sum() + 1e-12)
            entropies.append(shannon_entropy(p))
    features["entropy"] = float(np.mean(entropies))
    return features


def compute_geometry_features(gray: np.ndarray) -> dict:
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    hist, _ = np.histogram(gray, bins=GLCM_LEVELS, range=(0, 1))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    return {
        "height_mean": float(gray.mean()),
        "height_std": float(gray.std()),
        "hist_entropy": float(shannon_entropy(hist)),
        "gradient_mean": float(grad_mag.mean()),
    }


def extract_features_9d(gray: np.ndarray) -> np.ndarray:
    """Extract 9-dim feature vector from a float32 [0,1] heightmap."""
    gray_u8 = (gray * (GLCM_LEVELS - 1)).astype(np.uint8)
    glcm_f = compute_glcm_features(gray_u8)
    geo_f = compute_geometry_features(gray)
    return np.array(
        [glcm_f[n] for n in INPUT_COLS[:5]] + [geo_f[n] for n in INPUT_COLS[5:]],
        dtype=np.float32,
    )


# ──────────────────────────────────────────────────────────────
# Terrace processing (mirrors feature_analysis.py)
# ──────────────────────────────────────────────────────────────

def apply_terrace_processing(
    hf: np.ndarray,
    terrace_steps: int,
    gamma: float,
    edge_sigma: float,
    morph_strength: float,
) -> np.ndarray:
    hf = hf.copy().astype(np.float32)
    p2, p98 = np.percentile(hf, 2), np.percentile(hf, 98)
    hf = np.clip((hf - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)
    bg = gaussian_filter(hf, sigma=150)
    hf = hf - bg
    hf = hf - hf.min()
    hf = hf / (hf.max() + 1e-8)
    hf = np.power(hf, gamma)
    hf = hf / (hf.max() + 1e-8)
    n = terrace_steps - 1
    hf = np.round(hf * n) / n
    if edge_sigma > 0:
        hf = gaussian_filter(hf.astype(np.float32), sigma=edge_sigma)
    hf = np.clip(hf, 0.0, 1.0).astype(np.float32)
    radius_px = max(int(morph_strength * 20), 1)
    selem = morph_disk(radius_px)
    inv = (1.0 - hf).astype(np.float32)
    inv_u8 = (inv * 255).astype(np.uint8)
    opened = cv2.morphologyEx(inv_u8, cv2.MORPH_OPEN, selem)
    hf = np.clip(1.0 - opened.astype(np.float32) / 255.0, 0, 1).astype(np.float32)
    return hf


# ──────────────────────────────────────────────────────────────
# Load heightmap by ID
# ──────────────────────────────────────────────────────────────

def load_heightmap_by_id(hmap_id: str) -> np.ndarray | None:
    """Load heightmap from dataset_split/heightmap/ by stem ID."""
    for ext in (".png", ".jpg", ".jpeg"):
        path = HEIGHTMAP_DIR / (hmap_id + ext)
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                return img.astype(np.float32) / 255.0
    return None


# ──────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────

def validate(model_dir: Path, sweep_csv: Path, val_split: float = 0.2):
    print("=" * 60)
    print("End-to-End Validation: MLP -> Terrace -> GLCM")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading trained MLP...")
    model, scaler, meta = load_model(model_dir)
    print(f"  Val MAE from training: {meta['val_mae']}")

    # Load sweep data
    print("\n[2/4] Loading sweep data...")
    X_raw, y_raw = load_sweep_data(sweep_csv)
    n_total = len(X_raw)
    print(f"  Total rows: {n_total}")

    # Reconstruct val split (same seed as training)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n_total)
    n_val = int(n_total * val_split)
    val_idx_set = set(idx[:n_val].tolist())

    # Group by heightmap_id — pick ONE row per unique heightmap in val set
    rows_by_hmap: dict[str, dict] = {}
    with open(sweep_csv, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in val_idx_set:
                hid = row["heightmap_id"]
                if hid not in rows_by_hmap:
                    rows_by_hmap[hid] = row

    print(f"  Val rows: {n_val}, unique val heightmaps: {len(rows_by_hmap)}")

    # E2E validation
    print("\n[3/4] Running E2E validation...")
    if not HEIGHTMAP_DIR.is_dir():
        print(f"  ERROR: heightmap dir not found: {HEIGHTMAP_DIR}")
        print(f"  Run: python scripts/split_dataset.py")
        return False

    target_list = []
    output_list = []
    skipped = 0

    for hid, row in rows_by_hmap.items():
        # Target GLCM features
        target = np.array([float(row[c]) for c in INPUT_COLS], dtype=np.float32)

        # MLP predict parameters
        params = predict_intent(target, model, scaler, meta)

        # Load original heightmap
        hf = load_heightmap_by_id(hid)
        if hf is None:
            if skipped < 3:
                print(f"  [WARN] file not found for id: '{hid}'")
            skipped += 1
            continue

        # Apply terrace processing with predicted params
        processed = apply_terrace_processing(
            hf,
            terrace_steps=params["terrace_steps"],
            gamma=params["gamma"],
            edge_sigma=params["edge_sigma"],
            morph_strength=params["morph_strength"],
        )

        # Compute GLCM features on output
        output_feat = extract_features_9d(processed)

        target_list.append(target)
        output_list.append(output_feat)

    if len(target_list) == 0:
        print(f"  ERROR: no valid samples. All {skipped} heightmaps failed to load.")
        print(f"  Check that heightmap files exist in: {HEIGHTMAP_DIR}")
        return False

    target_arr = np.array(target_list)
    output_arr = np.array(output_list)
    n_valid = len(target_arr)
    print(f"  Validated: {n_valid}, skipped: {skipped}")

    # Results
    print("\n[4/4] Results:")
    print("-" * 65)
    print(f"  {'Feature':20s} {'MAE':>10s} {'Range':>10s} {'MAE%':>8s} {'Pass?':>7s}")
    print("-" * 65)

    all_pass = True
    for i, fname in enumerate(INPUT_COLS):
        mae = np.abs(target_arr[:, i] - output_arr[:, i]).mean()
        feat_range = target_arr[:, i].max() - target_arr[:, i].min()
        if feat_range < 1e-8:
            feat_range = 1.0
        pct = mae / feat_range * 100
        passed = pct < 10.0
        all_pass = all_pass and passed
        mark = "PASS" if passed else "FAIL"
        print(f"  {fname:20s} {mae:10.4f} {feat_range:10.4f} {pct:7.1f}% {mark:>7s}")

    print("-" * 65)
    if all_pass:
        print("  OVERALL: PASS — all features within 10% MAE threshold")
    else:
        print("  OVERALL: FAIL — some features exceed 10% MAE threshold")

    # Save detailed CSV
    out_csv = model_dir / "e2e_validation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heightmap_id"]
                    + [f"target_{c}" for c in INPUT_COLS]
                    + [f"output_{c}" for c in INPUT_COLS]
                    + [f"abs_err_{c}" for c in INPUT_COLS])
        for i, hid in enumerate(rows_by_hmap.keys()):
            if i >= n_valid:
                break
            errs = np.abs(target_arr[i] - output_arr[i])
            w.writerow([hid]
                        + [f"{v:.6f}" for v in target_arr[i]]
                        + [f"{v:.6f}" for v in output_arr[i]]
                        + [f"{v:.6f}" for v in errs])
    print(f"\n  Detailed results → {out_csv}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="E2E validation of Intent MLP")
    parser.add_argument("--model_dir", type=str, default=str(ROOT / "models" / "intent_mlp"))
    parser.add_argument("--sweep_csv", type=str, default=str(SWEEP_CSV))
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    success = validate(
        model_dir=Path(args.model_dir),
        sweep_csv=Path(args.sweep_csv),
        val_split=args.val_split,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
