"""
Feature Analysis: GLCM + Geometry → PCA/UMAP → CLIP Alignment

Usage (in conda tact env):
    pip install scikit-image scikit-learn umap-learn
    python scripts/feature_analysis.py

    # For CLIP step (requires torch):
    pip install git+https://github.com/openai/CLIP.git
    python scripts/feature_analysis.py --clip

    # Build parameter→GLCM mapping table for CVAE training:
    python scripts/feature_analysis.py --sweep
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk as morph_disk
from skimage.measure import shannon_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

DATASET_DIR = Path(r"D:\homework\lund\CS_project\dataset_split")
OUTPUT_DIR = Path(r"D:\homework\lund\CS_project\Tact_gen\outputs\feature_analysis")
CATEGORIES = ["Bark", "Wood", "Flooring"]
IMG_SIZE = 512
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS = 64

FEAT_NAMES = [
    "contrast", "homogeneity", "energy", "correlation", "entropy",
    "height_mean", "height_std", "hist_entropy", "gradient_mean",
]

COLOR_MAP = {"Bark": "#8B4513", "Wood": "#DAA520", "Flooring": "#708090"}


# ──────────────────────────────────────────────────────────────
# Step 1: Feature extraction
# ──────────────────────────────────────────────────────────────

def find_heightmaps(dataset_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (heightmap_path, category_label).

    Expects flat folder: dataset_split/heightmap/Bark__bark 01_height.png
    Category is extracted from the prefix before '__'.
    """
    hmap_dir = dataset_dir / "heightmap"
    if not hmap_dir.is_dir():
        print(f"[WARN] heightmap dir not found: {hmap_dir}")
        return []
    results = []
    for f in sorted(hmap_dir.iterdir()):
        if not f.is_file():
            continue
        if "_height" not in f.name.lower():
            continue
        # extract category from "Category__original_filename"
        if "__" in f.name:
            cat = f.name.split("__", 1)[0]
        else:
            cat = "Unknown"
        results.append((f, cat))
    return results


def load_heightmap_gray(path: Path) -> np.ndarray:
    """Load heightmap as float32 [0,1] grayscale, resized to IMG_SIZE."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def compute_glcm_features(gray_u8: np.ndarray) -> dict:
    """Compute GLCM texture features averaged over 4 directions & multiple distances."""
    glcm = graycomatrix(
        gray_u8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    features = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        features[prop] = float(vals.mean())
    # entropy: average over GLCM probability matrices
    entropies = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a].astype(np.float64)
            p = p / (p.sum() + 1e-12)
            entropies.append(shannon_entropy(p))
    features["entropy"] = float(np.mean(entropies))
    return features


def compute_geometry_features(gray: np.ndarray) -> dict:
    """Compute heightmap geometric / statistical features."""
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    hist, _ = np.histogram(gray, bins=GLCM_LEVELS, range=(0, 1))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)
    hist_entropy = shannon_entropy(hist)

    return {
        "height_mean": float(gray.mean()),
        "height_std": float(gray.std()),
        "hist_entropy": float(hist_entropy),
        "gradient_mean": float(grad_mag.mean()),
    }


def extract_features(paths_labels: list[tuple[Path, str]]) -> tuple[np.ndarray, list[str]]:
    """Extract full feature matrix. Returns (features, labels)."""
    rows = []
    labels = []
    for i, (path, cat) in enumerate(paths_labels):
        gray = load_heightmap_gray(path)
        gray_u8 = (gray * (GLCM_LEVELS - 1)).astype(np.uint8)

        glcm_f = compute_glcm_features(gray_u8)
        geo_f = compute_geometry_features(gray)

        row = [glcm_f[n] for n in FEAT_NAMES[:5]] + [geo_f[n] for n in FEAT_NAMES[5:]]
        rows.append(row)
        labels.append(cat)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(paths_labels)}] {path.parent.name}")

    return np.array(rows), labels


# ──────────────────────────────────────────────────────────────
# Step 2: PCA + UMAP visualisation
# ──────────────────────────────────────────────────────────────

def plot_pca(features: np.ndarray, labels: list[str], out: Path):
    """2D PCA scatter coloured by category."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(labels) if l == cat]
        ax.scatter(X2[idx, 0], X2[idx, 1], c=COLOR_MAP.get(cat, "gray"),
                   label=cat, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of GLCM + Geometry Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  PCA plot → {out}")

    # Print loadings for interpretation
    loadings = pca.components_
    print("\n  PCA loadings (feature → PC direction):")
    for j, name in enumerate(FEAT_NAMES):
        print(f"    {name:20s}  PC1={loadings[0,j]:+.3f}  PC2={loadings[1,j]:+.3f}")

    return X, pca, scaler


def plot_umap(features_scaled: np.ndarray, labels: list[str], out: Path):
    """2D UMAP scatter coloured by category."""
    try:
        import umap
    except ImportError:
        print("  [SKIP] umap-learn not installed. pip install umap-learn")
        return None

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="euclidean", random_state=42)
    X2 = reducer.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(labels) if l == cat]
        ax.scatter(X2[idx, 0], X2[idx, 1], c=COLOR_MAP.get(cat, "gray"),
                   label=cat, alpha=0.6, s=20, edgecolors="none")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP of GLCM + Geometry Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  UMAP plot → {out}")
    return X2


# ──────────────────────────────────────────────────────────────
# Step 3: CLIP semantic alignment
# ──────────────────────────────────────────────────────────────

CLIP_WORDS = ["soft", "rough", "hard", "smooth", "grainy",
              "bumpy", "flat", "coarse", "fine", "textured"]


def clip_alignment(features_scaled: np.ndarray, labels: list[str],
                   pca: PCA, scaler: StandardScaler, out: Path):
    """Project CLIP word embeddings into the PCA feature space and overlay."""
    try:
        import clip
        import torch
    except ImportError:
        print("  [SKIP] CLIP not installed.")
        print("         pip install git+https://github.com/openai/CLIP.git")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Encode target words
    text_tokens = clip.tokenize(CLIP_WORDS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    clip_vectors = text_features.cpu().numpy()

    # Category anchor prompts
    cat_prompts = {
        "Bark": "a rough tree bark texture",
        "Wood": "a wood grain texture",
        "Flooring": "a smooth floor tile texture",
    }
    cat_texts = [cat_prompts[c] for c in CATEGORIES]
    cat_tokens = clip.tokenize(cat_texts).to(device)
    with torch.no_grad():
        cat_features = model.encode_text(cat_tokens)
        cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)
    cat_clip = cat_features.cpu().numpy()

    # Category centroids in scaled feature space
    labels_arr = np.array(labels)
    cat_centroids_feat = []
    for cat in CATEGORIES:
        mask = labels_arr == cat
        cat_centroids_feat.append(features_scaled[mask].mean(axis=0))
    cat_centroids_feat = np.array(cat_centroids_feat)

    # Linear map: CLIP 512-d → 9-d feature space (ridge regression)
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=1.0)
    reg.fit(cat_clip, cat_centroids_feat)

    words_feat = reg.predict(clip_vectors)
    words_pca = pca.transform(words_feat)

    # Plot overlay on PCA space
    X2 = pca.transform(features_scaled)
    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(labels) if l == cat]
        ax.scatter(X2[idx, 0], X2[idx, 1], c=COLOR_MAP.get(cat, "gray"),
                   label=cat, alpha=0.3, s=15, edgecolors="none")
    ax.scatter(words_pca[:, 0], words_pca[:, 1], c="red", marker="*", s=200,
               zorder=5, label="CLIP words")
    for i, w in enumerate(CLIP_WORDS):
        ax.annotate(w, (words_pca[i, 0], words_pca[i, 1]),
                    fontsize=9, fontweight="bold", color="red",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("CLIP Semantic Alignment in GLCM Feature Space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  CLIP alignment plot → {out}")

    # Print word positions in feature space
    print("\n  CLIP word positions in feature space (scaled units):")
    for i, w in enumerate(CLIP_WORDS):
        vals = ", ".join(f"{n}={words_feat[i,j]:+.2f}" for j, n in enumerate(FEAT_NAMES))
        print(f"    {w:12s}  {vals}")


# ──────────────────────────────────────────────────────────────
# Step 4: Parameter sweep — build mapping table for CVAE
# ──────────────────────────────────────────────────────────────

# terrace_steps treated as categorical — expanded to 6 values
SWEEP_STEPS = [4, 6, 8, 10, 12, 16]

# Continuous parameters: 11 representative combos (Latin-hypercube-style + intent anchors)
SWEEP_CONTINUOUS = [
    # gamma  edge_sigma  morph_strength
    (0.5,   0.5,        0.5),
    (0.5,   3.0,        2.0),
    (0.5,   4.5,        2.0),   # soft anchor (gamma=0.5, morph=2.0, high edge)
    (0.5,   0.5,        2.0),   # hard anchor (gamma=0.5, morph=2.0, low edge)
    (1.0,   1.5,        0.5),
    (1.0,   4.5,        2.0),
    (1.5,   0.5,        1.5),
    (1.5,   1.5,        2.0),   # organic anchor (gamma=1.5, morph=2.0)
    (1.5,   3.0,        1.0),
    (2.0,   1.5,        2.0),
    (2.0,   4.5,        0.5),   # rough anchor (gamma=2.0, morph=0.5)
]

# One-hot column names for terrace_steps
STEPS_ONEHOT = [f"steps_{s}" for s in SWEEP_STEPS]

# Full CSV column names (terrace_steps kept as int for reference,
# plus one-hot columns for CVAE training)
SWEEP_PARAM_NAMES = ["terrace_steps"] + STEPS_ONEHOT + ["gamma", "edge_sigma", "morph_strength"]


def build_sweep_grid() -> list[tuple[int, float, float, float]]:
    """Build full sweep grid: each terrace_steps × each continuous combo.

    Returns list of (terrace_steps, gamma, edge_sigma, morph_strength).
    6 steps × 8 combos = 48 rows per sample.
    """
    grid = []
    for steps in SWEEP_STEPS:
        for gamma, esigma, morph in SWEEP_CONTINUOUS:
            grid.append((steps, gamma, esigma, morph))
    return grid


def apply_terrace_processing(
    hf: np.ndarray,
    terrace_steps: int,
    gamma: float,
    edge_sigma: float,
    morph_strength: float,
) -> np.ndarray:
    """Apply simplified terrace pipeline to a [0,1] heightfield.

    Pipeline:
      1. Percentile normalization (p2/p98)
      2. Detrend: subtract sigma=150 background
      3. Gamma correction
      4. Quantize to terrace_steps levels
      5. Gaussian soften edges (edge_sigma)
      6. Morphological opening (morph_strength → tool radius in px)

    Returns processed heightfield [0,1].
    """
    hf = hf.copy().astype(np.float32)

    # 1. Percentile normalization
    p2, p98 = np.percentile(hf, 2), np.percentile(hf, 98)
    hf = np.clip((hf - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)

    # 2. Detrend
    bg = gaussian_filter(hf, sigma=150)
    hf = hf - bg
    hf = hf - hf.min()
    hf = hf / (hf.max() + 1e-8)

    # 3. Gamma correction
    hf = np.power(hf, gamma)
    hf = hf / (hf.max() + 1e-8)

    # 4. Quantize to terrace_steps levels
    n = terrace_steps - 1
    hf = np.round(hf * n) / n

    # 5. Soften step edges
    if edge_sigma > 0:
        hf = gaussian_filter(hf.astype(np.float32), sigma=edge_sigma)
    hf = np.clip(hf, 0.0, 1.0).astype(np.float32)

    # 6. Morphological opening (simulate tool constraint)
    # morph_strength 0.5→2.0 maps to radius ~10→40 px at 512 resolution
    radius_px = max(int(morph_strength * 20), 1)
    selem = morph_disk(radius_px)
    inv = (1.0 - hf).astype(np.float32)
    inv_u8 = (inv * 255).astype(np.uint8)
    opened = cv2.morphologyEx(inv_u8, cv2.MORPH_OPEN, selem)
    hf = np.clip(1.0 - opened.astype(np.float32) / 255.0, 0, 1).astype(np.float32)

    return hf


def select_representative_samples(
    features_scaled: np.ndarray,
    labels: list[str],
    paths_labels: list[tuple[Path, str]],
    pca: PCA,
    n_bark: int = 10,
    n_wood: int = 12,
    n_floor: int = 8,
) -> list[tuple[Path, str]]:
    """Select samples spanning the PCA PC1 axis, stratified by category.

    Uses PC1 quantiles to ensure coverage of the feature space.
    """
    pc1 = features_scaled @ pca.components_[0]  # project onto PC1
    targets = {"Bark": n_bark, "Wood": n_wood, "Flooring": n_floor}
    selected = []

    for cat, n_pick in targets.items():
        cat_idx = [i for i, l in enumerate(labels) if l == cat]
        if len(cat_idx) == 0:
            continue
        cat_pc1 = pc1[cat_idx]
        # pick n_pick samples at evenly spaced quantiles
        quantiles = np.linspace(0, 1, n_pick)
        picked_local = np.searchsorted(np.sort(cat_pc1), np.quantile(cat_pc1, quantiles))
        picked_local = np.clip(picked_local, 0, len(cat_idx) - 1)
        picked_local = np.unique(picked_local)  # deduplicate
        for li in picked_local:
            orig_idx = cat_idx[li]
            selected.append(paths_labels[orig_idx])

    print(f"  Selected {len(selected)} representative samples:")
    for cat in CATEGORIES:
        n = sum(1 for _, c in selected if c == cat)
        print(f"    {cat}: {n}")
    return selected


def run_parameter_sweep(
    samples: list[tuple[Path, str]],
    output_dir: Path,
) -> Path:
    """Run terrace parameter sweep on selected samples.

    For each sample × 48 combinations (6 steps × 8 continuous combos):
    apply processing, compute GLCM features.  terrace_steps is one-hot
    encoded for CVAE training.

    Returns path to CSV.
    """
    grid = build_sweep_grid()
    sweep_csv = output_dir / "sweep_mapping.csv"
    total = len(samples) * len(grid)
    print(f"\n  Sweep: {len(samples)} samples × {len(grid)} combos = {total} runs")
    print(f"  terrace_steps values: {SWEEP_STEPS}")

    header = ["heightmap_id", "category"] + SWEEP_PARAM_NAMES + FEAT_NAMES
    rows = []
    done = 0

    for path, cat in samples:
        hf = load_heightmap_gray(path)
        hmap_id = path.stem  # e.g. "Bark__bark 01_height"

        for steps, gamma, esigma, morph in grid:
            processed = apply_terrace_processing(hf, steps, gamma, esigma, morph)
            gray_u8 = (processed * (GLCM_LEVELS - 1)).astype(np.uint8)

            glcm_f = compute_glcm_features(gray_u8)
            geo_f = compute_geometry_features(processed)

            # One-hot encode terrace_steps
            onehot = [1 if s == steps else 0 for s in SWEEP_STEPS]

            row = (
                [hmap_id, cat, steps]
                + onehot
                + [gamma, esigma, morph]
                + [glcm_f[n] for n in FEAT_NAMES[:5]]
                + [geo_f[n] for n in FEAT_NAMES[5:]]
            )
            rows.append(row)
            done += 1

        if done % 100 == 0:
            print(f"    [{done}/{total}]")

    with open(sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])

    print(f"  Sweep mapping saved → {sweep_csv}  ({len(rows)} rows)")
    return sweep_csv


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLCM feature analysis for heightmaps")
    parser.add_argument("--clip", action="store_true", help="Run CLIP alignment step (requires torch + clip)")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep to build CVAE mapping table")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover heightmaps ──
    print("Step 0: Discovering heightmaps...")
    paths_labels = find_heightmaps(dataset_dir)
    print(f"  Found {len(paths_labels)} heightmaps across {len(CATEGORIES)} categories")
    for cat in CATEGORIES:
        n = sum(1 for _, c in paths_labels if c == cat)
        print(f"    {cat}: {n}")

    if len(paths_labels) == 0:
        print("ERROR: No heightmaps found. Check dataset path.")
        sys.exit(1)

    # ── Step 1: Extract features ──
    print("\nStep 1: Extracting GLCM + geometry features...")
    features, labels = extract_features(paths_labels)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Features: {FEAT_NAMES}")

    # Save raw features to CSV
    feat_csv = output_dir / "features.csv"
    with open(feat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "category"] + FEAT_NAMES)
        for i, (path, cat) in enumerate(paths_labels):
            w.writerow([str(path), cat] + [f"{features[i,j]:.6f}" for j in range(len(FEAT_NAMES))])
    print(f"  Features CSV → {feat_csv}")

    # ── Step 2: PCA + UMAP ──
    print("\nStep 2: Visualising feature space...")
    X_scaled, pca_model, scaler = plot_pca(features, labels, output_dir / "pca_scatter.png")
    plot_umap(X_scaled, labels, output_dir / "umap_scatter.png")

    # ── Step 3: CLIP alignment ──
    if args.clip:
        print("\nStep 3: CLIP semantic alignment...")
        clip_alignment(X_scaled, labels, pca_model, scaler, output_dir / "clip_alignment.png")
    else:
        print("\nStep 3: [SKIPPED] Pass --clip to enable CLIP alignment.")

    # ── Step 4: Parameter sweep ──
    if args.sweep:
        print("\nStep 4: Parameter sweep for CVAE mapping...")
        samples = select_representative_samples(X_scaled, labels, paths_labels, pca_model)
        run_parameter_sweep(samples, output_dir)
    else:
        print("\nStep 4: [SKIPPED] Pass --sweep to enable parameter sweep.")

    print("\nDone. All outputs in:", output_dir)


if __name__ == "__main__":
    main()
