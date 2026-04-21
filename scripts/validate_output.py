"""
Validate a single heightmap PNG against CNC constraints.

Checks (thresholds from cnc_params.py):
  - max Sobel slope (pixel space)         <= MAX_SLOPE_PX
  - estimated minimum feature size (px)   >= MIN_FEAT_PX

Also saves a slope heat-map next to the input file as *_slope_map.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label, maximum_filter

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cnc_params import MAX_SLOPE_PX, MIN_FEAT_PX  # noqa: E402


def _sobel_slope(arr: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return np.sqrt(gx * gx + gy * gy)


def _estimate_min_feature_px(arr: np.ndarray) -> float:
    local_max = (arr == maximum_filter(arr, size=5))
    _, n = label(local_max)
    if n <= 1:
        return float("inf")
    H, W = arr.shape
    return float(np.sqrt(H * W / n))


def _save_slope_heatmap(slope: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(slope, cmap="inferno")
    ax.set_title(f"Slope map (max={slope.max():.4f}, limit={MAX_SLOPE_PX:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heightmap", required=True)
    args = p.parse_args()

    hm_path = Path(args.heightmap)
    raw = cv2.imread(str(hm_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"Failed to read {hm_path}")
        sys.exit(1)
    arr = raw.astype(np.float32) / 255.0

    slope_map = _sobel_slope(arr)
    max_slope = float(slope_map.max())
    min_feat = _estimate_min_feature_px(arr)

    slope_pass = max_slope <= MAX_SLOPE_PX
    feat_pass = min_feat >= MIN_FEAT_PX

    def _label(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    print(f"Heightmap: {hm_path}")
    print(f"  Max slope (px)    : {max_slope:.4f}   "
          f"(limit {MAX_SLOPE_PX:.4f})   {_label(slope_pass)}")
    print(f"  Min feature (px)  : {min_feat:.2f}   "
          f"(limit {MIN_FEAT_PX:.2f})   {_label(feat_pass)}")

    heatmap_path = hm_path.with_name(hm_path.stem + "_slope_map.png")
    _save_slope_heatmap(slope_map, heatmap_path)
    print(f"  Slope heatmap     : {heatmap_path}")

    if not (slope_pass and feat_pass):
        sys.exit(2)


if __name__ == "__main__":
    main()
