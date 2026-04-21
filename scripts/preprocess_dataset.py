"""
Preprocess the raw texture dataset so heightmaps satisfy CNC constraints.

Source layout (nested, one folder per sample):
    <src>/<Category>_resized/<sample>/<sample>_diffuse.png
    <src>/<Category>_resized/<sample>/<sample>_height.png

Output layout (flat):
    <dst>/diffuse/<sample>_diffuse.png   (copied unchanged)
    <dst>/height/<sample>_height.png     (smoothed + slope-fixed)

Pipeline (heightmap only):
  1. gaussian_filter(sigma=FEATURE_SIGMA)  to push ~10px detail past MIN_FEAT_PX
  2. iterative slope repair (<=40 iters): local sigma=1.5 blur on pixels whose
     Sobel slope exceeds MAX_SLOPE_PX * 0.85
  3. renormalize to [0, 1]
  4. save as 8-bit PNG
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.cnc_params import FEATURE_SIGMA, MAX_SLOPE_PX  # noqa: E402


SLOPE_LIMIT = MAX_SLOPE_PX * 0.85
MAX_ITERS = 40
LOCAL_SIGMA = 1.5


def _sobel_slope(arr: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return np.sqrt(gx * gx + gy * gy)


def _repair_slope(arr: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Iteratively blur slope-violating regions. Returns (arr, before, after, iters)."""
    before = float(_sobel_slope(arr).max())
    iters_used = 0
    for i in range(MAX_ITERS):
        slope = _sobel_slope(arr)
        mask = slope > SLOPE_LIMIT
        if not mask.any():
            break
        blurred = gaussian_filter(arr, sigma=LOCAL_SIGMA)
        arr = np.where(mask, blurred, arr)
        iters_used = i + 1
    after = float(_sobel_slope(arr).max())
    return arr, before, after, iters_used


def _process_height(src_path: Path, dst_path: Path) -> dict:
    raw = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Failed to read {src_path}")
    h = raw.astype(np.float32) / 255.0

    h = gaussian_filter(h, sigma=FEATURE_SIGMA)
    h, slope_before, slope_after, iters = _repair_slope(h)

    lo, hi = float(h.min()), float(h.max())
    if hi - lo > 1e-6:
        h = (h - lo) / (hi - lo)
    else:
        h = np.zeros_like(h)

    out = np.clip(h * 255.0, 0, 255).astype(np.uint8)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), out)

    return {
        "file": src_path.name,
        "slope_before": slope_before,
        "slope_after": slope_after,
        "iters": iters,
        "violated": slope_after > MAX_SLOPE_PX,
    }


def _collect_pairs(src_root: Path) -> list[tuple[Path, Path, str]]:
    """Walk the nested layout and return (diffuse, height, flat_basename) tuples."""
    pairs: list[tuple[Path, Path, str]] = []
    for sample_dir in sorted(src_root.rglob("*")):
        if not sample_dir.is_dir():
            continue
        diffuse = None
        height = None
        for f in sample_dir.iterdir():
            name = f.name.lower()
            if not name.endswith(".png"):
                continue
            if "_diffuse" in name:
                diffuse = f
            elif "_height" in name:
                height = f
        if diffuse is not None and height is not None:
            basename = sample_dir.name.replace(" ", "_")
            pairs.append((diffuse, height, basename))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="D:/homework/lund/CS_project/dataset_resize")
    p.add_argument("--dst", default="D:/homework/lund/CS_project/dataset_processed")
    args = p.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    (dst_root / "diffuse").mkdir(parents=True, exist_ok=True)
    (dst_root / "height").mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(src_root)
    if not pairs:
        print(f"No (diffuse, height) pairs under {src_root}")
        sys.exit(1)
    print(f"Found {len(pairs)} paired samples under {src_root}")
    print(f"FEATURE_SIGMA={FEATURE_SIGMA:.2f}px  slope_limit={SLOPE_LIMIT:.4f}")

    log = []
    t0 = time.time()
    for i, (d_src, h_src, base) in enumerate(pairs):
        d_dst = dst_root / "diffuse" / f"{base}_diffuse.png"
        h_dst = dst_root / "height" / f"{base}_height.png"
        shutil.copy2(d_src, d_dst)
        stats = _process_height(h_src, h_dst)
        stats["basename"] = base
        log.append(stats)
        if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
            dt = time.time() - t0
            print(f"  [{i+1}/{len(pairs)}] {base}  "
                  f"slope {stats['slope_before']:.4f} -> {stats['slope_after']:.4f}  "
                  f"iters={stats['iters']}  elapsed={dt:.1f}s")

    after_slopes = np.array([e["slope_after"] for e in log])
    n_viol = int(sum(1 for e in log if e["violated"]))
    print("\n--- Summary ---")
    print(f"  samples processed    : {len(log)}")
    print(f"  slope median (after) : {float(np.median(after_slopes)):.4f}")
    print(f"  slope max    (after) : {float(after_slopes.max()):.4f}")
    print(f"  max allowed          : {MAX_SLOPE_PX:.4f}")
    print(f"  still violating      : {n_viol}")

    log_path = dst_root / "preprocess_log.json"
    with open(log_path, "w") as f:
        json.dump(
            {
                "feature_sigma": FEATURE_SIGMA,
                "max_slope_px": MAX_SLOPE_PX,
                "slope_limit": SLOPE_LIMIT,
                "samples": log,
            },
            f,
            indent=2,
        )
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
