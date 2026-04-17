"""
Visual preprocessing: grayscale, edge detection, frequency features,
orientation extraction (Gabor filter bank).
Input: image path or numpy array
Output: preprocessed feature maps (H x W numpy arrays)
"""
from __future__ import annotations

import cv2
import numpy as np


def load_image_gray(path: str) -> np.ndarray:
    """Load image and convert to grayscale float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype(np.float32) / 255.0


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def extract_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map normalized to [0, 1]."""
    uint8 = (gray * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 以 Otsu 值为基础，按 1:2 比例设置双阈值
    low  = otsu_val * 0.5
    high = otsu_val * 1.0
    
    edges = cv2.Canny(uint8, threshold1=low, threshold2=high)

    return edges.astype(np.float32) / 255.0


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    """High-frequency content via Laplacian, normalized to [0, 1]."""
    uint8 = (gray * 255).astype(np.uint8)
    lap = cv2.Laplacian(uint8, cv2.CV_32F)
    lap = np.abs(lap)
    max_val = lap.max()
    if max_val > 0:
        lap /= max_val
    return lap


def extract_orientation(
    gray: np.ndarray,
    n_orientations: int = 8,
    ksize: int = 31,
    sigma: float = 4.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Gabor filter bank orientation extraction.

    Applies *n_orientations* Gabor kernels evenly spanning [0, pi).
    Each kernel is tuned to a specific orientation and detects texture
    energy along that direction.

    Args:
        gray: float32 (H, W) in [0, 1].
        n_orientations: number of orientation bins (default 8 → every 22.5°).
        ksize: Gabor kernel size in pixels (must be odd).
        sigma: Gaussian envelope std-dev — controls spatial locality.
        lambd: sinusoidal wavelength — should roughly match the texture
               period you want to capture (pixels).
        gamma: spatial aspect ratio (< 1 → elongated kernel → sharper
               directional selectivity).

    Returns:
        dict with:
          'dominant_orientation': float32 (H, W) in [0, 1], where 0 → 0°
              and 1 → 180°.  Encodes the strongest texture direction at
              each pixel.  Useful as a conditioning channel for the
              diffusion model so it can reproduce directional grain.
          'orientation_strength': float32 (H, W) in [0, 1].  High values
              mean the texture is strongly directional (e.g. wood grain);
              low values mean isotropic (e.g. sandpaper).  Feeds directly
              into the tactile 'directionality' descriptor.
          'gabor_responses': float32 (n_orientations, H, W) — per-angle
              energy maps (kept for downstream analysis / visualisation).
    """
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.empty((n_orientations, *gray.shape), dtype=np.float32)

    for i, theta in enumerate(thetas):
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
        )
        responses[i] = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))

    # ---- dominant orientation per pixel ----
    dominant_idx = responses.argmax(axis=0)                       # (H, W)
    dominant_orientation = dominant_idx.astype(np.float32) / n_orientations  # [0, 1)

    # ---- orientation strength (selectivity) ----
    #  ratio = (max_response - mean_response) / (max_response + eps)
    #  → 0 when all orientations respond equally (isotropic)
    #  → 1 when only one orientation dominates (strongly directional)
    max_resp = responses.max(axis=0)
    mean_resp = responses.mean(axis=0)
    eps = 1e-7
    orientation_strength = (max_resp - mean_resp) / (max_resp + eps)
    orientation_strength = orientation_strength.clip(0, 1)

    return {
        "dominant_orientation": dominant_orientation,
        "orientation_strength": orientation_strength,
        "gabor_responses": responses,
    }


def preprocess(path: str, size: tuple[int, int] = (512, 512)) -> dict[str, np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns dict with keys:
      'gray', 'edges', 'frequency',
      'dominant_orientation', 'orientation_strength', 'gabor_responses'
    All 2-D arrays are float32, shape (H, W), values in [0, 1].
    'gabor_responses' is (n_orientations, H, W).
    """
    gray = load_image_gray(path)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

    rgb = load_image_rgb(path)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    orientation = extract_orientation(gray)

    return {
        "gray": gray,
        "rgb": rgb,
        "edges": extract_edges(gray),
        "frequency": extract_frequency(gray),
        **orientation,
    }
