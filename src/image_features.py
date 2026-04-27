"""
Visual analysis pipeline: image loading, feature extraction, and tactile descriptors.

Section 1 — Preprocessing: grayscale conversion, edge detection (Canny + Otsu),
    high-frequency extraction (Laplacian), and orientation analysis (Gabor filter bank).

Section 2 — Tactile mapping: GLCM texture analysis → TactileDescriptor (roughness,
    directionality, dominant spatial frequency).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ---------------------------------------------------------------------------
# Section 1: Preprocessing
# ---------------------------------------------------------------------------

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
    """Canny edge map normalized to [0, 1].  Thresholds derived from Otsu value."""
    uint8 = (gray * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    Applies n_orientations Gabor kernels evenly spanning [0, pi).

    Returns dict with:
      'dominant_orientation': float32 (H, W) in [0, 1] — strongest texture direction.
      'orientation_strength': float32 (H, W) in [0, 1] — 0=isotropic, 1=directional.
      'gabor_responses': float32 (n_orientations, H, W) — per-angle energy maps.
    """
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)
    responses = np.empty((n_orientations, *gray.shape), dtype=np.float32)

    for i, theta in enumerate(thetas):
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
        )
        responses[i] = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel))

    dominant_idx = responses.argmax(axis=0)
    dominant_orientation = dominant_idx.astype(np.float32) / n_orientations

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
      'gray', 'rgb', 'edges', 'frequency',
      'dominant_orientation', 'orientation_strength', 'gabor_responses'
    All 2-D arrays are float32 in [0, 1].  'gabor_responses' is (n_orientations, H, W).
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


# ---------------------------------------------------------------------------
# Section 2: Tactile mapping
# ---------------------------------------------------------------------------

@dataclass
class TactileDescriptor:
    roughness: float        # 0.0 (smooth) – 1.0 (rough)
    directionality: float   # 0.0 (isotropic) – 1.0 (strongly directional)
    frequency: float        # normalized dominant spatial frequency


def _to_uint_levels(gray: np.ndarray, levels: int = 16) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    gray = gray - gray.min()
    if gray.max() > 0:
        gray = gray / gray.max()
    return np.clip((gray * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)


def compute_glcm_features(
    gray: np.ndarray,
    distances: tuple[int, ...] = (1, 2, 4),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 16,
) -> dict[str, float]:
    gray_q = _to_uint_levels(gray, levels=levels)
    glcm = graycomatrix(gray_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, "contrast"),
        "correlation": graycoprops(glcm, "correlation"),
        "homogeneity": graycoprops(glcm, "homogeneity"),
        "energy": graycoprops(glcm, "energy"),
    }


def compute_roughness(glcm_features: dict[str, np.ndarray]) -> float:
    contrast = float(glcm_features["contrast"].mean())
    homogeneity = float(glcm_features["homogeneity"].mean())
    rough = 0.7 * float(np.clip(contrast / 8.0, 0.0, 1.0)) + 0.3 * (1.0 - np.clip(homogeneity, 0.0, 1.0))
    return float(np.clip(rough, 0.0, 1.0))


def compute_directionality(glcm_features: dict[str, np.ndarray]) -> float:
    contrast = glcm_features["contrast"]
    per_angle_contrast = contrast.mean(axis=0)
    mean_val = float(per_angle_contrast.mean())
    if mean_val < 1e-8:
        return 0.0
    directional_var = float(np.std(per_angle_contrast) / (mean_val + 1e-8))
    return float(np.clip(directional_var, 0.0, 1.0))


def compute_frequency_descriptor(gray: np.ndarray) -> float:
    """Normalized dominant spatial frequency via FFT magnitude spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    total = magnitude.sum()
    if total == 0:
        return 0.0
    dominant_r = float((radius * magnitude).sum() / total)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    return min(dominant_r / max_r, 1.0)


def map_features(features: dict[str, np.ndarray]) -> TactileDescriptor:
    """Convert preprocessed feature maps to a TactileDescriptor."""
    glcm_features = compute_glcm_features(features["gray"])
    return TactileDescriptor(
        roughness=compute_roughness(glcm_features),
        directionality=compute_directionality(glcm_features),
        frequency=compute_frequency_descriptor(features["gray"]),
    )
