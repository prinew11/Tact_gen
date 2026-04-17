"""
Tactile feature mapping: derive roughness, directionality, frequency descriptors
from visual preprocessed feature maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TactileDescriptor:
    roughness: float        # 0.0 (smooth) – 1.0 (rough)
    directionality: float   # 0.0 (isotropic) – 1.0 (strongly directional)
    frequency: float        # normalized dominant spatial frequency


def compute_roughness(frequency_map: np.ndarray) -> float:
    """Mean high-frequency energy as roughness proxy."""
    return float(np.mean(frequency_map))


def compute_directionality(
    orientation_strength: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> float:
    """
    Directional selectivity of the texture, in [0, 1].

    Prefers Gabor-based orientation_strength (from preprocessing) when
    available; falls back to gradient histogram entropy on gray.
    """
    if orientation_strength is not None:
        return float(np.mean(orientation_strength))

    # fallback: gradient-based (kept for backward compat)
    if gray is None:
        return 0.0
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    angles = np.arctan2(gy, gx)
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32)
    if hist.sum() == 0:
        return 0.0
    hist /= hist.sum()
    entropy = -np.sum(hist * np.log(hist + 1e-9))
    max_entropy = np.log(36)
    return float(1.0 - entropy / max_entropy)


def compute_frequency_descriptor(gray: np.ndarray) -> float:
    """Normalized dominant spatial frequency via FFT magnitude spectrum."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    # weighted mean radius
    total = magnitude.sum()
    if total == 0:
        return 0.0
    dominant_r = float((radius * magnitude).sum() / total)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    return min(dominant_r / max_r, 1.0)


def map_features(features: dict[str, np.ndarray]) -> TactileDescriptor:
    """Convert preprocessed feature maps to a TactileDescriptor."""
    return TactileDescriptor(
        roughness=compute_roughness(features["frequency"]),
        directionality=compute_directionality(
            orientation_strength=features.get("orientation_strength"),
            gray=features["gray"],
        ),
        frequency=compute_frequency_descriptor(features["gray"]),
    )
