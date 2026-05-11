"""
Tactile feature mapping: derive roughness, directionality, frequency descriptors
from visual preprocessed feature maps.

GLCM features (ASM/Energy, Contrast, Homogeneity, Correlation) are computed
at multiple distances and angles, providing both scalar summaries and per-angle
directional profiles for orientation-aware processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from skimage.feature import graycomatrix, graycoprops

import numpy as np


@dataclass
class TactileDescriptor:
    roughness: float        # 0.0 (smooth) – 1.0 (rough)
    directionality: float   # 0.0 (isotropic) – 1.0 (strongly directional)
    frequency: float        # normalized dominant spatial frequency


@dataclass
class GLCMFeatures:
    """Scalar GLCM summaries (averaged across distances and angles)."""
    asm: float          # Angular Second Moment / Energy — texture regularity (0-1)
    contrast: float     # Intensity contrast — directly maps to bump depth
    homogeneity: float  # Local uniformity
    correlation: float  # Linear dependency of pixel pairs
    energy: float       # sqrt(ASM)


@dataclass
class GLCMDirectionalProfile:
    """Per-angle GLCM features for orientation detection.

    Each array has shape (n_angles,) — one value per angle after averaging
    across distances.
    """
    angles_rad: np.ndarray
    contrast_per_angle: np.ndarray
    correlation_per_angle: np.ndarray
    homogeneity_per_angle: np.ndarray
    energy_per_angle: np.ndarray

    @property
    def dominant_angle_rad(self) -> float:
        """Angle with highest contrast (most pronounced texture direction)."""
        return float(self.angles_rad[np.argmax(self.contrast_per_angle)])

    @property
    def dominant_angle_deg(self) -> float:
        return float(np.degrees(self.dominant_angle_rad))

    @property
    def contrast_coherence(self) -> float:
        """0=isotropic, 1=fully directional (based on contrast variance)."""
        mean_c = float(self.contrast_per_angle.mean())
        if mean_c < 1e-8:
            return 0.0
        return float(np.clip(
            np.std(self.contrast_per_angle) / (mean_c + 1e-8), 0.0, 1.0
        ))

    @property
    def correlation_coherence(self) -> float:
        """0=isotropic, 1=fully directional (based on correlation variance)."""
        vals = self.correlation_per_angle
        mean_c = float(vals.mean())
        if mean_c < 1e-8:
            return 0.0
        return float(np.clip(
            np.std(vals) / (abs(mean_c) + 1e-8), 0.0, 1.0
        ))


# ---------------------------------------------------------------------------
# GLCM helpers
# ---------------------------------------------------------------------------

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
) -> GLCMFeatures:
    """Compute scalar GLCM feature summaries (averaged across all distances/angles)."""
    gray_q = _to_uint_levels(gray, levels=levels)
    glcm = graycomatrix(
        gray_q, distances=distances, angles=angles,
        levels=levels, symmetric=True, normed=True,
    )
    return GLCMFeatures(
        asm=float(graycoprops(glcm, "ASM").mean()),
        contrast=float(graycoprops(glcm, "contrast").mean()),
        homogeneity=float(graycoprops(glcm, "homogeneity").mean()),
        correlation=float(graycoprops(glcm, "correlation").mean()),
        energy=float(graycoprops(glcm, "energy").mean()),
    )


def compute_glcm_directional_profile(
    gray: np.ndarray,
    distances: tuple[int, ...] = (1, 2, 4),
    n_angles: int = 8,
    levels: int = 16,
) -> GLCMDirectionalProfile:
    """Compute per-angle GLCM features for fine-grained orientation detection.

    Uses n_angles evenly spaced in [0, pi) (GLCM is symmetric).
    Returns per-angle values averaged across distances.
    """
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    gray_q = _to_uint_levels(gray, levels=levels)
    glcm = graycomatrix(
        gray_q, distances=distances, angles=angles.tolist(),
        levels=levels, symmetric=True, normed=True,
    )
    # graycoprops returns (n_distances, n_angles) — average across distances
    return GLCMDirectionalProfile(
        angles_rad=angles,
        contrast_per_angle=graycoprops(glcm, "contrast").mean(axis=0),
        correlation_per_angle=graycoprops(glcm, "correlation").mean(axis=0),
        homogeneity_per_angle=graycoprops(glcm, "homogeneity").mean(axis=0),
        energy_per_angle=graycoprops(glcm, "energy").mean(axis=0),
    )


# ---------------------------------------------------------------------------
# Tactile descriptor computation
# ---------------------------------------------------------------------------

def compute_roughness(glcm: GLCMFeatures) -> float:
    """Roughness from GLCM: high contrast + low homogeneity = rough."""
    rough = (
        0.7 * float(np.clip(glcm.contrast / 8.0, 0.0, 1.0))
        + 0.3 * (1.0 - float(np.clip(glcm.homogeneity, 0.0, 1.0)))
    )
    return float(np.clip(rough, 0.0, 1.0))


def compute_directionality_from_profile(profile: GLCMDirectionalProfile) -> float:
    """Directionality from per-angle contrast variance (0-1)."""
    return profile.contrast_coherence


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
    glcm = compute_glcm_features(features["gray"])
    profile = compute_glcm_directional_profile(features["gray"])
    return TactileDescriptor(
        roughness=compute_roughness(glcm),
        directionality=compute_directionality_from_profile(profile),
        frequency=compute_frequency_descriptor(features["gray"]),
    )
