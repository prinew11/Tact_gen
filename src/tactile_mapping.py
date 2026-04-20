"""
Tactile feature mapping: derive roughness, directionality, frequency descriptors
from visual preprocessed feature maps.
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


def _to_uint_levels(gray: np.ndarray, levels: int = 16) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    gray = gray - gray.min()
    if gray.max() > 0:
        gray = gray / gray.max()
    gray_q = np.clip((gray * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    return gray_q

def compute_glcm_features(gray: np.ndarray, 
                          distances:tuple[int,...]=(1,2,4), 
                          angles:tuple[float, ...] = (0.0, np.pi/4, np.pi/2, 3*np.pi/4),
                          levels:int=16) -> dict[str, float]:
    gray_q = _to_uint_levels(gray, levels=levels)
    glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, "contrast"),
        "correlation": graycoprops(glcm, "correlation"),
        "homogeneity": graycoprops(glcm, "homogeneity"),
        "energy": graycoprops(glcm, "energy"),
    }




def compute_roughness(glcm_features:dict[str,np.array]) -> float:
    contrast = float(glcm_features["contrast"].mean())
    homogeneity = float(glcm_features["homogeneity"].mean())
    rough = 0.7 * float(np.clip(contrast/8.0, 0.0, 1.0)) + 0.3 * (1.0 - np.clip(homogeneity, 0.0, 1.0))
    return float(np.clip(rough, 0.0, 1.0))


def compute_directionality(glcm_features:dict[str,np.array]) -> float:
    contrast = glcm_features["contrast"]
    per_angle_contrast = contrast.mean(axis=0)  # (n_angles,)
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
    # weighted mean radius
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
