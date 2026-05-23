"""
Heightmap analysis for agent decision-making.

Produces a structured report that the LLM agent uses to understand
the current state of the heightmap and plan transformations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import gaussian_filter

from tactile_mapping import (
    compute_glcm_features,
    compute_glcm_directional_profile,
    compute_roughness,
    compute_directionality_from_profile,
    compute_frequency_descriptor,
)


@dataclass
class HeightmapAnalysis:
    """Structured analysis report passed to the LLM agent."""
    roughness: float
    directionality: float
    frequency: float
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float
    glcm_correlation: float
    height_mean: float
    height_std: float
    height_range: float
    gradient_magnitude_mean: float
    dominant_gradient_angle_deg: float
    gradient_coherence: float
    low_freq_energy: float
    mid_freq_energy: float
    high_freq_energy: float
    dominant_wavelength_px: float
    histogram_skewness: float
    histogram_bimodality: float
    valley_fraction: float
    ridge_fraction: float


def analyze_heightmap(hf: np.ndarray) -> HeightmapAnalysis:
    """Full analysis of a heightmap. Returns structured report."""
    hf = np.asarray(hf, dtype=np.float32)
    h, w = hf.shape

    glcm = compute_glcm_features(hf)
    profile = compute_glcm_directional_profile(hf)
    roughness = compute_roughness(glcm)
    directionality = compute_directionality_from_profile(profile)
    frequency = compute_frequency_descriptor(hf)

    height_mean = float(hf.mean())
    height_std = float(hf.std())
    height_range = float(hf.max() - hf.min())

    gy, gx = np.gradient(hf)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_magnitude_mean = float(grad_mag.mean())
    grad_angle = np.arctan2(gy, gx)
    dominant_gradient_angle_deg = float(np.degrees(
        np.arctan2(np.mean(np.sin(grad_angle)), np.mean(np.cos(grad_angle)))
    ) % 180)

    sigma = max(h // 16, 2)
    J00 = gaussian_filter(gx * gx, sigma)
    J01 = gaussian_filter(gx * gy, sigma)
    J11 = gaussian_filter(gy * gy, sigma)
    trace = J00 + J11
    det = J00 * J11 - J01 * J01
    discriminant = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))
    lambda1 = (trace + discriminant) / 2
    lambda2 = (trace - discriminant) / 2
    coherence = float(np.mean((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)))

    fft = np.fft.fft2(hf)
    mag = np.abs(np.fft.fftshift(fft))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(cy, cx)
    total_energy = float(mag.sum()) + 1e-8
    low_freq_energy = float(mag[radius < 0.1 * max_r].sum()) / total_energy
    mid_freq_energy = float(mag[(radius >= 0.1 * max_r) & (radius < 0.4 * max_r)].sum()) / total_energy
    high_freq_energy = float(mag[radius >= 0.4 * max_r].sum()) / total_energy

    mag_flat = mag.flatten()
    radius_flat = radius.flatten()
    if mag_flat.sum() > 0:
        dominant_wavelength_px = float(1.0 / (np.average(1.0 / (radius_flat + 1), weights=mag_flat) / max_r + 1e-8))
    else:
        dominant_wavelength_px = float(max_r)

    flat = hf.flatten()
    mean = flat.mean()
    std = flat.std()
    if std > 1e-8:
        histogram_skewness = float(np.mean(((flat - mean) / std) ** 3))
    else:
        histogram_skewness = 0.0

    if std > 1e-8:
        skew = histogram_skewness
        kurt = float(np.mean(((flat - mean) / std) ** 4))
        bc = (skew ** 2 + 1) / (kurt + 1e-8)
        histogram_bimodality = float(np.clip(bc, 0, 2))
    else:
        histogram_bimodality = 0.0

    valley_fraction = float((flat < np.percentile(flat, 25)).mean())
    ridge_fraction = float((flat > np.percentile(flat, 75)).mean())

    return HeightmapAnalysis(
        roughness=roughness,
        directionality=directionality,
        frequency=frequency,
        glcm_contrast=glcm.contrast,
        glcm_homogeneity=glcm.homogeneity,
        glcm_energy=glcm.energy,
        glcm_correlation=glcm.correlation,
        height_mean=height_mean,
        height_std=height_std,
        height_range=height_range,
        gradient_magnitude_mean=gradient_magnitude_mean,
        dominant_gradient_angle_deg=dominant_gradient_angle_deg,
        gradient_coherence=coherence,
        low_freq_energy=low_freq_energy,
        mid_freq_energy=mid_freq_energy,
        high_freq_energy=high_freq_energy,
        dominant_wavelength_px=dominant_wavelength_px,
        histogram_skewness=histogram_skewness,
        histogram_bimodality=histogram_bimodality,
        valley_fraction=valley_fraction,
        ridge_fraction=ridge_fraction,
    )


def analysis_to_text(report: HeightmapAnalysis) -> str:
    """Convert analysis to LLM-readable text description."""
    def _level(val, thresholds=(0.33, 0.66)):
        if val < thresholds[0]:
            return "low"
        elif val < thresholds[1]:
            return "moderate"
        return "high"

    return f"""## Heightmap Analysis
- Roughness: {report.roughness:.2f} ({_level(report.roughness)})
- Directionality: {report.directionality:.2f} ({_level(report.directionality)})
- Frequency: {report.frequency:.2f} ({_level(report.frequency)})
- GLCM: contrast={report.glcm_contrast:.3f}, homogeneity={report.glcm_homogeneity:.3f}, energy={report.glcm_energy:.3f}
- Height: mean={report.height_mean:.3f}, std={report.height_std:.3f}, range={report.height_range:.3f}
- Gradient: mean_mag={report.gradient_magnitude_mean:.4f}, angle={report.dominant_gradient_angle_deg:.1f}deg, coherence={report.gradient_coherence:.2f}
- Frequency bands: low={report.low_freq_energy:.0%}, mid={report.mid_freq_energy:.0%}, high={report.high_freq_energy:.0%}
- Dominant wavelength: {report.dominant_wavelength_px:.1f}px
- Histogram: skewness={report.histogram_skewness:.2f}, bimodality={report.histogram_bimodality:.2f}
- Valley/ridge fraction: valley={report.valley_fraction:.0%}, ridge={report.ridge_fraction:.0%}"""


def compare_analyses(before: HeightmapAnalysis, after: HeightmapAnalysis) -> str:
    """Produce a diff description: what changed and by how much."""
    lines = ["## Before → After Comparison"]
    for field_name in [
        "roughness", "directionality", "frequency",
        "height_std", "height_range", "gradient_magnitude_mean", "gradient_coherence",
    ]:
        b = getattr(before, field_name)
        a = getattr(after, field_name)
        delta = a - b
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        lines.append(f"- {field_name}: {b:.3f} {arrow} {a:.3f} (Δ={delta:+.3f})")
    return "\n".join(lines)


def analysis_to_json(report: HeightmapAnalysis) -> str:
    """Serialize analysis to JSON string."""
    return json.dumps(asdict(report), indent=2)
