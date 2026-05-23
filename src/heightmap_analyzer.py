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
    valley_std: float = 0.0
    ridge_std: float = 0.0
    # Region classification (Type A = contour, Type B = directional stripe)
    region_b_fraction: float = 0.0       # fraction of area classified as Type B
    region_b_angle_deg: float = 0.0      # dominant grain angle in Type B regions
    region_b_mask: np.ndarray | None = None  # float32 [0,1], 1=Type B


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

    valley_mask_2d = hf < np.percentile(flat, 25)
    ridge_mask_2d = hf > np.percentile(flat, 75)
    valley_std = float(hf[valley_mask_2d].std()) if valley_mask_2d.any() else 0.0
    ridge_std = float(hf[ridge_mask_2d].std()) if ridge_mask_2d.any() else 0.0

    # ── Region classification: Type A (contour) vs Type B (directional stripe) ──
    # Type B = highly anisotropic structure tensor (λ2/λ1 ≈ 0)
    #   This means gradient points in ONE direction only → parallel stripes
    # Type A = more isotropic structure tensor (λ2/λ1 > 0)
    #   This means gradient varies in multiple directions → contours, knots
    local_win = max(h // 16, 16)
    if local_win % 2 == 0:
        local_win += 1

    import cv2
    # Local structure tensor
    J00_loc = cv2.blur(gx * gx, (local_win, local_win))
    J01_loc = cv2.blur(gx * gy, (local_win, local_win))
    J11_loc = cv2.blur(gy * gy, (local_win, local_win))
    tr_loc = J00_loc + J11_loc
    det_loc = J00_loc * J11_loc - J01_loc * J01_loc
    disc_loc = np.sqrt(np.maximum(tr_loc ** 2 - 4 * det_loc, 0))
    l1_loc = (tr_loc + disc_loc) / 2
    l2_loc = (tr_loc - disc_loc) / 2

    # Anisotropy ratio: λ2/λ1
    # Near 0 = perfectly directional (parallel stripes)
    # Near 1 = isotropic (equal variation in all directions)
    anisotropy_ratio = (l2_loc / (l1_loc + 1e-8)).astype(np.float32)

    # Threshold: λ2/λ1 < 0.15 means >85% of variation is in ONE direction = Type B
    aniso_thresh = 0.15
    # Gradient magnitude threshold: use Otsu's method to find natural bimodal split.
    # For stripe+contour images, gradient distribution is bimodal (low=stripes, high=contours).
    # For pure stripes, Otsu picks a low threshold → most pixels classified as Type B.
    # For pure contours, Otsu picks a high threshold → most pixels classified as Type A.
    grad_mag_local = cv2.blur(grad_mag, (local_win, local_win))
    grad_mag_uint8 = np.clip(grad_mag_local / (grad_mag_local.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    otsu_thresh_val, _ = cv2.threshold(grad_mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad_mag_thresh = float(otsu_thresh_val) / 255.0 * (grad_mag_local.max() + 1e-8)

    # Type B mask: highly anisotropic AND low gradient magnitude
    type_b_raw = ((anisotropy_ratio < aniso_thresh) & (grad_mag_local < grad_mag_thresh)).astype(np.float32)

    # Smooth the mask to avoid pixel-level noise.
    # Use smaller sigma to preserve mask strength for large uniform regions.
    smooth_sigma = max(local_win / 6.0, 2.0)
    region_b_mask = gaussian_filter(type_b_raw, sigma=smooth_sigma)
    region_b_mask = np.clip(region_b_mask, 0.0, 1.0).astype(np.float32)

    # Use 0.3 threshold instead of 0.5 to be more inclusive for Type B detection
    region_b_fraction = float((region_b_mask > 0.3).mean())

    # Dominant grain angle in Type B regions
    # Grain direction = perpendicular to gradient = eigenvector of λ2
    # Structure tensor angle gives gradient direction; grain = gradient + 90°
    if region_b_fraction > 0.01:
        b_pixels = region_b_mask > 0.3
        Jxx_b = float(J00_loc[b_pixels].mean())
        Jyy_b = float(J11_loc[b_pixels].mean())
        Jxy_b = float(J01_loc[b_pixels].mean())
        # Gradient direction
        grad_angle = 0.5 * np.arctan2(2.0 * Jxy_b, Jxx_b - Jyy_b + 1e-8)
        # Grain direction = gradient + 90°
        grain_angle = grad_angle + np.pi / 2.0
        region_b_angle_deg = float(np.degrees(grain_angle) % 180)
    else:
        region_b_angle_deg = 0.0

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
        valley_std=valley_std,
        ridge_std=ridge_std,
        region_b_fraction=region_b_fraction,
        region_b_angle_deg=region_b_angle_deg,
        region_b_mask=region_b_mask,
    )


def analysis_to_text(report: HeightmapAnalysis) -> str:
    """Convert analysis to LLM-readable text description."""
    def _level(val, thresholds=(0.33, 0.66)):
        if val < thresholds[0]:
            return "low"
        elif val < thresholds[1]:
            return "moderate"
        return "high"

    region_text = ""
    if report.region_b_fraction > 0.01:
        region_text = f"""
- Region classification:
  Type A (contour-terracing): {100 - report.region_b_fraction * 100:.0f}% of area
  Type B (directional-stepping): {report.region_b_fraction * 100:.0f}% of area, dominant angle {report.region_b_angle_deg:.0f}deg"""
    else:
        region_text = "\n- Region classification: 100% Type A (contour-terracing), no directional stripe regions detected"

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
- Valley/ridge fraction: valley={report.valley_fraction:.0%}, ridge={report.ridge_fraction:.0%}
- Region std: valley={report.valley_std:.4f}, ridge={report.ridge_std:.4f}{region_text}"""


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
    d = asdict(report)
    # region_b_mask is ndarray — convert to summary stats for JSON
    mask = d.pop("region_b_mask", None)
    if mask is not None:
        d["region_b_mask_shape"] = list(mask.shape)
        d["region_b_mask_dtype"] = str(mask.dtype)
    return json.dumps(d, indent=2)
