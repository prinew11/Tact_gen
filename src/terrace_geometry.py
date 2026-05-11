"""
Terrace pipeline: heightfield preprocessing and watertight terrace STL generation.

Preprocessing (formerly machining_filter.py):
  Normalize, downsample, prune high-frequency noise, suppress narrow recesses,
  smooth, terrace-quantize, and optionally compress height for slope limits.
  All operations are deterministic.  Hard constraint: any groove narrower than
  tool_diameter_mm (default 6 mm) cannot be machined and is suppressed.

Geometry (contour-based terrace mesh):
  Quantizes the preprocessed heightfield into N discrete levels and builds a
  watertight stepped mesh with flat horizontal top faces, 90-degree vertical
  risers, flat bottom, and vertical outer-perimeter walls.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion, label as ndimage_label, uniform_filter

# ---------------------------------------------------------------------------
# Preprocessing dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MachiningFilterConfig:
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_radius_mm: float = 3.0        # 6 mm diameter ball-end mill → radius = 3 mm
    max_slope_deg: float = 30.0
    face_limit: int = 500_000
    target_min_feature_factor: float = 1.5
    gaussian_sigma_px: float = 2.0     # prune sigma: 2px = light denoise only
    max_iterations: int = 10
    target_resolution_mode: str = "auto"   # "auto" | "fixed"
    # 0 = auto-compute from physical_size / tool_diameter;
    # 1 = no terracing; ≥2 = explicit step count.
    terrace_steps: int = 0
    # When True: skip Gaussian smoothing, skip height compression.
    # Only normalize + downsample + mild noise prune + morphological opening.
    terrace_mode: bool = False

@dataclass
class MachiningFilterReport:
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    pixel_size_mm: float = 0.0
    estimated_face_count: int = 0
    max_slope_deg_before: float = 0.0
    max_slope_deg_after: float = 0.0
    min_feature_target_mm: float = 0.0
    min_feature_estimate_mm: float = 0.0
    height_scale_applied: float = 1.0
    smoothing_sigma_px: float = 0.0
    morph_radius_px: float = 0.0
    terrace_steps_applied: int = 0
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Preprocessing helper functions
# ---------------------------------------------------------------------------

def normalize_heightfield(hf: np.ndarray) -> np.ndarray:
    """Percentile-based normalization: clip extremes, rescale to [0, 1]."""
    hf = hf.astype(np.float32)
    p_low  = np.percentile(hf, 2)
    p_high = np.percentile(hf, 98)
    hf = np.clip(hf, p_low, p_high)
    hf = (hf - p_low) / (p_high - p_low + 1e-8)
    return hf.astype(np.float32)


def remove_global_trend(hf: np.ndarray, sigma_px: float = 150.0) -> np.ndarray:
    """
    Remove global brightness gradient by subtracting a large-scale Gaussian background.

    sigma=150px (~30mm physical) is far larger than any texture feature,
    so only the global tilt is captured.  After subtraction, only local
    relative heights (knot depth, grain peaks) remain.

    Returns (H, W) float32 re-normalized to [0, 1].
    """
    hf = hf.astype(np.float32)
    background = gaussian_filter(hf, sigma=sigma_px)
    residual = hf - background
    # Re-normalize to [0, 1]
    r_min, r_max = float(residual.min()), float(residual.max())
    if r_max - r_min > 1e-8:
        residual = (residual - r_min) / (r_max - r_min)
    else:
        residual = np.full_like(residual, 0.5)
    return residual.astype(np.float32)


def remap_height_distribution(hf: np.ndarray) -> np.ndarray:
    """
    Non-linear height remapping: percentile clip + histogram equalisation.

    1. Clip [p2, p98] → [0.05, 0.95] (preserve headroom for terrace risers).
    2. Histogram equalisation: spread pixel values uniformly across [0, 1]
       so every terrace level gets roughly equal pixel count.

    Does NOT change spatial feature positions — only redistributes heights.
    """
    hf = hf.astype(np.float64)
    p_low  = np.percentile(hf, 2)
    p_high = np.percentile(hf, 98)
    if p_high - p_low < 1e-8:
        return np.full_like(hf, 0.5, dtype=np.float32)

    # Step 1: clip and map to [0.05, 0.95]
    hf_clipped = np.clip(hf, p_low, p_high)
    hf_norm = (hf_clipped - p_low) / (p_high - p_low)  # [0, 1]
    hf_norm = 0.05 + hf_norm * 0.90  # [0.05, 0.95]

    # Step 2: histogram equalisation via CDF
    n_bins = 1024
    hist, bin_edges = np.histogram(hf_norm, bins=n_bins, range=(0.0, 1.0))
    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf[-1]  # normalise to [0, 1]

    # Map each pixel through the CDF
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # np.interp: for each pixel value, look up its CDF value
    hf_eq = np.interp(hf_norm, bin_centers, cdf).astype(np.float32)

    return np.clip(hf_eq, 0.0, 1.0).astype(np.float32)

def compute_pixel_size_mm(physical_size_mm: float, resolution: int) -> float:
    """Physical distance per pixel (edge-to-edge: N-1 intervals)."""
    return physical_size_mm / max(resolution - 1, 1)

def estimate_face_count(resolution: int) -> int:
    """
    Estimate triangle count for a watertight heightfield mesh including side walls.
    Formula: 4*(N-1)*(N+1) = top quads + bottom quads + 4 side wall strips.
    At N=353 → 498,816 faces, safely under the 500k Fusion limit.
    """
    n = resolution - 1
    return 4 * n * (n + 2)

def estimate_target_resolution_for_face_budget(
    face_limit: int,
    current_res: int,
) -> int:
    """
    Return the largest resolution R such that estimate_face_count(R) <= face_limit.
    Returns current_res unchanged if already within budget.
    """
    if estimate_face_count(current_res) <= face_limit:
        return current_res
    r_max = int(math.floor(math.sqrt(face_limit / 4.0 + 1.0)))
    return max(r_max, 2)

def estimate_slope_map_deg(
    heightfield: np.ndarray,
    pixel_size_mm: float,
    max_height_mm: float,
) -> np.ndarray:
    """Compute per-pixel slope in degrees."""
    z = heightfield.astype(np.float32) * max_height_mm
    gy, gx = np.gradient(z, pixel_size_mm)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    return np.degrees(slope_rad)

def smooth_by_tool_scale(
    heightfield: np.ndarray,
    tool_radius_mm: float,
    pixel_size_mm: float,
    sigma_override_px: float = 0.0,
    terrace_steps: int = 1,
    riser_sigma_px: float = 0.0,
) -> np.ndarray:
    """
    Gaussian blur at sigma = tool_radius_mm / pixel_size_mm, then optionally
    quantize into discrete height levels to produce a terraced topology.

    Args:
        terrace_steps: number of discrete height levels.  1 = no terracing.
        riser_sigma_px: Gaussian sigma for step-edge softening.  0 = auto.
    """
    sigma = sigma_override_px if sigma_override_px > 0.0 else max(
        tool_radius_mm / pixel_size_mm, 1.0
    )
    hf = gaussian_filter(heightfield.astype(np.float32), sigma=sigma)

    if terrace_steps > 1:
        n = terrace_steps - 1
        hf = np.round(hf * n) / n
        r_sigma = riser_sigma_px if riser_sigma_px > 0.0 else max(sigma * 0.5, 1.0)
        hf = gaussian_filter(hf, sigma=r_sigma)
        hf = np.clip(hf, 0.0, 1.0)

    return hf

def apply_terracing(
    heightfield: np.ndarray,
    terrace_steps: int,
    riser_sigma_px: float = 1.0,
) -> np.ndarray:
    """
    Quantize a smooth heightfield into discrete terrace levels and soften
    the step risers.  Runs after morphological opening so the opening result
    blends smoothly before discrete levels are applied.
    """
    if terrace_steps <= 1:
        return heightfield.copy()
    n = terrace_steps - 1
    hf = np.round(heightfield.astype(np.float32) * n) / n
    hf = gaussian_filter(hf, sigma=max(riser_sigma_px, 1.0))
    return np.clip(hf, 0.0, 1.0)

def suppress_narrow_recesses(
    heightfield: np.ndarray,
    tool_radius_px: float,
    orientation_map: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Morphological opening on the inverted heightfield removes grooves and
    concavities narrower than tool_radius_px * 2 pixels (= tool diameter).

    When orientation_map is provided, uses anisotropic (elliptical) structuring
    elements oriented along the local texture direction — avoids creating
    circular arc artefacts at stripe endpoints.

    When mask is provided, those pixels (True = protect) are excluded from
    processing and restored from the original after opening.
    """
    H, W = heightfield.shape
    r = max(int(math.ceil(tool_radius_px)), 1)
    short_axis = max(r // 4, 1)  # 8px when r=16 (tool_radius_px≈16)

    inv = (1.0 - heightfield).astype(np.float32)

    if orientation_map is not None:
        # Anisotropic: rectangular structuring element per orientation bin.
        # Flat-end mill → rectangular footprint (not elliptical).
        angles_deg = np.rad2deg(orientation_map) % 180.0
        bins = [0, 45, 90, 135]
        inv_opened = np.empty_like(inv)

        for ang in bins:
            # Long rectangle: long axis = 2*r+1, short axis = 2*short_axis+1
            rect_local = np.ones((2 * r + 1, 2 * short_axis + 1), dtype=bool)

            # Rotate rectangle by ang degrees via coordinate mapping
            theta = np.deg2rad(ang)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            yi, xi = np.ogrid[-r:r + 1, -r:r + 1]
            # Map back to local rectangle coordinates
            xr = cos_t * xi + sin_t * yi
            yr = -sin_t * xi + cos_t * yi
            # Check if mapped coords fall inside the unrotated rectangle
            footprint = (np.abs(xr) <= r) & (np.abs(yr) <= short_axis)
            footprint = footprint.astype(bool)

            # Pixels whose orientation falls in this bin (±22.5°)
            if ang == 0:
                bin_mask = (angles_deg < 22.5) | (angles_deg >= 157.5)
            else:
                bin_mask = (angles_deg >= ang - 22.5) & (angles_deg < ang + 22.5)

            if not np.any(bin_mask):
                continue

            eroded = grey_erosion(inv, footprint=footprint)
            opened = grey_dilation(eroded, footprint=footprint)
            inv_opened[bin_mask] = opened[bin_mask]

        # Pixels not in any bin (shouldn't happen) — use rectangular fallback
        all_bins = np.zeros((H, W), dtype=bool)
        for ang in bins:
            if ang == 0:
                all_bins |= (angles_deg < 22.5) | (angles_deg >= 157.5)
            else:
                all_bins |= (angles_deg >= ang - 22.5) & (angles_deg < ang + 22.5)
        if not np.all(all_bins):
            rect = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
            fallback = grey_dilation(grey_erosion(inv, footprint=rect), footprint=rect)
            inv_opened[~all_bins] = fallback[~all_bins]
    else:
        # Isotropic: flat-end mill → rectangular structuring element
        rect = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
        inv_opened = grey_dilation(grey_erosion(inv, footprint=rect), footprint=rect)

    result = np.clip(1.0 - inv_opened, 0.0, 1.0).astype(np.float32)

    # Restore protected regions from original
    if mask is not None:
        result[mask] = heightfield[mask]

    return result

def prune_high_frequency_content(
    heightfield: np.ndarray,
    _tool_radius_mm: float = 0.0,
    _pixel_size_mm: float = 0.0,
    sigma_px: float = 2.0,
) -> np.ndarray:
    """
    Remove single-pixel noise before terracing.
    Uses a mild Gaussian (default sigma=2px) to suppress diffusion noise
    without merging adjacent stripe features.
    """
    return gaussian_filter(heightfield.astype(np.float32), sigma=sigma_px)

def compress_height_for_slope(
    heightfield: np.ndarray,
    max_slope_deg: float,
    pixel_size_mm: float,
    max_height_mm: float,
    max_iters: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Binary-search a height scale factor in (0, 1] so the physical slope stays
    within max_slope_deg.  Returns (heightfield_unchanged, scale_applied).
    The heightfield values are NOT modified — multiply max_height_mm by
    scale_applied when passing to geometry.heightfield_to_mesh.
    Returns scale=1.0 if the input already satisfies the slope limit.
    """
    slope_map = estimate_slope_map_deg(heightfield, pixel_size_mm, max_height_mm)
    if float(slope_map.max()) <= max_slope_deg:
        return heightfield.copy(), 1.0

    lo, hi = 0.0, 1.0
    best_scale = lo

    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        effective_height = max_height_mm * mid
        slope_map = estimate_slope_map_deg(heightfield, pixel_size_mm, effective_height)
        if float(slope_map.max()) <= max_slope_deg:
            best_scale = mid
            lo = mid
        else:
            hi = mid

    return heightfield.copy(), best_scale

def detect_knot_holes(
    heightfield: np.ndarray,
    depth_threshold: float = 0.08,
    circularity_threshold: float = 0.6,
    max_area_px: int = 80,
) -> np.ndarray:
    """
    Detect deep, circular depressions (knot holes) in the heightfield.

    Uses connected component analysis on the inverted heightfield to find
    dark regions that are:
      - deeper than depth_threshold (normalised [0,1])
      - more circular than circularity_threshold (4π·area / perimeter²)
      - smaller than max_area_px pixels

    Returns (H, W) bool mask — True where knot holes are detected.
    """
    hf = heightfield.astype(np.float32)
    mean_val = float(hf.mean())

    # Threshold: pixels significantly below the mean
    thresh_val = mean_val - depth_threshold
    binary = hf < thresh_val

    # Label connected components
    labeled, num_features = ndimage_label(binary)

    if num_features == 0:
        return np.zeros(hf.shape, dtype=bool)

    mask = np.zeros(hf.shape, dtype=bool)

    for i in range(1, num_features + 1):
        component = labeled == i
        area = int(component.sum())

        if area > max_area_px or area < 3:
            continue

        # Circularity: 4π·area / perimeter²
        # Approximate perimeter via boundary pixel count
        eroded = grey_erosion(component.astype(np.uint8), size=3)
        boundary = component & ~eroded.astype(bool)
        perimeter = max(int(boundary.sum()), 1)
        circularity = 4.0 * np.pi * area / (perimeter ** 2)

        if circularity >= circularity_threshold:
            mask |= component

    return mask

def filter_heightfield_for_machining(
    heightfield: np.ndarray,
    config: MachiningFilterConfig | None = None,
    orientation_map: np.ndarray | None = None,
    source_image: np.ndarray | None = None,
) -> tuple[np.ndarray, MachiningFilterReport]:
    """
    Apply all machining constraints to a heightfield in a deterministic sequence:
      1. Normalize to [0, 1].
      2. Downsample if needed to satisfy face_limit (auto mode).
      3. Prune sub-tool-scale high-frequency noise.
      3b. Detect and mask knot holes (protected from prune & ADC).
      3c. Modulate local contrast for fine-texture enhancement.
      4. Morphological opening to suppress sub-tool-diameter recesses.
      5. Slope measurement (before compression).
      6. Gaussian smoothing and terracing.
      7. Iterative height compression to satisfy max_slope_deg.
      8. Final slope measurement and report.
    """
    if config is None:
        config = MachiningFilterConfig()

    report = MachiningFilterReport()

    if heightfield.ndim != 2:
        raise ValueError(f"Expected 2-D heightfield, got shape {heightfield.shape}")
    if heightfield.shape[0] != heightfield.shape[1]:
        report.issues.append(
            f"Non-square heightfield {heightfield.shape} — forcing square output"
        )

    report.input_shape = (heightfield.shape[0], heightfield.shape[1])
    hf = normalize_heightfield(heightfield)

    # Step 2: Resolution targeting
    current_res = hf.shape[0]
    if config.target_resolution_mode == "auto":
        target_res = estimate_target_resolution_for_face_budget(
            config.face_limit, current_res
        )
        if target_res < current_res:
            hf = cv2.resize(
                hf, (target_res, target_res), interpolation=cv2.INTER_AREA
            )
            report.recommendations.append(
                f"Downsampled {current_res}→{target_res} px to satisfy face budget"
            )

    report.output_shape = (hf.shape[0], hf.shape[1])
    res = hf.shape[0]

    # Resize orientation map to match (possibly downsampled) heightfield
    if orientation_map is not None and orientation_map.shape[0] != res:
        orientation_map = cv2.resize(orientation_map, (res, res),
                                     interpolation=cv2.INTER_LINEAR)
    pixel_size_mm = compute_pixel_size_mm(config.physical_size_mm, res)
    report.pixel_size_mm = pixel_size_mm
    report.estimated_face_count = estimate_face_count(res)

    slope_before_map = estimate_slope_map_deg(hf, pixel_size_mm, config.max_height_mm)
    report.max_slope_deg_before = float(slope_before_map.max())

    # Step 3.5: Resolve terrace step count
    if config.terrace_steps == 0:
        tool_diameter_mm = config.tool_radius_mm * 2.0
        actual_terrace_steps = max(
            2,
            round(config.physical_size_mm / (tool_diameter_mm * config.target_min_feature_factor)),
        )
    else:
        actual_terrace_steps = config.terrace_steps
    report.terrace_steps_applied = actual_terrace_steps

    # Step 3.6: Prune sub-tool-scale noise before morphological opening
    hf = prune_high_frequency_content(hf, config.tool_radius_mm, pixel_size_mm)
    hf = np.clip(hf, 0.0, 1.0)

    # Step 3b: Detect knot holes — protect from prune & ADC
    knot_mask = detect_knot_holes(hf)

    # Step 4: Suppress sub-tool-diameter recesses (anisotropic if orientation provided)
    tool_radius_px = config.tool_radius_mm / pixel_size_mm
    report.morph_radius_px = tool_radius_px
    hf = suppress_narrow_recesses(hf, tool_radius_px,
                                  orientation_map=orientation_map,
                                  mask=knot_mask)
    hf = np.clip(hf, 0.0, 1.0)
    tool_diameter_mm = config.tool_radius_mm * 2.0
    report.recommendations.append(
        f"Sub-{tool_diameter_mm:.0f} mm recesses suppressed via morphological "
        f"opening (tool diameter {tool_diameter_mm:.1f} mm)"
    )

    if config.terrace_mode:
        report.smoothing_sigma_px = 0.0
        report.height_scale_applied = 1.0
        report.recommendations.append(
            "Terrace mode: Gaussian smoothing and slope compression skipped. "
            "Quantisation is handled by heightfield_to_terrace_mesh()."
        )
    else:
        # Step 5: Gaussian smoothing
        sigma_px = (
            config.gaussian_sigma_px
            if config.gaussian_sigma_px > 0.0
            else max(config.tool_radius_mm / pixel_size_mm, 1.0)
        )
        report.smoothing_sigma_px = sigma_px
        hf = smooth_by_tool_scale(
            hf, config.tool_radius_mm, pixel_size_mm,
            sigma_override_px=sigma_px,
            terrace_steps=1,
        )
        hf = np.clip(hf, 0.0, 1.0)

        # Step 5b: Terracing with slope-calibrated riser sigma
        if actual_terrace_steps > 1 and config.max_height_mm > 0 and config.max_slope_deg > 0:
            step_height_mm = config.max_height_mm / (actual_terrace_steps - 1)
            riser_sigma_px = max(
                step_height_mm / (
                    math.tan(math.radians(config.max_slope_deg))
                    * pixel_size_mm
                    * math.sqrt(2 * math.pi)
                ),
                1.0,
            )
        else:
            riser_sigma_px = max(sigma_px * 0.5, 1.0)
        hf = apply_terracing(hf, actual_terrace_steps, riser_sigma_px)
        hf = np.clip(hf, 0.0, 1.0)

        # Step 6: Height compression
        hf, scale = compress_height_for_slope(
            hf,
            config.max_slope_deg * 0.97,
            pixel_size_mm,
            config.max_height_mm,
            config.max_iterations,
        )
        report.height_scale_applied = scale
        hf = (hf * scale).astype(np.float32)
        if scale < 0.95:
            report.recommendations.append(
                f"Height compressed to {scale:.2f}× "
                f"({config.max_height_mm * scale:.2f} mm effective). "
                "Scale is embedded in the saved heightfield — use max_height_mm as-is."
            )

    hf = np.clip(hf, 0.0, 1.0)

    slope_after = estimate_slope_map_deg(hf, pixel_size_mm, config.max_height_mm)
    report.max_slope_deg_after = float(slope_after.max())

    report.min_feature_target_mm = tool_diameter_mm
    report.min_feature_estimate_mm = pixel_size_mm * tool_radius_px * 2.0

    if not config.terrace_mode and report.max_slope_deg_after > config.max_slope_deg:
        report.issues.append(
            f"Slope {report.max_slope_deg_after:.1f}° still exceeds limit "
            f"{config.max_slope_deg}° after {config.max_iterations} iterations. "
            "Increase max_iterations or reduce max_height_mm in GeometryConfig."
        )
    if report.estimated_face_count > config.face_limit:
        report.issues.append(
            f"Estimated face count {report.estimated_face_count:,} "
            f"exceeds limit {config.face_limit:,}."
        )
    if pixel_size_mm > tool_diameter_mm:
        report.issues.append(
            f"Pixel size {pixel_size_mm:.2f} mm > tool diameter {tool_diameter_mm:.1f} mm "
            "— resolution is too coarse to reliably detect sub-feature violations."
        )

    report.passed = len(report.issues) == 0
    return hf, report

def save_report_json(report: MachiningFilterReport, out_path: str | Path) -> None:
    """Serialize MachiningFilterReport to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input_shape": list(report.input_shape),
        "output_shape": list(report.output_shape),
        "pixel_size_mm": report.pixel_size_mm,
        "estimated_face_count": report.estimated_face_count,
        "max_slope_deg_before": report.max_slope_deg_before,
        "max_slope_deg_after": report.max_slope_deg_after,
        "min_feature_target_mm": report.min_feature_target_mm,
        "min_feature_estimate_mm": report.min_feature_estimate_mm,
        "height_scale_applied": report.height_scale_applied,
        "smoothing_sigma_px": report.smoothing_sigma_px,
        "morph_radius_px": report.morph_radius_px,
        "terrace_steps_applied": report.terrace_steps_applied,
        "passed": report.passed,
        "issues": report.issues,
        "recommendations": report.recommendations,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved machining filter report: {out_path}")

# ---------------------------------------------------------------------------
# Terrace geometry dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TerraceConfig:
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    base_thickness_mm: float = 2.0
    terrace_steps: int = 5          # number of discrete height levels
    tool_diameter_mm: float = 6.0   # 6 mm ball-end mill — primary hard rule
    mesh_resolution: int = 256      # resize heightfield to this before building mesh
    face_limit: int = 500_000       # warn if exceeded after build

@dataclass
class TerraceReport:
    levels_used: int = 0
    face_count: int = 0
    vertex_count: int = 0
    watertight: bool = False
    min_recess_enforced_mm: float = 0.0
    issues: list[str] = field(default_factory=list)
    passes: bool = False

# ---------------------------------------------------------------------------
# Internal mesh-builder helpers
# ---------------------------------------------------------------------------

def _quantize(heightfield: np.ndarray, n_levels: int) -> np.ndarray:
    """Quantize [0, 1] heightfield to integer labels 0 .. n_levels-1 (sharp, no blur)."""
    clipped = np.clip(heightfield, 0.0, 1.0)
    labels = np.floor(clipped * n_levels).astype(np.int32)
    return np.clip(labels, 0, n_levels - 1)

def _resolve_checkerboard(labels: np.ndarray) -> np.ndarray:
    """
    Fix 2x2 checkerboard saddle patterns that cause 4 riser faces to share a
    single vertical edge (non-manifold).

    Two patterns exist:
      a,b / b,a  (a<b) → raise (pr,pc) and (pr+1,pc+1) to b
      a,b / b,a  (a>b) → raise (pr,pc+1) and (pr+1,pc) to a

    Iterates to convergence. Every iteration strictly raises at least one label,
    so the loop always terminates.
    """
    result = labels.copy()
    h, w = result.shape
    changed = True
    while changed:
        changed = False
        for pr in range(h - 1):
            for pc in range(w - 1):
                a = int(result[pr,     pc])
                b = int(result[pr,     pc + 1])
                c = int(result[pr + 1, pc])
                d = int(result[pr + 1, pc + 1])
                if a == d and b == c and a != b:
                    hi = max(a, b)
                    if a < b:
                        result[pr,     pc] = hi
                        result[pr + 1, pc + 1] = hi
                    else:
                        result[pr,     pc + 1] = hi
                        result[pr + 1, pc] = hi
                    changed = True
    return result

def _enforce_min_recess_width(
    labels: np.ndarray,
    tool_radius_px: float,
    n_levels: int,
) -> np.ndarray:
    """
    Fill recessed regions whose XY width is <= tool_diameter_mm (6 mm default).

    For each level L (highest to lowest), the binary mask (labels >= L)
    represents all pixels at level L or above.  Morphological CLOSING fills
    narrow holes in this mask, and any pixel that transitions from below-L
    to at-or-above-L is raised to level L.  Processing top-down prevents
    oscillation.
    """
    result = labels.copy()
    r = max(int(math.ceil(tool_radius_px)), 1)
    yi, xi = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (xi ** 2 + yi ** 2 <= r ** 2).astype(np.uint8)

    for level in range(n_levels - 1, 0, -1):
        above = (result >= level).astype(np.uint8)
        closed = cv2.morphologyEx(above, cv2.MORPH_CLOSE, disk)
        fill = (closed == 1) & (result < level)
        result[fill] = level

    return result

def _z_of_label(
    label: int,
    n_levels: int,
    max_height_mm: float,
    base_mm: float,
) -> float:
    if n_levels <= 1:
        return base_mm
    return label / (n_levels - 1) * max_height_mm + base_mm

# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def heightfield_to_terrace_mesh(
    heightfield: np.ndarray,
    config: TerraceConfig | None = None,
) -> tuple[trimesh.Trimesh, TerraceReport]:
    """
    Build a watertight stepped-terrace mesh from a [0, 1] float heightfield.

    The resulting STL has:
      - Flat horizontal top faces at each discrete level.
      - 90-degree vertical risers at every level boundary.
      - Flat bottom at z = 0, fan-triangulated to match the perimeter.
      - Vertical outer-perimeter walls.

    Returns (mesh, TerraceReport).
    """
    if config is None:
        config = TerraceConfig()

    report = TerraceReport()
    report.levels_used = config.terrace_steps
    report.min_recess_enforced_mm = config.tool_diameter_mm

    # Resize heightfield to target mesh resolution.
    res = config.mesh_resolution
    if heightfield.shape[0] != res or heightfield.shape[1] != res:
        heightfield = cv2.resize(
            heightfield.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA
        )
    h, w = heightfield.shape  # rows, cols

    px_size = config.physical_size_mm / (w - 1)   # mm per pixel edge
    tool_radius_px = (config.tool_diameter_mm / 2.0) / px_size
    n = config.terrace_steps

    # Step 1: Sharp quantisation — no blur.
    labels = _quantize(heightfield, n)

    # Step 2: Enforce minimum recess width (6 mm hard rule).
    labels = _enforce_min_recess_width(labels, tool_radius_px, n)

    # Step 3: Resolve checkerboard saddle points that produce non-manifold edges.
    labels = _resolve_checkerboard(labels)

    # Step 4: Flip rows so image-top maps to STL back (y=H_mm), not y=0.
    labels = np.flipud(labels)

    # Precompute z heights for each label value.
    z_table = np.array(
        [_z_of_label(lv, n, config.max_height_mm, config.base_thickness_mm)
         for lv in range(n)],
        dtype=np.float64,
    )

    # Coordinate helpers
    def xc(col: int) -> float:
        return col * px_size

    def yr(row: int) -> float:
        return row * px_size

    # ---------------------------------------------------------------------------
    # Allocate vertex / face buffers with a safe upper bound.
    # ---------------------------------------------------------------------------
    max_verts = (h * w * 4
                 + (w - 1) * h * 4
                 + w * (h - 1) * 4
                 + 2 * (h + w) * 4
                 + 2 * (h + w) + 2)
    max_faces = (h * w * 2
                 + (w - 1) * h * 2
                 + w * (h - 1) * 2
                 + 2 * (h + w) * 2
                 + 2 * (h + w))

    vbuf = np.empty((max_verts, 3), dtype=np.float64)
    fbuf = np.empty((max_faces, 3), dtype=np.int64)
    nv = 0
    nf = 0

    def av(x: float, y: float, z: float) -> int:
        nonlocal nv
        vbuf[nv] = (x, y, z)
        i = nv
        nv += 1
        return i

    def at(a: int, b: int, c: int) -> None:
        nonlocal nf
        fbuf[nf] = (a, b, c)
        nf += 1

    def aq(a: int, b: int, c: int, d: int) -> None:
        """Quad a-b-c-d → triangles (a,b,c) and (a,c,d)."""
        at(a, b, c)
        at(a, c, d)

    # ---------------------------------------------------------------------------
    # 3a  Top faces — one flat quad per pixel (normal = +Z).
    #
    #   Winding CCW from +Z → normals +Z.
    # ---------------------------------------------------------------------------
    for pr in range(h):
        for pc in range(w):
            z = z_table[labels[pr, pc]]
            tl = av(xc(pc),     yr(pr),     z)
            tr = av(xc(pc + 1), yr(pr),     z)
            br = av(xc(pc + 1), yr(pr + 1), z)
            bl = av(xc(pc),     yr(pr + 1), z)
            aq(tl, tr, br, bl)

    # ---------------------------------------------------------------------------
    # 3b  Internal vertical risers (horizontal adjacency).
    #
    #   Wall at x = xc(pc+1).  la<lb → normal=-X; la>lb → normal=+X.
    # ---------------------------------------------------------------------------
    for pr in range(h):
        row_labels = labels[pr]
        for pc in range(w - 1):
            la = int(row_labels[pc])
            lb = int(row_labels[pc + 1])
            if la == lb:
                continue
            z_lo = z_table[min(la, lb)]
            z_hi = z_table[max(la, lb)]
            x = xc(pc + 1)
            y0, y1 = yr(pr), yr(pr + 1)
            v0 = av(x, y0, z_lo)
            v1 = av(x, y1, z_lo)
            v2 = av(x, y1, z_hi)
            v3 = av(x, y0, z_hi)
            if la < lb:
                aq(v0, v3, v2, v1)   # normal = -X
            else:
                aq(v0, v1, v2, v3)   # normal = +X

    # ---------------------------------------------------------------------------
    # 3c  Internal vertical risers (vertical adjacency).
    #
    #   Wall at y = yr(pr+1).  la>lb → normal=+Y; la<lb → normal=-Y.
    # ---------------------------------------------------------------------------
    for pr in range(h - 1):
        for pc in range(w):
            la = int(labels[pr, pc])
            lb = int(labels[pr + 1, pc])
            if la == lb:
                continue
            z_lo = z_table[min(la, lb)]
            z_hi = z_table[max(la, lb)]
            y = yr(pr + 1)
            x0, x1 = xc(pc), xc(pc + 1)
            v0 = av(x0, y, z_lo)
            v1 = av(x1, y, z_lo)
            v2 = av(x1, y, z_hi)
            v3 = av(x0, y, z_hi)
            if la > lb:
                aq(v0, v3, v2, v1)   # normal = +Y
            else:
                aq(v0, v1, v2, v3)   # normal = -Y

    # ---------------------------------------------------------------------------
    # Pre-create all z=0 perimeter vertices shared by outer walls and bottom fan.
    # ---------------------------------------------------------------------------
    W_mm = xc(w)
    H_mm = yr(h)

    bot: dict[tuple[int, int], int] = {}
    for pc in range(w + 1):
        bot[(0, pc)] = av(xc(pc), yr(0), 0.0)   # front row
        bot[(h, pc)] = av(xc(pc), yr(h), 0.0)   # back row
    for pr in range(1, h):
        bot[(pr, 0)] = av(xc(0), yr(pr), 0.0)
        bot[(pr, w)] = av(xc(w), yr(pr), 0.0)

    # ---------------------------------------------------------------------------
    # 3d  Outer perimeter walls.
    #
    #   Front (-Y), Back (+Y), Left (-X), Right (+X).
    # ---------------------------------------------------------------------------
    for pc in range(w):
        z = z_table[labels[0, pc]]
        b0, b1 = bot[(0, pc)], bot[(0, pc + 1)]
        v2 = av(xc(pc + 1), yr(0), z)
        v3 = av(xc(pc),     yr(0), z)
        aq(b0, b1, v2, v3)   # -Y

        z = z_table[labels[h - 1, pc]]
        b0, b1 = bot[(h, pc)], bot[(h, pc + 1)]
        v2 = av(xc(pc + 1), yr(h), z)
        v3 = av(xc(pc),     yr(h), z)
        aq(b0, v3, v2, b1)   # +Y

    for pr in range(h):
        z = z_table[labels[pr, 0]]
        b0, b1 = bot[(pr, 0)], bot[(pr + 1, 0)]
        v2 = av(xc(0), yr(pr + 1), z)
        v3 = av(xc(0), yr(pr),     z)
        aq(b0, v3, v2, b1)   # -X

        z = z_table[labels[pr, w - 1]]
        b0, b1 = bot[(pr, w)], bot[(pr + 1, w)]
        v2 = av(xc(w), yr(pr + 1), z)
        v3 = av(xc(w), yr(pr),     z)
        aq(b0, b1, v2, v3)   # +X

    # ---------------------------------------------------------------------------
    # 3e  Bottom face — fan from center, perimeter traversed for normal = -Z.
    # ---------------------------------------------------------------------------
    cx_idx = av(W_mm / 2.0, H_mm / 2.0, 0.0)

    def fan(p_i: int, p_j: int) -> None:
        at(cx_idx, p_i, p_j)

    for pr in range(h):
        fan(bot[(pr, 0)], bot[(pr + 1, 0)])
    for pc in range(w):
        fan(bot[(h, pc)], bot[(h, pc + 1)])
    for pr in range(h):
        fan(bot[(h - pr, w)], bot[(h - 1 - pr, w)])
    for pc in range(w):
        fan(bot[(0, w - pc)], bot[(0, w - 1 - pc)])

    # ---------------------------------------------------------------------------
    # Assemble mesh.
    # ---------------------------------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=vbuf[:nv],
        faces=fbuf[:nf],
        process=True,
    )

    report.face_count = len(mesh.faces)
    report.vertex_count = len(mesh.vertices)
    report.watertight = bool(mesh.is_watertight)
    report.passes = report.watertight

    if not report.watertight:
        report.issues.append(
            "Mesh is not watertight — internal geometry error; "
            "check face winding or level boundary coverage."
        )
    if report.face_count > config.face_limit:
        report.issues.append(
            f"Face count {report.face_count:,} exceeds Fusion limit "
            f"{config.face_limit:,}. Reduce mesh_resolution or terrace_steps."
        )
        report.passes = False

    return mesh, report

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_stl(mesh: trimesh.Trimesh, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"Terrace STL saved: {out_path}  ({len(mesh.faces):,} faces)")

# ---------------------------------------------------------------------------
# Tactile Saliency Guidance
# ---------------------------------------------------------------------------

@dataclass
class SaliencyConfig:
    # Multi-scale FFT
    fft_scales: list[int] = field(default_factory=lambda: [32, 64, 128])
    fft_stride: int = 16
    fft_energy_threshold: float = 0.20

    # Structure tensor
    structure_sigma: float = 10.0       # smoothing for structure tensor (px)
    tool_angle_deg: float = 0.0         # reference toolpath angle (0 = horizontal)

    # Height range sensitivity
    tool_tolerance_mm: float = 0.05     # height range below this → negligible

    # Gaussian smoothing for weight map
    weight_blur_sigma: float = 8.0

    # Terrace allocation
    terrace_steps_high: int = 12
    terrace_steps_low: int = 4
    saliency_threshold_high: float = 0.65
    saliency_threshold_low: float = 0.30

@dataclass
class SaliencyReport:
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_mean: float = 0.0
    high_weight_fraction: float = 0.0
    low_weight_fraction: float = 0.0
    mean_period_px: float = 0.0
    non_periodic_fraction: float = 0.0
    mean_coherence: float = 0.0
    mean_height_range_mm: float = 0.0
    unmachinable_ratio: float = 0.0   # fraction of windows with period < tool_diameter

def compute_local_structure_tensor(
    gray: np.ndarray,
    sigma: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel structure tensor → local orientation, coherence, scale.

    Returns:
        angle_map   : (H, W) float64, orientation in radians
        coherence   : (H, W) float32 [0, 1], 0=isotropic, 1=perfectly aligned
        scale_map   : (H, W) float64, approximate local texture period (px)
    """
    gy, gx = np.gradient(gray.astype(np.float64))
    Jxx = gaussian_filter(gx * gx, sigma)
    Jxy = gaussian_filter(gx * gy, sigma)
    Jyy = gaussian_filter(gy * gy, sigma)

    diff = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    trace = Jxx + Jyy + 1e-10

    lambda1 = 0.5 * (trace + diff)   # larger eigenvalue
    lambda2 = 0.5 * (trace - diff)   # smaller eigenvalue

    angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)).astype(np.float32)
    coherence = np.clip(coherence, 0.0, 1.0)
    scale = sigma * np.sqrt(lambda1 / (lambda2 + 1e-10))

    return angle, coherence, scale

def compute_orientation_weight(
    angle_map: np.ndarray,
    coherence_map: np.ndarray,
    tool_angle_deg: float = 0.0,
) -> np.ndarray:
    """
    Per-pixel weight based on texture–toolpath alignment.

    Isotropic regions (coherence → 0) always get weight 1.0.
    Anisotropic regions: weight depends on |cos(angle − tool_angle)|.

    Returns (H, W) float32 [0, 1].
    """
    tool_rad = np.deg2rad(tool_angle_deg)
    alignment = np.abs(np.cos(angle_map - tool_rad))
    weight = (1.0 - coherence_map) + coherence_map * alignment
    return np.clip(weight, 0.0, 1.0).astype(np.float32)

def compute_local_height_range(
    heightfield: np.ndarray,
    patch_size: int = 64,
    stride: int = 16,
) -> np.ndarray:
    """
    Per-pixel local peak-valley height difference (in [0,1] normalised space).

    Sliding window: for each patch, range = max − min.
    Overlapping patches are averaged.

    Returns (H, W) float32, range in [0, 1].
    """
    H, W = heightfield.shape
    range_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = heightfield[y:y + patch_size, x:x + patch_size]
            r = float(patch.max() - patch.min())
            range_sum[y:y + patch_size, x:x + patch_size] += r
            count[y:y + patch_size, x:x + patch_size] += 1.0

    count = np.maximum(count, 1.0)
    return (range_sum / count).astype(np.float32)

def compute_machinability_weight(
    gray: np.ndarray,
    heightfield: np.ndarray,
    physical_size_mm: float,
    tool_diameter_mm: float,
    max_height_mm: float = 5.0,
    config: SaliencyConfig | None = None,
) -> tuple[np.ndarray, SaliencyReport]:
    """
    Machinability weight from multi-scale FFT + structure tensor + height range.

    Three signals are fused multiplicatively:
      w = w_period × w_orientation × w_height_range

    1. w_period: multi-scale FFT (32/64/128 px windows), energy-weighted average.
       Period ≥ 2×tool_diameter → 1.0, period ≤ tool_diameter → 0.0.
    2. w_orientation: structure tensor coherence × alignment with toolpath.
       Isotropic regions (knots) → 1.0 always.
    3. w_height_range: local peak-valley height difference.
       Range < tool_tolerance → 0.0 (negligible feature, let ADC smooth).

    Returns (weight_map [0,1], SaliencyReport).
    """
    if config is None:
        config = SaliencyConfig()

    H, W = gray.shape
    stride = config.fft_stride
    scales = sorted(config.fft_scales)
    min_win = scales[0]

    pixel_size_mm = physical_size_mm / max(H - 1, W - 1)
    min_machinable = tool_diameter_mm / pixel_size_mm  # in pixels

    # Accumulators
    w_period_sum = np.zeros((H, W), dtype=np.float64)
    period_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    non_periodic_count = 0
    unmachinable_count = 0  # windows with period < min_machinable
    total_windows = 0

    for y in range(0, H - min_win + 1, stride):
        for x in range(0, W - min_win + 1, stride):
            total_windows += 1
            weighted_period = 0.0
            total_energy = 0.0
            is_non_periodic = True

            for win in scales:
                # Centre the larger window on the same position
                cy_win = y + min_win // 2
                cx_win = x + min_win // 2
                y0 = max(cy_win - win // 2, 0)
                x0 = max(cx_win - win // 2, 0)
                y1 = min(y0 + win, H)
                x1 = min(x0 + win, W)

                patch = np.zeros((win, win), dtype=np.float64)
                ph, pw = y1 - y0, x1 - x0
                patch[:ph, :pw] = gray[y0:y1, x0:x1].astype(np.float64)

                hann = np.hanning(win)[:, None] * np.hanning(win)[None, :]
                windowed = patch * hann

                spectrum = np.fft.fftshift(np.fft.fft2(windowed))
                power = np.abs(spectrum) ** 2

                # Zero DC
                power[win // 2, win // 2] = 0.0

                peak_idx = np.unravel_index(np.argmax(power), power.shape)
                peak_energy = float(power[peak_idx])
                scale_total = float(power.sum())

                if scale_total < 1e-10:
                    continue

                energy_ratio = peak_energy / scale_total

                if energy_ratio >= config.fft_energy_threshold:
                    is_non_periodic = False
                    fy = (peak_idx[0] - win // 2) / win
                    fx = (peak_idx[1] - win // 2) / win
                    freq_mag = np.sqrt(fx ** 2 + fy ** 2)
                    p = 1.0 / freq_mag if freq_mag > 1e-10 else float(win)
                    weighted_period += p * peak_energy
                    total_energy += peak_energy

            # Accumulate into the min_win region
            region = (slice(y, y + min_win), slice(x, x + min_win))

            if is_non_periodic or total_energy < 1e-10:
                w_period_sum[region] += 1.0
                period_sum[region] += float(min_win)
                count[region] += 1.0
                non_periodic_count += 1
            else:
                avg_period = weighted_period / total_energy
                w = np.clip((avg_period - min_machinable) / min_machinable, 0.0, 1.0)
                w_period_sum[region] += w
                period_sum[region] += avg_period
                count[region] += 1.0
                if avg_period < min_machinable:
                    unmachinable_count += 1

    # Average overlapping windows
    count_safe = np.maximum(count, 1.0)
    w_period = (w_period_sum / count_safe).astype(np.float32)
    period_map = period_sum / count_safe

    # --- Structure tensor: orientation + coherence ---
    angle_map, coherence_map, _ = compute_local_structure_tensor(
        gray, sigma=config.structure_sigma
    )
    w_orientation = compute_orientation_weight(
        angle_map, coherence_map, config.tool_angle_deg
    )

    # --- Height range ---
    hr_patch = max(scales[-1], 32)
    hr_map = compute_local_height_range(heightfield, patch_size=hr_patch, stride=stride)
    height_range_mm = hr_map * max_height_mm
    w_height = np.clip(height_range_mm / config.tool_tolerance_mm, 0.0, 1.0).astype(np.float32)

    # --- Fuse: w = w_period × w_orientation × w_height ---
    weight_map = w_period * w_orientation * w_height

    # Smooth
    weight_map = gaussian_filter(weight_map, sigma=config.weight_blur_sigma)
    weight_map = np.clip(weight_map, 0.0, 1.0).astype(np.float32)

    # Report
    report = SaliencyReport()
    report.weight_min = float(weight_map.min())
    report.weight_max = float(weight_map.max())
    report.weight_mean = float(weight_map.mean())
    report.high_weight_fraction = float(
        (weight_map >= config.saliency_threshold_high).mean()
    )
    report.low_weight_fraction = float(
        (weight_map <= config.saliency_threshold_low).mean()
    )
    report.mean_period_px = float(period_map.mean())
    report.non_periodic_fraction = (
        non_periodic_count / total_windows if total_windows > 0 else 0.0
    )
    report.unmachinable_ratio = (
        (non_periodic_count + unmachinable_count) / total_windows
        if total_windows > 0 else 0.0
    )
    report.mean_coherence = float(coherence_map.mean())
    report.mean_height_range_mm = float(height_range_mm.mean())

    return weight_map, report, angle_map, period_map


def detect_knot_mask(
    gray: np.ndarray,
    local_sigma: float = 15.0,
    depth_threshold: float = 0.08,
    min_area: int = 20,
    max_area: int = 2000,
) -> np.ndarray:
    """
    Detect knot holes and cracks for protection during three-layer reconstruction.

    Criteria: pixels darker than local mean (sigma=15px) by depth_threshold,
    forming connected components with area in [min_area, max_area].
    No shape constraint — catches both circular knots and vertical cracks.

    Returns (H, W) bool mask — True where knots/cracks detected.
    """
    g = gray.astype(np.float32)
    local_mean = gaussian_filter(g, sigma=local_sigma)
    binary = g < (local_mean - depth_threshold)
    labeled, num_features = ndimage_label(binary)

    if num_features == 0:
        return np.zeros(g.shape, dtype=bool)

    mask = np.zeros(g.shape, dtype=bool)
    for i in range(1, num_features + 1):
        component = labeled == i
        area = int(component.sum())
        if min_area <= area <= max_area:
            mask |= component

    return mask


def three_layer_reconstruction(
    heightfield: np.ndarray,
    source_image: np.ndarray,
    unmachinable_ratio: float = 0.0,
    orientation_angle_deg: float = 0.0,
    tool_radius_px: float = 16.0,
) -> np.ndarray:
    """
    Three-layer heightfield reconstruction for improved terrace quality.

    Layers:
      1. Base plane (sigma=60px): large-scale tilt and smooth trends. Weight 0.5.
      2. Macro relief: mid-frequency residuals (knots, curves), rectangular ADC.
         Knot regions masked during ADC. Weight 0.35.
      3. Micro texture: synthetic fine detail.
         - unmachinable < 70%: directional Gaussian noise (sigma_par=20, sigma_perp=3),
           amplitude 0.04-0.08 modulated by source image local contrast.
         - unmachinable >= 70%: sinusoidal stripes (period=32px),
           amplitude 0.04-0.08 modulated by source image local contrast.
         Weight 0.15.

    Blend: final = Base * 0.5 + Macro * 0.35 + Micro * 0.15
    Knot regions: Micro weight = 0 (only Base + Macro preserved).

    Returns (H, W) float32 [0, 1].
    """
    H, W = heightfield.shape
    hf = heightfield.astype(np.float32)

    # --- Phase 3: Detect knot/crack mask ---
    gray = source_image.astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2) / 255.0
    knot_mask = detect_knot_mask(gray)

    # Smooth mask boundary for seamless blending
    knot_float = gaussian_filter(knot_mask.astype(np.float32), sigma=10.0)
    knot_float = np.clip(knot_float, 0.0, 1.0)

    # --- Layer 1: Base plane ---
    base = gaussian_filter(hf, sigma=60.0).astype(np.float32)

    # --- Layer 2: Macro relief ---
    macro_raw = hf - base  # mid-frequency residual
    macro_raw = np.clip(macro_raw + 0.5, 0.0, 1.0).astype(np.float32)  # shift to [0,1]

    # Rectangular ADC on macro layer, skipping knot regions
    r = max(int(math.ceil(tool_radius_px)), 1)
    rect = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    inv = (1.0 - macro_raw).astype(np.float32)
    eroded = grey_erosion(inv, footprint=rect)
    opened = grey_dilation(eroded, footprint=rect)
    macro_adc = np.clip(1.0 - opened, 0.0, 1.0).astype(np.float32)

    # Restore knot regions from original macro
    macro_adc[knot_mask] = macro_raw[knot_mask]

    # Normalise macro to [0, 1] (weight applied in blend formula)
    macro_min, macro_max = float(macro_adc.min()), float(macro_adc.max())
    if macro_max - macro_min > 1e-6:
        macro_norm = (macro_adc - macro_min) / (macro_max - macro_min)
    else:
        macro_norm = np.full_like(macro_adc, 0.3)

    # --- Layer 3: Micro texture (synthetic) ---
    # Source image local contrast for amplitude modulation
    block = 32
    mean_sq = uniform_filter(gray * gray, size=block, mode='reflect')
    sq_mean = uniform_filter(gray, size=block, mode='reflect') ** 2
    local_std = np.sqrt(np.maximum(mean_sq - sq_mean, 0.0))
    std_max = float(local_std.max())
    density = (local_std / std_max).astype(np.float32) if std_max > 1e-6 else np.full((H, W), 0.5, dtype=np.float32)

    y_coords, x_coords = np.mgrid[0:H, 0:W].astype(np.float32)
    angle_rad = np.radians(orientation_angle_deg)
    proj = x_coords * np.cos(angle_rad) + y_coords * np.sin(angle_rad)

    if unmachinable_ratio < 0.70:
        # Route 1: directional Gaussian noise
        sigma_par, sigma_perp = 20.0, 3.0
        noise = np.random.default_rng(42).standard_normal((H, W)).astype(np.float32)
        # Anisotropic blur: blur along perpendicular direction more
        noise_blur = gaussian_filter(noise, sigma=[sigma_perp, sigma_par])
        amp_lo, amp_hi = 0.04, 0.08
        amplitude = amp_lo + (amp_hi - amp_lo) * density
        micro = amplitude * noise_blur
    else:
        # Route 2: sinusoidal stripes at 32px period (vertical, perpendicular to horizontal toolpath)
        stripe_period_px = 32.0
        freq = 1.0 / stripe_period_px
        stripes = np.sin(2.0 * np.pi * freq * proj).astype(np.float32)
        amp_lo, amp_hi = 0.04, 0.08
        amplitude = amp_lo + (amp_hi - amp_lo) * density
        micro = amplitude * stripes

    micro = np.clip(micro, -0.1, 0.1).astype(np.float32)
    # Shift to centred range for blending
    micro = micro - micro.mean()

    # --- Blend: Base 0.5 + Macro 0.35 + Micro 0.15 ---
    # Knot regions: micro weight = 0 (preserve real features)
    micro_weight_map = 0.15 * (1.0 - knot_float)
    final = base * 0.5 + macro_norm * 0.35 + micro * micro_weight_map

    # Normalise to [0, 1]
    f_min, f_max = float(final.min()), float(final.max())
    if f_max - f_min > 1e-6:
        final = (final - f_min) / (f_max - f_min)
    else:
        final = np.full_like(final, 0.5)

    return np.clip(final, 0.0, 1.0).astype(np.float32)


def adaptive_adc_refinement(
    heightfield:  np.ndarray,       # (H, W) float32 [0,1]
    weight_map:   np.ndarray,       # (H, W) float32 [0,1]
    tool_radius_px: float,
    orientation_map: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Continuous ADC strength based on machinability weight.

    weight=1 → no ADC; weight=0 → full ADC at tool_radius_px.
    Intermediate weights get proportionally scaled ADC radius.

    Computes ADC at 4 discrete radius levels and linearly interpolates
    per-pixel for smooth, continuous processing strength.
    """
    # 4 radius levels: identity, 33%, 67%, 100%
    r0 = 0.0
    r1 = max(int(math.ceil(tool_radius_px * 0.33)), 1)
    r2 = max(int(math.ceil(tool_radius_px * 0.67)), 1)
    r3 = max(int(math.ceil(tool_radius_px)), 1)

    radii = [r0, r1, r2, r3]
    adc = [heightfield]  # r0 = identity
    for r in [r1, r2, r3]:
        adc.append(suppress_narrow_recesses(heightfield, r,
                                            orientation_map=orientation_map,
                                            mask=mask))

    # Per-pixel target radius from weight: weight=1 → r=0, weight=0 → r=max
    target_r = tool_radius_px * (1.0 - weight_map)

    # Interpolate between the two nearest ADC levels
    result = np.empty_like(heightfield, dtype=np.float32)
    for i in range(3):
        lo, hi = radii[i], radii[i + 1]
        band = (target_r >= lo) & (target_r < hi)
        if hi == lo:
            continue
        t = ((target_r - lo) / (hi - lo)).astype(np.float32)
        t = np.clip(t, 0.0, 1.0)
        result[band] = (1.0 - t[band]) * adc[i][band] + t[band] * adc[i + 1][band]

    # Pixels at or above max radius → full ADC
    result[target_r >= radii[-1]] = adc[-1][target_r >= radii[-1]]
    # Pixels at min radius (weight=1) → original
    result[target_r <= radii[0]] = heightfield[target_r <= radii[0]]

    return np.clip(result, 0.0, 1.0).astype(np.float32)

def heightfield_to_saliency_adaptive_terrace(
    heightfield:  np.ndarray,       # (H, W) float32 [0,1], already refined
    saliency_map: np.ndarray,       # (H, W) float32 [0,1]
    config: SaliencyConfig | None = None,
    terrace_config: TerraceConfig | None = None,
) -> tuple[trimesh.Trimesh, TerraceReport]:
    """
    Build terrace mesh with saliency-adaptive level allocation.

    Strategy:
      1. Divide heightfield into saliency zones (high/mid/low).
      2. For each zone, quantize with different step counts.
      3. Blend zone boundaries smoothly to avoid hard seams.
      4. Pass the composite quantized heightfield to heightfield_to_terrace_mesh.

    High-saliency zones get more terrace levels -> finer height resolution
    Low-saliency zones get fewer levels -> simpler geometry
    """
    if config is None:
        config = SaliencyConfig()
    if terrace_config is None:
        terrace_config = TerraceConfig()

    W = heightfield.shape[1]

    steps_high = config.terrace_steps_high
    steps_low  = config.terrace_steps_low

    # Soft zone assignment via saliency thresholds
    t_hi = config.saliency_threshold_high
    t_lo = config.saliency_threshold_low

    # Per-pixel step count (float for smooth blending)
    step_map = np.where(
        saliency_map >= t_hi,
        float(steps_high),
        np.where(
            saliency_map <= t_lo,
            float(steps_low),
            # Linear interpolation in the middle zone
            steps_low + (steps_high - steps_low) *
            (saliency_map - t_lo) / (t_hi - t_lo + 1e-8)
        )
    ).astype(np.float32)

    # Build composite heightfield by blending quantizations at different steps
    def quantize_at(hf: np.ndarray, n_steps: int) -> np.ndarray:
        if n_steps <= 1:
            return hf.copy()
        n = n_steps - 1
        q = np.round(hf * n) / n
        return q.astype(np.float32)

    hf_q_high = quantize_at(heightfield, steps_high)
    hf_q_low  = quantize_at(heightfield, steps_low)

    # Per-pixel blend weight: 1.0 at steps_high, 0.0 at steps_low
    blend_w = (step_map - steps_low) / (steps_high - steps_low + 1e-8)
    blend_w = np.clip(blend_w, 0.0, 1.0)

    hf_composite = blend_w * hf_q_high + (1.0 - blend_w) * hf_q_low

    # Smooth riser edges globally
    pixel_size_mm = terrace_config.physical_size_mm / (W - 1)
    if terrace_config.max_height_mm > 0:
        step_height_mm = terrace_config.max_height_mm / (steps_high - 1)
        riser_sigma = max(
            step_height_mm / (
                math.tan(math.radians(terrace_config.physical_size_mm))
                * pixel_size_mm * math.sqrt(2 * math.pi)
            ),
            1.0,
        )
    else:
        riser_sigma = 2.0

    hf_composite = gaussian_filter(hf_composite, sigma=riser_sigma)
    hf_composite = np.clip(hf_composite, 0.0, 1.0).astype(np.float32)

    # Use the standard terrace mesh builder with steps_high
    tc = TerraceConfig(
        physical_size_mm  = terrace_config.physical_size_mm,
        max_height_mm     = terrace_config.max_height_mm,
        base_thickness_mm = terrace_config.base_thickness_mm,
        terrace_steps     = steps_high,
        tool_diameter_mm  = terrace_config.tool_diameter_mm,
        mesh_resolution   = terrace_config.mesh_resolution,
        face_limit        = terrace_config.face_limit,
    )

    mesh, report = heightfield_to_terrace_mesh(hf_composite, tc)
    return mesh, report

def run_saliency_pipeline(
    raw_heightmap_path: str,
    source_image_path:  str | None = None,  # unused by FFT pipeline, kept for API compat
    config:             MachiningFilterConfig | None = None,
    terrace_config:     TerraceConfig  | None = None,
    saliency_config:    SaliencyConfig | None = None,
    save_saliency_map:  str | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray, MachiningFilterReport, TerraceReport, SaliencyReport]:
    """
    Complete Tactile Saliency Guidance pipeline (FFT-based).

    Steps:
      0. Load heightfield.
      1. Local FFT frequency analysis → machinability weight map.
      2. Frequency-weighted ADC refinement.
      3. Standard machining filter (normalize, slope compress).
      4. Saliency-adaptive terrace mesh generation.

    Returns:
      (mesh, final_heightfield, machining_report, terrace_report, saliency_report)
    """
    if config is None:
        config = MachiningFilterConfig()
    if terrace_config is None:
        terrace_config = TerraceConfig()
    if saliency_config is None:
        saliency_config = SaliencyConfig()

    # Step 0: Load heightfield
    hf = (np.load(raw_heightmap_path)
          if raw_heightmap_path.endswith(".npy")
          else np.array(Image.open(raw_heightmap_path).convert("L"),
                        dtype=np.float32) / 255.0)

    res = hf.shape[0]

    # Step 0a: Percentile normalization (p2/p98)
    p2, p98 = np.percentile(hf, 2), np.percentile(hf, 98)
    hf = np.clip((hf - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)

    # Step 0b: Detrend — subtract global background surface (sigma=150px)
    print("Removing global trend (sigma=150px)...")
    background = gaussian_filter(hf, sigma=150)
    hf = hf - background
    hf = hf - hf.min()
    hf = hf / (hf.max() + 1e-8)
    hf = hf.astype(np.float32)
    print(f"  After detrend: std={hf.std():.4f} (should be > 0.14)")

    # Step 1: Multi-scale FFT + structure tensor + height range → weight map
    print("Computing machinability weight map...")
    tool_diameter_mm = config.tool_radius_mm * 2.0
    weight_map, sal_report, orientation_map, period_map = compute_machinability_weight(
        hf, hf, config.physical_size_mm, tool_diameter_mm,
        config.max_height_mm, saliency_config
    )
    print(f"  Weight: mean={sal_report.weight_mean:.3f}, "
          f"high={sal_report.high_weight_fraction:.1%}, "
          f"low={sal_report.low_weight_fraction:.1%}, "
          f"coherence={sal_report.mean_coherence:.3f}, "
          f"unmachinable={sal_report.unmachinable_ratio:.1%}")

    # Step 2: Three-layer reconstruction — DISABLED for detrending verification
    # Verify detrending produces correct terrace contours first,
    # then re-enable three-layer reconstruction as enhancement.
    print("  (Three-layer reconstruction skipped for detrending verification)")

    # Knot mask still needed for machining filter's ADC protection
    pixel_size_mm = config.physical_size_mm / max(res - 1, 1)
    tool_radius_px = config.tool_radius_mm / pixel_size_mm

    if save_saliency_map:
        w_vis = (weight_map * 255).astype(np.uint8)
        Image.fromarray(w_vis).save(save_saliency_map)
        print(f"  Weight map saved: {save_saliency_map}")

    hf = normalize_heightfield(hf)

    # Step 3: Standard machining filter (prune, knot protection, ADC, slope)
    print("Applying machining filter...")
    hf_filtered, mach_report = filter_heightfield_for_machining(
        hf,
        MachiningFilterConfig(
            physical_size_mm       = config.physical_size_mm,
            max_height_mm          = config.max_height_mm,
            tool_radius_mm         = config.tool_radius_mm,
            max_slope_deg          = config.max_slope_deg,
            face_limit             = config.face_limit,
            gaussian_sigma_px      = config.gaussian_sigma_px,
            max_iterations         = config.max_iterations,
            target_resolution_mode = config.target_resolution_mode,
            terrace_steps          = 1,
            terrace_mode           = True,
        ),
        orientation_map=orientation_map,
        source_image=None,  # already handled by three_region preprocessing
    )
    print(f"  Machining filter: passed={mach_report.passed}")

    # Step 4: Saliency-adaptive terrace
    print("Building saliency-adaptive terrace mesh...")

    hf_res = hf_filtered.shape[0]
    weight_resized = cv2.resize(weight_map, (hf_res, hf_res),
                                interpolation=cv2.INTER_AREA)

    mesh, ter_report = heightfield_to_saliency_adaptive_terrace(
        hf_filtered, weight_resized, saliency_config, terrace_config
    )
    print(f"  Terrace: watertight={ter_report.watertight}, "
          f"faces={ter_report.face_count:,}")

    return mesh, hf_filtered, mach_report, ter_report, sal_report