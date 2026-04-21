"""
Physics-aware, deterministic pre-geometry filter.

Sits between fabrication_corrector and geometry.  Does NOT duplicate slope
propagation sweeps (those live in fabrication_corrector.py).  Instead:

  1. Normalize heightfield to strict [0, 1].
  2. Optionally downsample to satisfy a triangle face budget.
  3. Apply Gaussian smoothing at tool scale (sigma = tool_radius_mm / pixel_size_mm).
  4. Apply morphological opening to suppress grooves/recesses narrower than tool
     diameter (hard 6 mm constraint with default config).
  5. Iteratively compress height scale until max slope satisfies max_slope_deg.
  6. Re-normalize and emit MachiningFilterReport.

All operations are deterministic (no randomness, no memory queries).

Hard manufacturability constraint:
  Any groove, channel, or recessed feature narrower than tool_radius_mm * 2
  (default: 6 mm) cannot be machined and is suppressed by morphological opening.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MachiningFilterConfig:
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_radius_mm: float = 3.0        # 6 mm diameter ball-end mill → radius = 3 mm
    max_slope_deg: float = 45.0
    face_limit: int = 500_000
    target_min_feature_factor: float = 1.5
    gaussian_sigma_px: float = 0.0     # 0 = auto from tool scale
    max_iterations: int = 10
    target_resolution_mode: str = "auto"   # "auto" | "fixed"
    # 0 = auto-compute from physical_size / tool_diameter (enables terracing);
    # 1 = no terracing; ≥2 = explicit step count.
    terrace_steps: int = 0


@dataclass
class MachiningFilterReport:
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    pixel_size_mm: float = 0.0
    estimated_face_count: int = 0
    max_slope_deg_before: float = 0.0
    max_slope_deg_after: float = 0.0
    min_feature_target_mm: float = 0.0      # = tool_radius_mm * 2 (hard limit)
    min_feature_estimate_mm: float = 0.0    # estimated from pixel size after filtering
    height_scale_applied: float = 1.0
    smoothing_sigma_px: float = 0.0
    morph_radius_px: float = 0.0
    terrace_steps_applied: int = 0
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_heightfield(hf: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and linearly rescale to use full dynamic range."""
    hf = np.clip(hf, 0.0, 1.0).astype(np.float32)
    lo, hi = float(hf.min()), float(hf.max())
    if hi - lo < 1e-7:
        return hf
    return ((hf - lo) / (hi - lo)).astype(np.float32)


def compute_pixel_size_mm(physical_size_mm: float, resolution: int) -> float:
    """Physical distance per pixel (edge-to-edge: N-1 intervals)."""
    return physical_size_mm / max(resolution - 1, 1)


def estimate_face_count(resolution: int) -> int:
    """
    Estimate triangle count for a watertight heightfield mesh including side walls.
    Formula: 4*(N-1)*(N+1) = top quads + bottom quads + 4 side wall strips.
      top/bottom: 2*(N-1)^2 each = 4*(N-1)^2 total
      4 side walls: 4*(N-1)*2 total
      combined: 4*(N-1)*((N-1)+2) = 4*(N-1)*(N+1)
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
    # Solve 4*(R-1)*(R+1) <= face_limit => R^2 <= face_limit/4 + 1
    r_max = int(math.floor(math.sqrt(face_limit / 4.0 + 1.0)))
    return max(r_max, 2)


def estimate_slope_map_deg(
    heightfield: np.ndarray,
    pixel_size_mm: float,
    max_height_mm: float,
) -> np.ndarray:
    """
    Compute per-pixel slope in degrees.
    Same formula as fabrication_corrector._compute_slope_map (not imported
    because that symbol is private; the formula itself is two lines).
    """
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
    quantize into discrete height levels to produce a circular-step "terraced"
    topology (like a topographic contour map).

    Args:
        terrace_steps: number of discrete height levels.  1 = no terracing.
        riser_sigma_px: Gaussian sigma for step-edge softening.
            0 = auto (sigma * 0.5).  Pass a slope-calibrated value from
            filter_heightfield_for_machining for physically correct risers.
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
    the step risers.  Separating this from smooth_by_tool_scale lets the
    morphological opening run first (on the smooth surface), so it never
    inadvertently widens or distorts the step edges.
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
) -> np.ndarray:
    """
    Morphological opening on the inverted heightfield removes grooves, channels,
    and concavities narrower than tool_radius_px * 2 pixels (= tool diameter).

    Physical reasoning: a ball-end mill of radius R cannot enter any depression
    narrower than 2R.  Morphological opening on the inverted surface fills these
    features, making the heightfield machinable.

    Uses grey (grayscale) erosion/dilation, NOT binary morphology, because
    heightfields are continuous-valued.
    """
    r = max(int(math.ceil(tool_radius_px)), 1)
    yi, xi = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (xi ** 2 + yi ** 2 <= r ** 2)

    inv = (1.0 - heightfield).astype(np.float32)
    inv_eroded = grey_erosion(inv, footprint=disk)
    inv_opened = grey_dilation(inv_eroded, footprint=disk)

    return np.clip(1.0 - inv_opened, 0.0, 1.0).astype(np.float32)


def prune_high_frequency_content(
    heightfield: np.ndarray,
    tool_radius_mm: float,
    pixel_size_mm: float,
) -> np.ndarray:
    """
    Remove spatial noise finer than half the tool radius before terracing.
    Avoids ragged step edges caused by sub-tool-scale texture in the diffusion
    output.  Uses a mild Gaussian (sigma = 0.5 * tool_radius / pixel_size).
    """
    sigma = max(tool_radius_mm * 0.5 / pixel_size_mm, 0.5)
    return gaussian_filter(heightfield.astype(np.float32), sigma=sigma)


def compress_height_for_slope(
    heightfield: np.ndarray,
    max_slope_deg: float,
    pixel_size_mm: float,
    max_height_mm: float,
    max_iters: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Binary-search a height scale factor in (0, 1] so the physical slope stays
    within max_slope_deg.

    Returns (heightfield_unchanged, scale_applied).
    The heightfield values are NOT modified — the caller should multiply
    max_height_mm by scale_applied when passing to geometry.heightfield_to_mesh.
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
            lo = mid   # can try a larger scale
        else:
            hi = mid   # must go smaller

    return heightfield.copy(), best_scale


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_heightfield_for_machining(
    heightfield: np.ndarray,
    config: MachiningFilterConfig | None = None,
) -> tuple[np.ndarray, MachiningFilterReport]:
    """
    Apply all machining constraints to a heightfield in a deterministic sequence.

    Steps:
      1. Normalize to [0, 1].
      2. Downsample if needed to satisfy face_limit (auto mode).
      3. Gaussian smoothing at tool scale.
      4. Morphological opening to suppress sub-tool-diameter recesses.
      5. Slope measurement (before compression).
      6. Iterative height compression to satisfy max_slope_deg.
      7. Re-normalize to preserve relative relief.
      8. Final slope measurement and report.

    Args:
        heightfield: (H, W) float32 in [0, 1].
        config: MachiningFilterConfig; defaults used if None.

    Returns:
        (filtered_heightfield, MachiningFilterReport)
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

    # --- Step 2: Resolution targeting ---
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
    pixel_size_mm = compute_pixel_size_mm(config.physical_size_mm, res)
    report.pixel_size_mm = pixel_size_mm
    report.estimated_face_count = estimate_face_count(res)

    # --- Step 3: Measure slope BEFORE any spatial filtering ---
    # (after normalize + optional downsample only, so the report shows
    #  the improvement delivered by the filter as a whole)
    slope_before_map = estimate_slope_map_deg(hf, pixel_size_mm, config.max_height_mm)
    report.max_slope_deg_before = float(slope_before_map.max())

    # --- Step 3.5: Resolve terrace step count ---
    # 0 = auto: each plateau ~ tool_diameter * factor wide on average.
    if config.terrace_steps == 0:
        tool_diameter_mm = config.tool_radius_mm * 2.0
        actual_terrace_steps = max(
            2,
            round(config.physical_size_mm / (tool_diameter_mm * config.target_min_feature_factor)),
        )
    else:
        actual_terrace_steps = config.terrace_steps
    report.terrace_steps_applied = actual_terrace_steps

    # --- Step 3.6: Prune sub-tool-scale noise ---
    hf = prune_high_frequency_content(hf, config.tool_radius_mm, pixel_size_mm)
    hf = np.clip(hf, 0.0, 1.0)

    # --- Step 4: Suppress sub-tool-diameter recesses BEFORE full Gaussian ---
    # Must run on the relatively unsmoothed surface: the full-scale Gaussian
    # (σ = tool_radius / pixel_size) broadens narrow trenches to FWHM ≈ 2.35·σ,
    # which can exceed the disk diameter and defeat the opening.  Pruning (mild
    # σ = 0.5·tool_radius) leaves trench widths well below the disk diameter.
    tool_radius_px = config.tool_radius_mm / pixel_size_mm
    report.morph_radius_px = tool_radius_px
    hf = suppress_narrow_recesses(hf, tool_radius_px)
    hf = np.clip(hf, 0.0, 1.0)
    tool_diameter_mm = config.tool_radius_mm * 2.0
    report.recommendations.append(
        f"Sub-{tool_diameter_mm:.0f} mm recesses suppressed via morphological "
        f"opening (tool diameter {tool_diameter_mm:.1f} mm)"
    )

    # --- Step 5: Gaussian smoothing (no terracing yet) ---
    # Terracing is deferred until after smoothing so the opening result blends
    # smoothly into the surface before discrete levels are applied.
    sigma_px = (
        config.gaussian_sigma_px
        if config.gaussian_sigma_px > 0.0
        else max(config.tool_radius_mm / pixel_size_mm, 1.0)
    )
    report.smoothing_sigma_px = sigma_px
    hf = smooth_by_tool_scale(
        hf, config.tool_radius_mm, pixel_size_mm,
        sigma_override_px=sigma_px,
        terrace_steps=1,          # Gaussian only — terracing applied in step 5b
    )
    hf = np.clip(hf, 0.0, 1.0)

    # --- Step 5b: Terracing with slope-calibrated riser sigma ---
    # Riser sigma is derived so the step edges are as sharp as possible while
    # still satisfying the max_slope_deg constraint.
    # For a Gaussian-convolved unit step: max_slope = step_height /
    #   (riser_sigma * pixel_size * sqrt(2π)).  Solving for riser_sigma:
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

    # --- Step 6: Height compression ---
    # Scale is embedded into heightfield values so geometry.py can use
    # max_height_mm unchanged and still get the correct physical height.
    # Use 97% of max_slope_deg as the target: np.gradient slightly underestimates
    # slopes compared to mesh-normal computation, so a small margin prevents the
    # fabrication check from reporting a borderline failure.
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

    # --- Step 7: Clip to [0, 1] (no re-normalize: avoids amplifying tiny
    #   range differences introduced by morphological opening) ---
    hf = np.clip(hf, 0.0, 1.0)

    # --- Step 8: Final metrics ---
    # hf already has scale baked in (hf = original * scale), so physical height
    # is hf * max_height_mm — do NOT multiply by scale again.
    slope_after = estimate_slope_map_deg(hf, pixel_size_mm, config.max_height_mm)
    report.max_slope_deg_after = float(slope_after.max())

    report.min_feature_target_mm = tool_diameter_mm
    report.min_feature_estimate_mm = pixel_size_mm * tool_radius_px * 2.0

    # --- Pass/fail checks ---
    if report.max_slope_deg_after > config.max_slope_deg:
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_heightfield(heightfield: np.ndarray, out_path: str | Path) -> None:
    """Save heightfield as .npy (mirrors diffusion_pipeline.save_heightfield)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), heightfield)
    print(f"Saved machinable heightfield: {out_path}")


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
