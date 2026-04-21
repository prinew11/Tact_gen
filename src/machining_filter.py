"""
ADC-style machining filter.

Replaces the older ``fabrication_corrector + machining_filter`` two-stage pipeline.

Pipeline:
  1. Clip + normalize to [0, 1].
  2. Optional downsample to satisfy Fusion face budget.
  3. Morphological opening on the inverted surface to suppress grooves
     narrower than the tool diameter.
  4. Hard ADC-style quantization into N discrete terrace levels.
     No riser softening — vertical step walls are machinable on a 3-axis
     ball-end mill (the side of the ball cuts the wall).

All corrections are deterministic and vectorised.  Slope in the report is
plateau-only (riser pixels excluded) because step walls are expected at
~90° and are not a violation for terraced output.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MachiningFilterConfig:
    physical_size_mm: float = 100.0     # 10 cm workpiece
    max_height_mm: float = 10.0         # 1 cm relief
    tool_radius_mm: float = 3.0         # 6 mm ball end mill
    max_slope_deg: float = 45.0         # plateau-only slope target
    face_limit: int = 500_000
    # terrace_steps = 0 → auto from physical_size / (tool_diameter * factor)
    # terrace_steps = 1 → disable quantisation entirely (debug only)
    terrace_steps: int = 0
    # Lower factor → more levels → finer gradation but smaller plateaus.
    # 1.0 gives ~17 levels at 100 mm / 6 mm tool (~0.6 mm per step over 10 mm).
    target_min_feature_factor: float = 1.0


@dataclass
class MachiningFilterReport:
    input_shape: tuple[int, int] = (0, 0)
    output_shape: tuple[int, int] = (0, 0)
    pixel_size_mm: float = 0.0
    estimated_face_count: int = 0
    terrace_steps_applied: int = 0
    step_height_mm: float = 0.0
    plateau_fraction: float = 0.0
    max_plateau_slope_deg: float = 0.0  # excludes riser pixels
    max_raw_slope_deg: float = 0.0      # includes risers (will be ~90°)
    min_feature_target_mm: float = 0.0
    morph_radius_px: float = 0.0
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(hf: np.ndarray) -> np.ndarray:
    hf = np.clip(hf, 0.0, 1.0).astype(np.float32)
    lo, hi = float(hf.min()), float(hf.max())
    if hi - lo < 1e-7:
        return hf
    return ((hf - lo) / (hi - lo)).astype(np.float32)


def _pixel_size_mm(physical_size_mm: float, resolution: int) -> float:
    return physical_size_mm / max(resolution - 1, 1)


def _estimate_face_count(resolution: int) -> int:
    """Top + bottom + 4 side walls for a heightfield mesh."""
    n = resolution - 1
    return 4 * n * (n + 2)


def _target_resolution_for_face_budget(face_limit: int, current_res: int) -> int:
    if _estimate_face_count(current_res) <= face_limit:
        return current_res
    r_max = int(math.floor(math.sqrt(face_limit / 4.0 + 1.0)))
    return max(r_max, 2)


def _slope_map_deg(hf: np.ndarray, pixel_size_mm: float, max_height_mm: float) -> np.ndarray:
    z = hf.astype(np.float32) * max_height_mm
    gy, gx = np.gradient(z, pixel_size_mm)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))


def _suppress_narrow_recesses(hf: np.ndarray, tool_radius_px: float) -> np.ndarray:
    """Grey morphological opening on the inverted surface — fills sub-tool grooves."""
    r = max(int(math.ceil(tool_radius_px)), 1)
    yi, xi = np.ogrid[-r: r + 1, -r: r + 1]
    disk = (xi ** 2 + yi ** 2 <= r ** 2)
    inv = (1.0 - hf).astype(np.float32)
    inv = grey_erosion(inv, footprint=disk)
    inv = grey_dilation(inv, footprint=disk)
    return np.clip(1.0 - inv, 0.0, 1.0).astype(np.float32)


def _quantize_adc(hf: np.ndarray, n_levels: int) -> np.ndarray:
    """Uniform ADC-style quantisation to n_levels discrete heights in [0, 1]."""
    if n_levels <= 1:
        return hf.copy()
    n = n_levels - 1
    return (np.round(hf * n) / n).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_heightfield_for_machining(
    heightfield: np.ndarray,
    config: MachiningFilterConfig | None = None,
) -> tuple[np.ndarray, MachiningFilterReport]:
    """
    Apply the full ADC machining filter.  See module docstring for steps.
    """
    if config is None:
        config = MachiningFilterConfig()

    if heightfield.ndim != 2:
        raise ValueError(f"Expected 2-D heightfield, got shape {heightfield.shape}")

    report = MachiningFilterReport()
    report.input_shape = (heightfield.shape[0], heightfield.shape[1])

    # --- 1. Normalize ---
    hf = _normalize(heightfield)

    # --- 2. Resolution / face budget ---
    target_res = _target_resolution_for_face_budget(config.face_limit, hf.shape[0])
    if target_res < hf.shape[0]:
        hf = cv2.resize(hf, (target_res, target_res), interpolation=cv2.INTER_AREA)
        report.recommendations.append(
            f"Downsampled {heightfield.shape[0]}→{target_res} px for face budget"
        )
    report.output_shape = hf.shape
    res = hf.shape[0]
    pixel_size_mm = _pixel_size_mm(config.physical_size_mm, res)
    report.pixel_size_mm = pixel_size_mm
    report.estimated_face_count = _estimate_face_count(res)

    # --- 3. Morphological opening (kill sub-tool grooves) ---
    # This is the only "smoothing" step — it enforces the hard physical
    # constraint that grooves narrower than the tool diameter cannot exist.
    # Extra Gaussian pre-blur is intentionally omitted: it would destroy
    # genuine features above the tool scale for no machinability gain.
    tool_radius_px = config.tool_radius_mm / pixel_size_mm
    report.morph_radius_px = tool_radius_px
    hf = _suppress_narrow_recesses(hf, tool_radius_px)

    # --- 4. ADC quantisation (hard steps, no softening) ---
    if config.terrace_steps == 0:
        tool_diameter_mm = config.tool_radius_mm * 2.0
        n_levels = max(
            2,
            round(config.physical_size_mm /
                  (tool_diameter_mm * config.target_min_feature_factor)),
        )
    else:
        n_levels = max(1, config.terrace_steps)
    hf = _quantize_adc(hf, n_levels)
    report.terrace_steps_applied = n_levels
    report.step_height_mm = (
        config.max_height_mm / (n_levels - 1) if n_levels > 1 else 0.0
    )

    # --- 6. Metrics (plateau-only slope excludes riser pixels) ---
    slope_map = _slope_map_deg(hf, pixel_size_mm, config.max_height_mm)
    report.max_raw_slope_deg = float(slope_map.max())

    # riser pixels: any pixel whose 3x3 neighbourhood contains a different level.
    if n_levels > 1:
        # int-level map for cheap comparisons
        levels = np.round(hf * (n_levels - 1)).astype(np.int16)
        dil = grey_dilation(levels, size=(3, 3))
        ero = grey_erosion(levels, size=(3, 3))
        riser_mask = (dil != ero)
        plateau_mask = ~riser_mask
    else:
        plateau_mask = np.ones_like(hf, dtype=bool)
    report.plateau_fraction = float(plateau_mask.mean())
    report.max_plateau_slope_deg = (
        float(slope_map[plateau_mask].max()) if plateau_mask.any() else 0.0
    )
    report.min_feature_target_mm = config.tool_radius_mm * 2.0

    # --- 7. Pass/fail ---
    if report.max_plateau_slope_deg > config.max_slope_deg:
        report.issues.append(
            f"Plateau slope {report.max_plateau_slope_deg:.1f}° exceeds limit "
            f"{config.max_slope_deg}° — quantisation did not fully flatten."
        )
    if pixel_size_mm > report.min_feature_target_mm:
        report.issues.append(
            f"Pixel size {pixel_size_mm:.2f} mm > tool diameter "
            f"{report.min_feature_target_mm:.1f} mm — resolution too coarse."
        )
    if report.estimated_face_count > config.face_limit:
        report.issues.append(
            f"Estimated face count {report.estimated_face_count:,} exceeds "
            f"limit {config.face_limit:,}."
        )
    if n_levels > 1:
        report.recommendations.append(
            f"{n_levels} terrace levels, step height {report.step_height_mm:.2f} mm. "
            "Vertical risers are machinable by the ball mill's side; plateau "
            f"fraction = {report.plateau_fraction*100:.1f}%."
        )
    report.passed = len(report.issues) == 0
    return hf, report


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_heightfield(hf: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), hf)


def save_report_json(report: MachiningFilterReport, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input_shape": list(report.input_shape),
        "output_shape": list(report.output_shape),
        "pixel_size_mm": report.pixel_size_mm,
        "estimated_face_count": report.estimated_face_count,
        "terrace_steps_applied": report.terrace_steps_applied,
        "step_height_mm": report.step_height_mm,
        "plateau_fraction": report.plateau_fraction,
        "max_plateau_slope_deg": report.max_plateau_slope_deg,
        "max_raw_slope_deg": report.max_raw_slope_deg,
        "min_feature_target_mm": report.min_feature_target_mm,
        "morph_radius_px": report.morph_radius_px,
        "passed": report.passed,
        "issues": report.issues,
        "recommendations": report.recommendations,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
