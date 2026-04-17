"""
Heightfield post-processing driven by the memory store.

Queries procedural memory for hard constraints and semantic memory for
correction strategy, then applies the minimum modification needed to make
the heightfield machinable while preserving tactile fidelity.

Correction pipeline (order matters):
  1. Clamp height range to [0, 1].
  2. Morphological opening to remove sub-tool-diameter features.
  3. Slope enforcement via bidirectional constraint propagation.
  4. Light smoothing for C1 continuity (reduces tool chatter).
  5. Final clamp.

All corrections operate on the 2-D heightfield *before* mesh conversion,
which is far cheaper than fixing the mesh after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from memory_store import MemoryStore, RetrievalResult


@dataclass
class CorrectionConfig:
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_radius_mm: float = 0.5
    max_slope_deg: float = 45.0
    propagation_sweeps: int = 6
    resolution: int = 512


@dataclass
class CorrectionReport:
    original_max_slope_deg: float = 0.0
    corrected_max_slope_deg: float = 0.0
    slope_violations_before: int = 0
    slope_violations_after: int = 0
    morph_pixels_changed: int = 0
    propagation_sweeps_used: int = 0
    constraints_applied: list[str] = field(default_factory=list)
    guidance_used: list[str] = field(default_factory=list)


def _compute_slope_map(
    heightfield: np.ndarray,
    pixel_spacing_mm: float,
    max_height_mm: float,
) -> np.ndarray:
    z = heightfield * max_height_mm
    gy, gx = np.gradient(z, pixel_spacing_mm)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad)


def _morphological_open(
    heightfield: np.ndarray,
    tool_radius_px: float,
) -> np.ndarray:
    r = max(1, int(round(tool_radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    hf_uint16 = (heightfield * 65535).astype(np.uint16)
    opened = cv2.morphologyEx(hf_uint16, cv2.MORPH_OPEN, kernel)
    return opened.astype(np.float32) / 65535.0


def _enforce_max_slope(
    heightfield: np.ndarray,
    max_dz: float,
    n_sweeps: int = 6,
) -> np.ndarray:
    """
    Enforce maximum per-pixel height difference via bidirectional sweeps.

    Each sweep propagates constraints in one direction: if a pixel differs
    from its already-visited neighbor by more than *max_dz*, it gets clamped.
    Forward + backward sweeps in both axes converge quickly to a solution
    that satisfies the slope constraint everywhere while staying as close
    to the original as possible (in the L-infinity sense).
    """
    hf = heightfield.copy()
    H, W = hf.shape

    for _ in range(n_sweeps):
        # Forward pass: top-left → bottom-right
        for r in range(H):
            for c in range(W):
                if r > 0:
                    hf[r, c] = np.clip(hf[r, c], hf[r - 1, c] - max_dz, hf[r - 1, c] + max_dz)
                if c > 0:
                    hf[r, c] = np.clip(hf[r, c], hf[r, c - 1] - max_dz, hf[r, c - 1] + max_dz)

        # Backward pass: bottom-right → top-left
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, -1, -1):
                if r < H - 1:
                    hf[r, c] = np.clip(hf[r, c], hf[r + 1, c] - max_dz, hf[r + 1, c] + max_dz)
                if c < W - 1:
                    hf[r, c] = np.clip(hf[r, c], hf[r, c + 1] - max_dz, hf[r, c + 1] + max_dz)

    return hf


def _enforce_max_slope_fast(
    heightfield: np.ndarray,
    max_dz: float,
    n_sweeps: int = 6,
) -> np.ndarray:
    """
    Vectorized slope enforcement using cumulative min/max operations.

    Equivalent to _enforce_max_slope but runs in numpy rather than
    per-pixel Python loops.  Each directional pass ensures that
    consecutive rows/columns differ by at most *max_dz*.
    """
    hf = heightfield.copy()
    H, W = hf.shape

    for _ in range(n_sweeps):
        # Forward along rows (left → right)
        for c in range(1, W):
            lo = hf[:, c - 1] - max_dz
            hi = hf[:, c - 1] + max_dz
            hf[:, c] = np.clip(hf[:, c], lo, hi)

        # Forward along columns (top → bottom)
        for r in range(1, H):
            lo = hf[r - 1, :] - max_dz
            hi = hf[r - 1, :] + max_dz
            hf[r, :] = np.clip(hf[r, :], lo, hi)

        # Backward along rows (right → left)
        for c in range(W - 2, -1, -1):
            lo = hf[:, c + 1] - max_dz
            hi = hf[:, c + 1] + max_dz
            hf[:, c] = np.clip(hf[:, c], lo, hi)

        # Backward along columns (bottom → top)
        for r in range(H - 2, -1, -1):
            lo = hf[r + 1, :] - max_dz
            hi = hf[r + 1, :] + max_dz
            hf[r, :] = np.clip(hf[r, :], lo, hi)

    return hf


def _retrieve_correction_strategy(
    store: MemoryStore,
    config: CorrectionConfig,
) -> tuple[dict[str, Any], list[str]]:
    constraints = store.get_constraints_dict()
    semantic_results: list[RetrievalResult] = store.retrieve(
        "slope smoothing tactile preservation machining correction",
        top_k=5,
        type_filter="semantic",
    )
    guidance_ids = [r.entry.id for r in semantic_results]
    return constraints, guidance_ids


def correct_heightfield(
    heightfield: np.ndarray,
    config: CorrectionConfig | None = None,
    store: MemoryStore | None = None,
    material_hint: str | None = None,
) -> tuple[np.ndarray, CorrectionReport]:
    """
    Post-process a raw heightfield to satisfy fabrication constraints.

    Args:
        heightfield: (H, W) float32 in [0, 1] from the diffusion model.
        config: correction parameters; uses defaults if None.
        store: MemoryStore instance; created with default KB if None.
        material_hint: optional material name (e.g. "wood", "bark") to
            retrieve material-specific guidance from semantic memory.

    Returns:
        corrected: (H, W) float32 in [0, 1], fabrication-ready.
        report: CorrectionReport with metrics.
    """
    if config is None:
        config = CorrectionConfig()
    if store is None:
        store = MemoryStore()

    report = CorrectionReport()

    constraints, guidance_ids = _retrieve_correction_strategy(store, config)
    report.guidance_used = guidance_ids

    if material_hint:
        mat_results = store.retrieve_for_material(material_hint)
        report.guidance_used += [r.entry.id for r in mat_results]

    # Memory provides defaults; user's explicit config takes priority
    slope_constraint = constraints.get("max_slope_deg")
    if slope_constraint:
        memory_slope = float(slope_constraint["value"])
        # Only use memory value if config is at the dataclass default
        if config.max_slope_deg == CorrectionConfig.max_slope_deg:
            config.max_slope_deg = memory_slope
        report.constraints_applied.append(f"max_slope_deg={config.max_slope_deg}")

    feature_constraint = constraints.get("min_feature_mm")
    if feature_constraint:
        report.constraints_applied.append(
            f"min_feature_mm={config.tool_radius_mm * 2}"
        )

    report.constraints_applied.append("height_range")

    hf = heightfield.copy().astype(np.float32)

    # --- Step 1: Clamp height range ---
    hf = np.clip(hf, 0.0, 1.0)

    pixel_spacing_mm = config.physical_size_mm / (config.resolution - 1)

    # --- Step 2: Morphological opening (min feature size) ---
    tool_radius_px = config.tool_radius_mm / pixel_spacing_mm
    if tool_radius_px >= 1.0:
        hf_before = hf.copy()
        hf = _morphological_open(hf, tool_radius_px)
        report.morph_pixels_changed = int(np.sum(np.abs(hf - hf_before) > 1e-4))

    # --- Step 3: Measure original slopes ---
    slope_map = _compute_slope_map(hf, pixel_spacing_mm, config.max_height_mm)
    report.original_max_slope_deg = float(slope_map.max())
    report.slope_violations_before = int(np.sum(slope_map > config.max_slope_deg))

    # --- Step 4: Slope enforcement via constraint propagation ---
    if report.slope_violations_before > 0:
        # Divide by sqrt(2) so that diagonal (combined X+Y) slopes also stay within limit
        max_dz = pixel_spacing_mm * np.tan(np.radians(config.max_slope_deg)) / (config.max_height_mm * np.sqrt(2))
        hf = _enforce_max_slope_fast(hf, max_dz, n_sweeps=config.propagation_sweeps)
        report.propagation_sweeps_used = config.propagation_sweeps

    # --- Step 5: Light Gaussian smooth for C1 continuity ---
    sigma_smooth = max(tool_radius_px * 0.3, 0.5)
    hf = ndimage.gaussian_filter(hf, sigma=sigma_smooth)

    # --- Step 6: Final clamp ---
    hf = np.clip(hf, 0.0, 1.0)

    slope_map = _compute_slope_map(hf, pixel_spacing_mm, config.max_height_mm)
    report.corrected_max_slope_deg = float(slope_map.max())
    report.slope_violations_after = int(np.sum(slope_map > config.max_slope_deg))

    return hf, report


def print_correction_report(report: CorrectionReport) -> None:
    print("\n--- Fabrication Correction Report ---")
    print(f"  Constraints applied  : {', '.join(report.constraints_applied)}")
    print(f"  Semantic guidance    : {', '.join(report.guidance_used)}")
    print(f"  Slope before         : {report.original_max_slope_deg:.1f}° "
          f"({report.slope_violations_before:,} violations)")
    print(f"  Slope after          : {report.corrected_max_slope_deg:.1f}° "
          f"({report.slope_violations_after:,} violations)")
    print(f"  Morph pixels changed : {report.morph_pixels_changed:,}")
    print(f"  Propagation sweeps   : {report.propagation_sweeps_used}")
