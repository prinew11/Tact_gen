"""
Heightfield post-processing driven by the memory store.

Queries procedural memory for hard constraints and semantic memory for
correction strategy, then applies the minimum modification needed to make
the heightfield machinable while preserving tactile fidelity.

Correction pipeline (order matters):
  1. Clamp height range to [0, 1].
  2. Morphological opening to remove sub-tool-diameter features.
  3. Light smoothing for C1 continuity (reduces tool chatter).
  4. Final clamp.

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
    resolution: int = 512


@dataclass
class CorrectionReport:
    morph_pixels_changed: int = 0
    constraints_applied: list[str] = field(default_factory=list)
    guidance_used: list[str] = field(default_factory=list)


def _morphological_open(
    heightfield: np.ndarray,
    tool_radius_px: float,
) -> np.ndarray:
    r = max(1, int(round(tool_radius_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    hf_uint16 = (heightfield * 65535).astype(np.uint16)
    opened = cv2.morphologyEx(hf_uint16, cv2.MORPH_OPEN, kernel)
    return opened.astype(np.float32) / 65535.0


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

    # --- Step 3: Light Gaussian smooth for C1 continuity ---
    sigma_smooth = max(tool_radius_px * 0.3, 0.5)
    hf = ndimage.gaussian_filter(hf, sigma=sigma_smooth)

    # --- Step 4: Final clamp ---
    hf = np.clip(hf, 0.0, 1.0)

    return hf, report


def print_correction_report(report: CorrectionReport) -> None:
    print("\n--- Fabrication Correction Report ---")
    print(f"  Constraints applied  : {', '.join(report.constraints_applied)}")
    print(f"  Semantic guidance    : {', '.join(report.guidance_used)}")
    print(f"  Morph pixels changed : {report.morph_pixels_changed:,}")
