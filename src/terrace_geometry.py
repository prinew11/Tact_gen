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
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion


# ---------------------------------------------------------------------------
# Preprocessing dataclasses
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
) -> np.ndarray:
    """
    Morphological opening on the inverted heightfield removes grooves and
    concavities narrower than tool_radius_px * 2 pixels (= tool diameter).
    Uses grey (grayscale) morphology because heightfields are continuous-valued.
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


def filter_heightfield_for_machining(
    heightfield: np.ndarray,
    config: MachiningFilterConfig | None = None,
) -> tuple[np.ndarray, MachiningFilterReport]:
    """
    Apply all machining constraints to a heightfield in a deterministic sequence:
      1. Normalize to [0, 1].
      2. Downsample if needed to satisfy face_limit (auto mode).
      3. Prune sub-tool-scale high-frequency noise.
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

    # Step 4: Suppress sub-tool-diameter recesses
    tool_radius_px = config.tool_radius_mm / pixel_size_mm
    report.morph_radius_px = tool_radius_px
    hf = suppress_narrow_recesses(hf, tool_radius_px)
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


def save_heightfield(heightfield: np.ndarray, out_path: str | Path) -> None:
    """Save heightfield as .npy."""
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
# Terrace-aware preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_terrace(
    heightfield: np.ndarray,
    tool_diameter_mm: float = 6.0,
    physical_size_mm: float = 50.0,
    target_resolution: int = 256,
) -> np.ndarray:
    """
    Lightweight preprocessing for terrace mode:
      1. Normalize to [0, 1].
      2. Downsample to target_resolution (INTER_AREA).
      3. Mild high-frequency pruning (sigma = 0.5 * tool_radius_px).
      4. Morphological opening to suppress narrow recesses narrower than
         tool_diameter_mm (grey morphology).

    No slope optimisation, no terracing, no riser blurring.
    Quantisation is performed inside heightfield_to_terrace_mesh().
    """
    hf = normalize_heightfield(heightfield)

    if hf.shape[0] != target_resolution or hf.shape[1] != target_resolution:
        hf = cv2.resize(hf, (target_resolution, target_resolution),
                        interpolation=cv2.INTER_AREA)

    px_size = physical_size_mm / (target_resolution - 1)
    tool_radius_mm = tool_diameter_mm / 2.0
    tool_radius_px = tool_radius_mm / px_size

    hf = prune_high_frequency_content(hf, tool_radius_mm, px_size)
    hf = suppress_narrow_recesses(hf, tool_radius_px)

    return np.clip(hf, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_stl(mesh: trimesh.Trimesh, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"Terrace STL saved: {out_path}  ({len(mesh.faces):,} faces)")
