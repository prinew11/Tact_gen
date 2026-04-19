"""
Fabrication checks: slope, minimum feature size, watertightness, face count.
Validates mesh is compatible with Fusion 360 CAM + GRBL machining.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FabricationConfig:
    max_slope_deg: float = 45.0       # 3-axis machining limit
    tool_radius_mm: float = 3.0       # ball end mill radius (6 mm diameter)
    max_face_count: int = 500_000     # Fusion CAM stability limit
    min_feedrate_mm_min: float = 45.0 # GRBL hard lower limit
    physical_size_mm: float = 50.0    # XY extent for GRBL workspace check
    max_height_mm: float = 5.0        # Z range
    base_thickness_mm: float = 2.0    # flat bottom


@dataclass
class FabricationReport:
    watertight: bool
    face_count: int
    max_slope_deg: float
    min_feature_mm: float
    passes: bool
    grbl_compatible: bool = True
    issues: list[str] = field(default_factory=list)
    p95_slope_deg: float = 0.0
    estimated_target_resolution: int = 0
    estimated_min_feature_mm: float = 0.0


def check_mesh(
    mesh: "trimesh.Trimesh",
    config: FabricationConfig | None = None,
) -> FabricationReport:
    if config is None:
        config = FabricationConfig()

    issues: list[str] = []

    # Watertightness
    watertight = bool(mesh.is_watertight)
    if not watertight:
        issues.append("Mesh is not watertight")

    # Face count
    face_count = len(mesh.faces)
    if face_count > config.max_face_count:
        issues.append(
            f"Face count {face_count:,} exceeds Fusion limit {config.max_face_count:,}"
        )

    # Slope — only check TOP SURFACE faces, not side walls or bottom
    normals = mesh.face_normals
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_z = normals @ z_axis

    # Top surface faces have normals pointing upward (cos_z > 0.01)
    # Side walls have normals nearly perpendicular to Z (|cos_z| ≈ 0)
    # Bottom faces have normals pointing downward (cos_z < -0.01)
    top_mask = cos_z > 0.01
    p95_slope = 0.0
    if top_mask.any():
        top_cos = np.clip(np.abs(cos_z[top_mask]), 0.0, 1.0)
        top_slopes = np.degrees(np.arccos(top_cos))
        max_slope = float(top_slopes.max())
        p95_slope = float(np.percentile(top_slopes, 95))
    else:
        max_slope = 0.0

    if max_slope > config.max_slope_deg:
        issues.append(
            f"Max top-surface slope {max_slope:.1f}° exceeds limit {config.max_slope_deg}°"
        )

    # Minimum feature size: median edge length of top-surface edges only.
    # Using top-surface-only median avoids the tall side-wall edges and diagonal
    # cross-braces that make mesh.edges_unique_length.min() unreliable.
    z_threshold = config.base_thickness_mm + 0.1
    top_vert_mask = mesh.vertices[:, 2] > z_threshold
    top_vert_idx = set(np.where(top_vert_mask)[0].tolist())

    top_edges = [
        e for e in mesh.edges_unique
        if e[0] in top_vert_idx and e[1] in top_vert_idx
    ]
    if top_edges:
        top_edge_arr = np.array(top_edges)
        v0 = mesh.vertices[top_edge_arr[:, 0]]
        v1 = mesh.vertices[top_edge_arr[:, 1]]
        top_edge_lengths = np.linalg.norm(v1 - v0, axis=1)
        min_feature = float(np.median(top_edge_lengths))
    else:
        edge_lengths = mesh.edges_unique_length
        min_feature = float(np.median(edge_lengths)) if len(edge_lengths) > 0 else 0.0

    tool_diameter_mm = config.tool_radius_mm * 2.0
    if min_feature < tool_diameter_mm:
        issues.append(
            f"Estimated minimum feature {min_feature:.2f} mm is below tool diameter "
            f"{tool_diameter_mm:.1f} mm — grooves/recesses may be unreachable"
        )

    # GRBL workspace checks
    grbl_ok = True
    bounds = mesh.bounds  # (2, 3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    z_range = bounds[1][2] - bounds[0][2]

    if bounds[0][0] < -0.01 or bounds[0][1] < -0.01 or bounds[0][2] < -0.01:
        issues.append("Mesh has negative coordinates — GRBL expects origin at corner")
        grbl_ok = False

    total_z = config.max_height_mm + config.base_thickness_mm
    if z_range > total_z * 1.1:
        issues.append(
            f"Z range {z_range:.2f} mm exceeds expected {total_z:.1f} mm"
        )
        grbl_ok = False

    passes = len(issues) == 0

    target_res = int(math.floor(1.0 + math.sqrt(config.max_face_count / 4.0)))

    return FabricationReport(
        watertight=watertight,
        face_count=face_count,
        max_slope_deg=max_slope,
        min_feature_mm=min_feature,
        passes=passes,
        grbl_compatible=grbl_ok,
        issues=issues,
        p95_slope_deg=p95_slope,
        estimated_target_resolution=target_res,
        estimated_min_feature_mm=min_feature,
    )


def check_heightfield(
    heightfield: np.ndarray,
    config: FabricationConfig | None = None,
) -> FabricationReport:
    """
    Lightweight fabrication check operating directly on the heightfield.
    Returns a FabricationReport with a subset of fields populated.
    watertight is always True (not computable without mesh); face_count is 0.

    Explicitly flags features smaller than tool diameter in physical mm.
    """
    if config is None:
        config = FabricationConfig()

    issues: list[str] = []
    h, w = heightfield.shape

    pixel_size_mm = config.physical_size_mm / max(w - 1, 1)

    z = heightfield.astype(np.float32) * config.max_height_mm
    gy, gx = np.gradient(z, pixel_size_mm)
    slope_map = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

    max_slope = float(slope_map.max())
    p95_slope = float(np.percentile(slope_map, 95))

    if max_slope > config.max_slope_deg:
        issues.append(
            f"Max slope {max_slope:.1f}° exceeds limit {config.max_slope_deg}°"
        )

    # Smallest representable physical feature = one pixel
    min_feature_mm = pixel_size_mm
    tool_diameter_mm = config.tool_radius_mm * 2.0
    if min_feature_mm < tool_diameter_mm:
        issues.append(
            f"Pixel size {min_feature_mm:.3f} mm < tool diameter {tool_diameter_mm:.1f} mm "
            f"— sub-{tool_diameter_mm:.0f} mm features may not be machinable"
        )

    grbl_ok = True
    if config.physical_size_mm > 300.0:
        issues.append(
            f"Physical size {config.physical_size_mm} mm may exceed GRBL workspace"
        )
        grbl_ok = False

    passes = len(issues) == 0
    target_res = int(math.floor(1.0 + math.sqrt(config.max_face_count / 4.0)))

    return FabricationReport(
        watertight=True,
        face_count=0,
        max_slope_deg=max_slope,
        min_feature_mm=min_feature_mm,
        passes=passes,
        grbl_compatible=grbl_ok,
        issues=issues,
        p95_slope_deg=p95_slope,
        estimated_target_resolution=target_res,
        estimated_min_feature_mm=min_feature_mm,
    )


def print_report(report: FabricationReport) -> None:
    status = "PASS" if report.passes else "FAIL"
    print(f"\n=== Fabrication Check: {status} ===")
    print(f"  Watertight    : {report.watertight}")
    if report.face_count > 0:
        print(f"  Face count    : {report.face_count:,}")
    print(
        f"  Max slope     : {report.max_slope_deg:.1f}° "
        f"(p95: {report.p95_slope_deg:.1f}°, top surface only)"
    )
    print(f"  Min feature   : {report.min_feature_mm:.3f} mm")
    if report.estimated_target_resolution > 0:
        print(
            f"  Target res    : ≤{report.estimated_target_resolution} px for face budget"
        )
    print(f"  GRBL compat   : {report.grbl_compatible}")
    if report.issues:
        print("  Issues:")
        for issue in report.issues:
            print(f"    - {issue}")
