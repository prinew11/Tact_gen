"""
Fabrication checks: slope, minimum feature size, watertightness, face count.
Validates mesh is compatible with Fusion 360 CAM + GRBL machining.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FabricationConfig:
    max_slope_deg: float = 45.0       # 3-axis machining limit
    tool_radius_mm: float = 0.5       # ball end mill radius
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
    if top_mask.any():
        top_cos = np.clip(np.abs(cos_z[top_mask]), 0.0, 1.0)
        top_slopes = np.degrees(np.arccos(top_cos))
        max_slope = float(top_slopes.max())
    else:
        max_slope = 0.0

    if max_slope > config.max_slope_deg:
        issues.append(
            f"Max top-surface slope {max_slope:.1f}° exceeds limit {config.max_slope_deg}°"
        )

    # Minimum feature size: grid spacing (XY step between vertices)
    # This is the physical distance per mesh cell — determines the smallest
    # detail the mesh can represent.  The morphological opening in the
    # corrector already ensures heightfield features are ≥ tool diameter;
    # here we just report the grid resolution so the user can judge.
    edge_lengths = mesh.edges_unique_length
    grid_spacing = float(np.median(edge_lengths)) if len(edge_lengths) > 0 else 0.0
    min_feature = grid_spacing

    # GRBL workspace checks
    grbl_ok = True
    bounds = mesh.bounds  # (2, 3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    x_range = bounds[1][0] - bounds[0][0]
    y_range = bounds[1][1] - bounds[0][1]
    z_range = bounds[1][2] - bounds[0][2]

    if bounds[0][0] < -0.01 or bounds[0][1] < -0.01 or bounds[0][2] < -0.01:
        issues.append(f"Mesh has negative coordinates — GRBL expects origin at corner")
        grbl_ok = False

    total_z = config.max_height_mm + config.base_thickness_mm
    if z_range > total_z * 1.1:
        issues.append(
            f"Z range {z_range:.2f} mm exceeds expected {total_z:.1f} mm"
        )
        grbl_ok = False

    passes = len(issues) == 0

    return FabricationReport(
        watertight=watertight,
        face_count=face_count,
        max_slope_deg=max_slope,
        min_feature_mm=min_feature,
        passes=passes,
        grbl_compatible=grbl_ok,
        issues=issues,
    )


def print_report(report: FabricationReport) -> None:
    status = "PASS" if report.passes else "FAIL"
    print(f"\n=== Fabrication Check: {status} ===")
    print(f"  Watertight    : {report.watertight}")
    print(f"  Face count    : {report.face_count:,}")
    print(f"  Max slope     : {report.max_slope_deg:.1f}° (top surface only)")
    print(f"  Grid spacing  : {report.min_feature_mm:.3f} mm")
    print(f"  GRBL compat   : {report.grbl_compatible}")
    if report.issues:
        print("  Issues:")
        for issue in report.issues:
            print(f"    - {issue}")
