"""
Fabrication validation and Fusion 360 export preparation.

check_mesh(): validates watertightness, face count, slope (informational),
    minimum feature size, and GRBL workspace compatibility.
prepare_for_fusion(): runs the check, copies the STL to an export directory,
    and writes a CAM setup note for the manual Fusion 360 workflow.

Slope is measured and reported but does not affect pass/fail.
Terrace mode additionally checks minimum recess width and counts terrace levels.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FabricationConfig:
    tool_radius_mm: float = 3.0       # 6 mm ball end mill radius = 3 mm
    max_face_count: int = 500_000     # Fusion CAM stability limit
    min_feedrate_mm_min: float = 45.0 # GRBL hard lower limit
    physical_size_mm: float = 50.0    # XY extent for GRBL workspace check
    max_height_mm: float = 5.0        # Z range
    base_thickness_mm: float = 2.0    # flat bottom
    terrace_mode: bool = False


@dataclass
class FabricationReport:
    watertight: bool
    face_count: int
    max_slope_deg: float
    min_feature_mm: float
    passes: bool
    grbl_compatible: bool = True
    issues: list[str] = field(default_factory=list)
    # Terrace-mode extras (populated when FabricationConfig.terrace_mode is True)
    terrace_levels_detected: int = 0
    min_recess_width_mm: float = 0.0


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

    # Minimum feature size: median unique edge length approximates the smallest
    # representable detail.  In terrace mode this also estimates the minimum
    # recess floor width (the narrowest flat plateau in the mesh).
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

    # Terrace-mode extras
    terrace_levels = 0
    min_recess_width = 0.0
    if config.terrace_mode:
        tool_diameter_mm = config.tool_radius_mm * 2.0
        min_recess_width = min_feature
        if min_recess_width <= tool_diameter_mm:
            issues.append(
                f"Minimum recess width ~{min_recess_width:.2f} mm is <= tool diameter "
                f"{tool_diameter_mm:.1f} mm.  Some recesses may be unmachineable."
            )
        # Estimate terrace level count from distinct Z values among upward-facing faces.
        if top_mask.any():
            top_z = mesh.vertices[mesh.faces[top_mask], 2]
            unique_z = np.unique(np.round(top_z, decimals=4))
            terrace_levels = int(len(unique_z))

    passes = len(issues) == 0

    return FabricationReport(
        watertight=watertight,
        face_count=face_count,
        max_slope_deg=max_slope,
        min_feature_mm=min_feature,
        passes=passes,
        grbl_compatible=grbl_ok,
        issues=issues,
        terrace_levels_detected=terrace_levels,
        min_recess_width_mm=min_recess_width,
    )


def print_report(report: FabricationReport) -> None:
    status = "PASS" if report.passes else "FAIL"
    print(f"\n=== Fabrication Check: {status} ===")
    print(f"  Watertight    : {report.watertight}")
    print(f"  Face count    : {report.face_count:,}")
    print(f"  Max slope     : {report.max_slope_deg:.1f}° (top surface only)")
    print(f"  Grid spacing  : {report.min_feature_mm:.3f} mm")
    print(f"  GRBL compat   : {report.grbl_compatible}")
    if report.terrace_levels_detected:
        print(f"  Terrace levels: {report.terrace_levels_detected}")
        print(f"  Min recess w  : {report.min_recess_width_mm:.3f} mm")
    if report.issues:
        print("  Issues:")
        for issue in report.issues:
            print(f"    - {issue}")


# ---------------------------------------------------------------------------
# Fusion 360 export preparation (formerly fusion_export.py)
# ---------------------------------------------------------------------------

def prepare_for_fusion(
    stl_path: str | Path,
    out_dir: str | Path,
    tool_radius_mm: float = 0.5,
    stepover_mm: float = 0.3,
) -> Path:
    """
    Validate STL and write a Fusion 360 CAM setup note alongside it.

    Args:
        stl_path: path to the fabrication STL.
        out_dir: directory to place the export-ready copy + setup note.
        tool_radius_mm: ball end mill radius in mm.
        stepover_mm: recommended stepover for 3D parallel toolpath.

    Returns:
        Path to the written setup note (.txt).
    """
    import trimesh

    stl_path = Path(stl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(stl_path))
    config = FabricationConfig(tool_radius_mm=tool_radius_mm)
    report = check_mesh(mesh, config)
    print_report(report)

    if not report.passes:
        raise ValueError(
            "Mesh failed fabrication checks — fix issues before Fusion export.\n"
            + "\n".join(f"  - {i}" for i in report.issues)
        )

    dest_stl = out_dir / stl_path.name
    shutil.copy2(stl_path, dest_stl)

    note_path = out_dir / "fusion_cam_setup.txt"
    note_path.write_text(
        f"Fusion 360 CAM Setup Note\n"
        f"=========================\n"
        f"STL file      : {dest_stl.name}\n"
        f"Workspace     : Manufacture\n"
        f"Strategy      : 3D Parallel\n"
        f"Tool          : Ball end mill  r={tool_radius_mm} mm\n"
        f"Stepover      : {stepover_mm} mm\n"
        f"Post processor: grbl.cps  (see fusion/grbl.cps)\n"
        f"Max slope     : {report.max_slope_deg:.1f}° (informational)\n"
        f"Face count    : {report.face_count:,}\n"
        f"\nSteps:\n"
        f"  1. Import {dest_stl.name} into Fusion 360\n"
        f"  2. Switch to Manufacture workspace\n"
        f"  3. Create Setup → origin at bottom-left corner, Z-up\n"
        f"  4. Add 3D Parallel operation with above parameters\n"
        f"  5. Post-process with grbl.cps → save .gcode to outputs/gcode/\n"
    )

    print(f"Fusion export ready: {out_dir}")
    return note_path
