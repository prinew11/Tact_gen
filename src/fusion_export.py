"""
Fusion 360 export preparation: runs fabrication checks, copies STL to
export-ready location, and generates a brief setup note for manual CAM workflow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


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
    import shutil
    import trimesh

    from fabrication import FabricationConfig, check_mesh, print_report

    stl_path = Path(stl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and check
    mesh = trimesh.load(str(stl_path))
    config = FabricationConfig(tool_radius_mm=tool_radius_mm)
    report = check_mesh(mesh, config)
    print_report(report)

    if not report.passes:
        raise ValueError(
            "Mesh failed fabrication checks — fix issues before Fusion export.\n"
            + "\n".join(f"  - {i}" for i in report.issues)
        )

    # Copy STL
    dest_stl = out_dir / stl_path.name
    shutil.copy2(stl_path, dest_stl)

    # Write CAM setup note
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
        f"Max slope     : {report.max_slope_deg:.1f}° (limit 45°)\n"
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
