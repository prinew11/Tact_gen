"""
Geometry conversion: heightfield (512x512 numpy array) → watertight STL mesh.
Full fabrication resolution, Z-up, flat bottom.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import trimesh
from trimesh.creation import triangulate_polygon

import numpy as np


@dataclass
class GeometryConfig:
    resolution: int = 512          # heightfield grid size
    mesh_resolution: int = 350     # downsample to this for STL (keeps face count < 500K)
    physical_size_mm: float = 50.0 # physical XY extent in mm
    max_height_mm: float = 5.0     # Z range in mm
    base_thickness_mm: float = 2.0 # flat bottom thickness in mm


def heightfield_to_mesh(
    heightfield: np.ndarray,
    config: GeometryConfig | None = None,
) -> "trimesh.Trimesh":
    """
    Convert a 2D heightfield array to a watertight trimesh.

    Args:
        heightfield: float32 (H, W) array, values in [0, 1].
        config: GeometryConfig.

    Returns:
        trimesh.Trimesh: watertight solid mesh.
    """
    

    import cv2

    if config is None:
        config = GeometryConfig()

    # Downsample heightfield to mesh_resolution to control face count
    mr = config.mesh_resolution
    if heightfield.shape[0] != mr or heightfield.shape[1] != mr:
        heightfield = cv2.resize(heightfield, (mr, mr), interpolation=cv2.INTER_AREA)

    h, w = heightfield.shape
    dx = config.physical_size_mm / (w - 1)
    dy = config.physical_size_mm / (h - 1)

    # Build vertex grid
    xs = np.linspace(0, config.physical_size_mm, w)
    ys = np.linspace(0, config.physical_size_mm, h)
    xv, yv = np.meshgrid(xs, ys)

    z_top = heightfield * config.max_height_mm + config.base_thickness_mm
    z_bot = np.zeros_like(z_top)

    # Top surface vertices
    verts_top = np.stack([xv.ravel(), yv.ravel(), z_top.ravel()], axis=1)
    # Bottom surface vertices (flat)
    verts_bot = np.stack([xv.ravel(), yv.ravel(), z_bot.ravel()], axis=1)

    n = w * h
    vertices = np.vstack([verts_top, verts_bot])  # (2n, 3)

    def grid_idx(row: int, col: int) -> int:
        return row * w + col

    faces = []

    # Top and bottom quad → 2 triangles each
    for r in range(h - 1):
        for c in range(w - 1):
            i00 = grid_idx(r, c)
            i10 = grid_idx(r + 1, c)
            i01 = grid_idx(r, c + 1)
            i11 = grid_idx(r + 1, c + 1)
            # Top (CCW when viewed from +Z)
            faces += [[i00, i01, i11], [i00, i11, i10]]
            # Bottom (CW when viewed from +Z → outward normal is -Z)
            faces += [[n + i00, n + i11, n + i01], [n + i00, n + i10, n + i11]]

    # Side walls
    for c in range(w - 1):
        # Front (row=0)
        i0, i1 = grid_idx(0, c), grid_idx(0, c + 1)
        faces += [[i0, n + i0, n + i1], [i0, n + i1, i1]]
        # Back (row=h-1)
        i0, i1 = grid_idx(h - 1, c), grid_idx(h - 1, c + 1)
        faces += [[i0, i1, n + i1], [i0, n + i1, n + i0]]

    for r in range(h - 1):
        # Left (col=0)
        i0, i1 = grid_idx(r, 0), grid_idx(r + 1, 0)
        faces += [[i0, i1, n + i1], [i0, n + i1, n + i0]]
        # Right (col=w-1)
        i0, i1 = grid_idx(r, w - 1), grid_idx(r + 1, w - 1)
        faces += [[i0, n + i0, n + i1], [i0, n + i1, i1]]

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int64),
        process=True,
    )
    return mesh


def save_stl(mesh: "trimesh.Trimesh", out_path: str | Path) -> None:
    """Export mesh as binary STL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"STL saved: {out_path}  ({len(mesh.faces):,} faces)")
