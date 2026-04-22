"""
Geometry conversion: heightfield (512x512 numpy array) → watertight STL mesh.
Full fabrication resolution, Z-up, flat bottom.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import trimesh

import numpy as np


@dataclass
class GeometryConfig:
    physical_size_mm: float = 100.0 # physical XY extent in mm (10 cm)
    max_height_mm: float = 10.0     # Z range in mm (1 cm)
    base_thickness_mm: float = 2.0 # flat bottom thickness in mm
    face_limit: int = 500_000      # Fusion CAM stability limit


def _target_resolution_for_face_budget(face_limit: int, current_res: int) -> int:
    """Largest R s.t. 4*(R-1)*(R+1) <= face_limit."""
    import math
    n = current_res - 1
    if 4 * n * (n + 2) <= face_limit:
        return current_res
    r_max = int(math.floor(math.sqrt(face_limit / 4.0 + 1.0)))
    return max(r_max, 2)


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

    target_res = _target_resolution_for_face_budget(
        config.face_limit, min(heightfield.shape)
    )
    if target_res < min(heightfield.shape):
        heightfield = cv2.resize(
            heightfield, (target_res, target_res), interpolation=cv2.INTER_AREA
        )

    h, w = heightfield.shape
    n = w * h

    # Vertex grid (fully vectorised)
    # Flip heightfield vertically so image row 0 (top) maps to world y_max (top).
    # Keep xs/ys ascending so face winding / normals stay correct.
    heightfield = np.flipud(heightfield)
    xs = np.linspace(0, config.physical_size_mm, w)
    ys = np.linspace(0, config.physical_size_mm, h)
    xv, yv = np.meshgrid(xs, ys)
    z_top = heightfield * config.max_height_mm + config.base_thickness_mm
    verts_top = np.stack([xv.ravel(), yv.ravel(), z_top.ravel()], axis=1)
    verts_bot = np.stack([xv.ravel(), yv.ravel(), np.zeros_like(z_top).ravel()], axis=1)
    vertices = np.vstack([verts_top, verts_bot]).astype(np.float64)

    # --- Vectorised face construction ---
    rr, cc = np.mgrid[:h - 1, :w - 1]
    i00 = (rr * w + cc).ravel()
    i01 = i00 + 1
    i10 = i00 + w
    i11 = i10 + 1

    top_faces = np.concatenate([
        np.stack([i00, i01, i11], axis=1),
        np.stack([i00, i11, i10], axis=1),
    ])
    # Bottom (reverse winding so normal points -Z)
    bot_faces = np.concatenate([
        np.stack([n + i00, n + i11, n + i01], axis=1),
        np.stack([n + i00, n + i10, n + i11], axis=1),
    ])

    # Side walls
    c_idx = np.arange(w - 1)
    r_idx = np.arange(h - 1)
    # Front (row=0)
    f0, f1 = c_idx, c_idx + 1
    front = np.concatenate([
        np.stack([f0, n + f0, n + f1], axis=1),
        np.stack([f0, n + f1, f1], axis=1),
    ])
    # Back (row=h-1)
    b0 = (h - 1) * w + c_idx
    b1 = b0 + 1
    back = np.concatenate([
        np.stack([b0, b1, n + b1], axis=1),
        np.stack([b0, n + b1, n + b0], axis=1),
    ])
    # Left (col=0)
    l0 = r_idx * w
    l1 = l0 + w
    left = np.concatenate([
        np.stack([l0, l1, n + l1], axis=1),
        np.stack([l0, n + l1, n + l0], axis=1),
    ])
    # Right (col=w-1)
    ri0 = r_idx * w + (w - 1)
    ri1 = ri0 + w
    right = np.concatenate([
        np.stack([ri0, n + ri0, n + ri1], axis=1),
        np.stack([ri0, n + ri1, ri1], axis=1),
    ])

    faces = np.concatenate([top_faces, bot_faces, front, back, left, right]).astype(np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # Decimate to merge coplanar faces on flat terrace regions.
    # Requires the optional fast_simplification package; skipped if absent.
    target = min(len(mesh.faces), config.face_limit)
    if len(mesh.faces) > target:
        try:
            mesh = mesh.simplify_quadric_decimation(target)
        except (ImportError, ModuleNotFoundError):
            pass

    return mesh


def save_stl(mesh: "trimesh.Trimesh", out_path: str | Path) -> None:
    """Export mesh as binary STL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"STL saved: {out_path}  ({len(mesh.faces):,} faces)")
