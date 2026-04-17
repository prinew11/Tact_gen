"""
Mockup module: low-res preview mesh for visual confirmation before CAM.
Input : heightfield numpy array (512x512)
Output: .obj file at 256x256 resolution with z-exaggeration (2x)
NOT the fabrication mesh — that stays at full 512x512 resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_MOCKUP_RES = 256
_Z_SCALE = 2.0


def generate_mockup(
    heightfield: np.ndarray,
    out_path: str | Path,
    physical_size_mm: float = 50.0,
    max_height_mm: float = 5.0,
) -> Path:
    """
    Downsample heightfield to 256x256, apply 2x Z-exaggeration, export as OBJ.

    Args:
        heightfield: float32 (512, 512) array in [0, 1].
        out_path: destination .obj file path.
        physical_size_mm: XY physical extent in mm.
        max_height_mm: Z range before exaggeration, in mm.

    Returns:
        Path to the written .obj file.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Downsample
    small = cv2.resize(heightfield, (_MOCKUP_RES, _MOCKUP_RES), interpolation=cv2.INTER_AREA)

    h, w = small.shape
    xs = np.linspace(0, physical_size_mm, w)
    ys = np.linspace(0, physical_size_mm, h)
    xv, yv = np.meshgrid(xs, ys)
    zv = small * max_height_mm * _Z_SCALE

    with open(out_path, "w") as f:
        f.write("# Tactile mockup preview (256x256, z_scale=2.0)\n")
        # Vertices
        for r in range(h):
            for c in range(w):
                f.write(f"v {xv[r, c]:.4f} {yv[r, c]:.4f} {zv[r, c]:.4f}\n")
        # Faces (1-indexed)
        for r in range(h - 1):
            for c in range(w - 1):
                i00 = r * w + c + 1
                i10 = (r + 1) * w + c + 1
                i01 = r * w + (c + 1) + 1
                i11 = (r + 1) * w + (c + 1) + 1
                f.write(f"f {i00} {i01} {i11}\n")
                f.write(f"f {i00} {i11} {i10}\n")

    print(f"Mockup OBJ saved: {out_path}")
    return out_path


def render_mockup(obj_path: str | Path) -> None:
    """Quick matplotlib render of the mockup OBJ for visual confirmation."""
    import matplotlib.pyplot as plt

    vertices, faces = [], []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                _, a, b, c = line.split()
                faces.append((int(a) - 1, int(b) - 1, int(c) - 1))

    vertices = np.array(vertices)
    z = vertices[:, 2].reshape(_MOCKUP_RES, _MOCKUP_RES)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(z, origin="lower", cmap="gray")
    plt.colorbar(im, ax=ax, label="Height (mm × 2 exaggeration)")
    ax.set_title("Mockup preview")
    plt.tight_layout()
    plt.show()
