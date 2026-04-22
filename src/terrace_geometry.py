"""
Contour-based terrace geometry: heightfield -> watertight terrace STL.

Geometry model:
  - Heightfield is quantized to N discrete levels.
  - Every pixel is a flat platform at its level's z height.
  - Transitions between adjacent levels are 90-degree vertical risers.
  - Bottom is a fan-triangulated flat face at z = 0.
  - Outer perimeter is a set of vertical walls from z = 0 to the pixel height.

Hard manufacturability rule (enforced before geometry build):
  Any recessed feature whose XY width is <= tool_diameter_mm cannot be machined
  with a 6 mm ball-end mill and is filled / raised using binary morphological
  CLOSING on the per-level occupancy mask.  Default tool_diameter_mm = 6.0.

No slope optimisation is performed here.  The caller is responsible for any
diffusion / fabrication-correction pre-processing before calling
heightfield_to_terrace_mesh().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Config / report dataclasses
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
# Internal helpers
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

    Iterates to convergence. Every iteration that sets changed=True also
    strictly raises at least one label value, so the loop terminates.
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

    Strategy: for each level L (highest to lowest), the binary mask
    (labels >= L) represents all pixels at level L or above.  Narrow holes
    in this mask are depressions too small for the tool to reach.
    Morphological CLOSING (dilation then erosion) fills those holes, and any
    pixel that transitions from below-L to at-or-above-L is raised to level L.

    Processing top-down ensures fills accumulate without oscillation.
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
    #   top faces:       h * w * 2  triangles, 4 unique verts each → h*w*4
    #   horizontal risers: (w-1)*h  pairs → up to (w-1)*h * 2 tris, 4 verts each
    #   vertical risers:   w*(h-1)  pairs → same
    #   perimeter walls:   2*(h+w)  columns/rows → 2*(h+w) * 2 tris, 4 verts each
    #   bottom fan:        2*(h+w)  triangles, 1 center + 2*(h+w)+1 perimeter verts
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
    #   Pixel (pr, pc) occupies [xc(pc), xc(pc+1)] x [yr(pr), yr(pr+1)].
    #   Winding CCW from +Z → normals +Z.
    #   Verified: triangle (tl, tr, br): cross((tr-tl),(br-tl)) = (0,0,+dx*dy) → +Z.
    # ---------------------------------------------------------------------------
    for pr in range(h):
        for pc in range(w):
            z = z_table[labels[pr, pc]]
            tl = av(xc(pc),     yr(pr),     z)
            tr = av(xc(pc + 1), yr(pr),     z)
            br = av(xc(pc + 1), yr(pr + 1), z)
            bl = av(xc(pc),     yr(pr + 1), z)
            aq(tl, tr, br, bl)   # (tl,tr,br), (tl,br,bl) → +Z normal

    # ---------------------------------------------------------------------------
    # 3b  Internal vertical risers (horizontal adjacency: (pr, pc) vs (pr, pc+1)).
    #
    #   Wall at x = xc(pc+1), y in [yr(pr), yr(pr+1)], z from z_lo to z_hi.
    #   If la < lb (left lower): outward normal = -X → aq(v0, v3, v2, v1).
    #   If la > lb (right lower): outward normal = +X → aq(v0, v1, v2, v3).
    #   Verified by cross product on first triangle of each winding.
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
    # 3c  Internal vertical risers (vertical adjacency: (pr, pc) vs (pr+1, pc)).
    #
    #   Wall at y = yr(pr+1), x in [xc(pc), xc(pc+1)], z from z_lo to z_hi.
    #   If la > lb (top pixel higher): outward normal = +Y → aq(v0, v3, v2, v1).
    #   If la < lb (bottom pixel higher): outward normal = -Y → aq(v0, v1, v2, v3).
    #   Verified by cross product on first triangle of each winding.
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
    # Pre-create all z=0 perimeter vertices once so that the outer walls (3d)
    # and the bottom fan (3e) can share the SAME vertex indices — no merging needed.
    #
    #   bot[(pr, pc)] holds the vertex index for grid corner (pr, pc) at z=0.
    #   Front/back rows cover all w+1 columns including the four corners.
    #   Left/right columns skip the corner rows (already in front/back).
    # ---------------------------------------------------------------------------
    W_mm = xc(w)
    H_mm = yr(h)

    bot: dict[tuple[int, int], int] = {}
    for pc in range(w + 1):
        bot[(0, pc)] = av(xc(pc), yr(0), 0.0)   # front row
        bot[(h, pc)] = av(xc(pc), yr(h), 0.0)   # back row
    for pr in range(1, h):                        # left/right, skip corners
        bot[(pr, 0)] = av(xc(0), yr(pr), 0.0)
        bot[(pr, w)] = av(xc(w), yr(pr), 0.0)

    # ---------------------------------------------------------------------------
    # 3d  Outer perimeter walls — z = 0 up to pixel height.
    #
    #   Uses pre-created bot[] indices for the bottom edge so that wall and fan
    #   share the same vertices (no trimesh merge needed for these edges).
    #
    #   Front (y = 0):  normal = -Y → aq(b0, b1, v2, v3)
    #   Back  (y = H):  normal = +Y → aq(b0, v3, v2, b1)
    #   Left  (x = 0):  normal = -X → aq(b0, v3, v2, b1)
    #   Right (x = W):  normal = +X → aq(b0, b1, v2, v3)
    # ---------------------------------------------------------------------------
    for pc in range(w):
        # Front row (pr = 0), y = yr(0)
        z = z_table[labels[0, pc]]
        b0, b1 = bot[(0, pc)], bot[(0, pc + 1)]
        v2 = av(xc(pc + 1), yr(0), z)
        v3 = av(xc(pc),     yr(0), z)
        aq(b0, b1, v2, v3)   # -Y

        # Back row (pr = h-1), y = yr(h)
        z = z_table[labels[h - 1, pc]]
        b0, b1 = bot[(h, pc)], bot[(h, pc + 1)]
        v2 = av(xc(pc + 1), yr(h), z)
        v3 = av(xc(pc),     yr(h), z)
        aq(b0, v3, v2, b1)   # +Y

    for pr in range(h):
        # Left col (pc = 0), x = xc(0)
        z = z_table[labels[pr, 0]]
        b0, b1 = bot[(pr, 0)], bot[(pr + 1, 0)]
        v2 = av(xc(0), yr(pr + 1), z)
        v3 = av(xc(0), yr(pr),     z)
        aq(b0, v3, v2, b1)   # -X

        # Right col (pc = w-1), x = xc(w)
        z = z_table[labels[pr, w - 1]]
        b0, b1 = bot[(pr, w)], bot[(pr + 1, w)]
        v2 = av(xc(w), yr(pr + 1), z)
        v3 = av(xc(w), yr(pr),     z)
        aq(b0, b1, v2, v3)   # +X

    # ---------------------------------------------------------------------------
    # 3e  Bottom face — fan from center, perimeter traversed for normal = -Z.
    #
    #   Uses the same bot[] vertex indices as the perimeter walls so every
    #   bottom edge is shared exactly — no floating-point merge needed.
    #
    #   Traversal: LEFT edge UP → BACK edge RIGHT → RIGHT edge DOWN → FRONT LEFT.
    #   fan(p_i, p_j) = at(cx_idx, p_i, p_j) → -Z normal for this winding.
    # ---------------------------------------------------------------------------
    cx_idx = av(W_mm / 2.0, H_mm / 2.0, 0.0)

    def fan(p_i: int, p_j: int) -> None:
        at(cx_idx, p_i, p_j)

    # Left edge UP: (pr=0,pc=0) → (pr=1,pc=0) → ... → (pr=h,pc=0)
    for pr in range(h):
        fan(bot[(pr, 0)], bot[(pr + 1, 0)])

    # Back edge RIGHT: (pr=h,pc=0) → (pr=h,pc=1) → ... → (pr=h,pc=w)
    for pc in range(w):
        fan(bot[(h, pc)], bot[(h, pc + 1)])

    # Right edge DOWN: (pr=h,pc=w) → (pr=h-1,pc=w) → ... → (pr=0,pc=w)
    for pr in range(h):
        fan(bot[(h - pr, w)], bot[(h - 1 - pr, w)])

    # Front edge LEFT: (pr=0,pc=w) → (pr=0,pc=w-1) → ... → (pr=0,pc=0)
    for pc in range(w):
        fan(bot[(0, w - pc)], bot[(0, w - 1 - pc)])

    # ---------------------------------------------------------------------------
    # Assemble mesh.
    # ---------------------------------------------------------------------------
    mesh = trimesh.Trimesh(
        vertices=vbuf[:nv],
        faces=fbuf[:nf],
        process=True,          # merges coincident vertices, fixes winding
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
# Terrace-aware preprocessing (replaces slope-driven machining_filter steps)
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
      3. Mild high-frequency pruning (Gaussian sigma = 0.5 * tool_radius_px).
      4. Morphological opening on the heightfield to suppress continuous narrow
         recesses narrower than tool_diameter_mm (grey morphology, same as the
         machining_filter's suppress_narrow_recesses step).

    No slope optimisation, no terracing, no Gaussian riser blurring.
    Quantisation is performed inside heightfield_to_terrace_mesh().
    """
    from scipy.ndimage import gaussian_filter, grey_erosion, grey_dilation

    hf = np.clip(heightfield, 0.0, 1.0).astype(np.float32)
    lo, hi = float(hf.min()), float(hf.max())
    if hi - lo > 1e-7:
        hf = (hf - lo) / (hi - lo)

    # Downsample
    h0, w0 = hf.shape
    if h0 != target_resolution or w0 != target_resolution:
        hf = cv2.resize(hf, (target_resolution, target_resolution),
                        interpolation=cv2.INTER_AREA)

    px_size = physical_size_mm / (target_resolution - 1)
    tool_radius_mm = tool_diameter_mm / 2.0
    tool_radius_px = tool_radius_mm / px_size

    # Mild noise prune (half-tool-radius sigma)
    sigma_prune = max(tool_radius_px * 0.5, 0.5)
    hf = gaussian_filter(hf, sigma=sigma_prune)
    hf = np.clip(hf, 0.0, 1.0)

    # Morphological opening — suppress sub-tool-diameter recesses
    r = max(int(math.ceil(tool_radius_px)), 1)
    yi, xi = np.ogrid[-r : r + 1, -r : r + 1]
    disk = (xi ** 2 + yi ** 2 <= r ** 2)
    inv = (1.0 - hf).astype(np.float32)
    inv_eroded = grey_erosion(inv, footprint=disk)
    inv_opened = grey_dilation(inv_eroded, footprint=disk)
    hf = np.clip(1.0 - inv_opened, 0.0, 1.0).astype(np.float32)

    return hf


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_stl(mesh: trimesh.Trimesh, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"Terrace STL saved: {out_path}  ({len(mesh.faces):,} faces)")
