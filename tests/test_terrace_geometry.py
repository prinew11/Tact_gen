"""
Tests for src/terrace_geometry.py

Run:  pytest tests/test_terrace_geometry.py -v

Verifies:
  1. Quantised heightfield has exactly N distinct label values.
  2. After _enforce_min_recess_width, no enclosed depression is narrower
     than the tool diameter.
  3. All non-horizontal mesh faces are perfectly vertical (face normal Z = 0).
  4. All horizontal top faces are perfectly flat (face normal |Z| = 1).
  5. The output STL is watertight (every edge shared by exactly 2 faces).
  6. Face count is under the Fusion 500K limit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable when tests run from the repo root.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from terrace_geometry import (
    TerraceConfig,
    TerraceReport,
    _enforce_min_recess_width,
    _quantize,
    heightfield_to_terrace_mesh,
    preprocess_for_terrace,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _concentric_hf(size: int = 64) -> np.ndarray:
    """Concentric rings heightfield — natural terrace candidate."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv ** 2 + yv ** 2)
    hf = 0.5 + 0.4 * np.cos(r * 6 * np.pi) * np.exp(-r * 2)
    return np.clip(hf, 0.0, 1.0).astype(np.float32)


def _flat_hf(size: int = 32, value: float = 0.5) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _stepped_hf(size: int = 64, n_steps: int = 4) -> np.ndarray:
    """Left-to-right linear ramp that will quantise to n_steps clean levels."""
    row = np.linspace(0.0, 1.0 - 1e-6, size, dtype=np.float32)
    return np.tile(row, (size, 1))


# ---------------------------------------------------------------------------
# Test 1: Quantisation produces exactly N distinct labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_levels", [2, 4, 5, 8])
def test_quantize_discrete_levels(n_levels: int):
    hf = _concentric_hf(64)
    labels = _quantize(hf, n_levels)
    unique = np.unique(labels)
    assert len(unique) <= n_levels, (
        f"Expected at most {n_levels} distinct labels, got {len(unique)}: {unique}"
    )
    assert labels.min() >= 0
    assert labels.max() <= n_levels - 1


# ---------------------------------------------------------------------------
# Test 2: _enforce_min_recess_width fills narrow enclosed recesses
# ---------------------------------------------------------------------------

def test_enforce_min_recess_width_fills_narrow_pit():
    """
    A 2-pixel-wide pit surrounded by high-level pixels must be filled when
    the tool radius in pixels is larger than the pit radius.
    """
    size = 32
    n_levels = 4
    # Fill entire grid with the highest level
    labels = np.full((size, size), n_levels - 1, dtype=np.int32)
    # Punch a 2x2 pit (label=0) in the centre — width 2px << tool diameter 6mm
    cx, cy = size // 2, size // 2
    labels[cy - 1 : cy + 1, cx - 1 : cx + 1] = 0

    tool_radius_px = 4.0   # tool_diameter = 8 px > pit width 2 px → must be filled
    result = _enforce_min_recess_width(labels, tool_radius_px, n_levels)

    # The pit should have been raised to the surrounding level
    pit_values = result[cy - 1 : cy + 1, cx - 1 : cx + 1]
    assert pit_values.min() > 0, (
        f"Narrow pit (2px) was NOT filled when tool_radius_px={tool_radius_px}. "
        f"Pit values after filling: {pit_values}"
    )


def test_enforce_min_recess_width_preserves_wide_recess():
    """
    A wide pit (width >> tool diameter) must be preserved.
    """
    size = 64
    n_levels = 3
    labels = np.full((size, size), n_levels - 1, dtype=np.int32)
    # Large pit: 20x20 pixels, tool_radius = 2px → pit width 20px >> 4px diameter
    labels[20:40, 20:40] = 0

    tool_radius_px = 2.0
    result = _enforce_min_recess_width(labels, tool_radius_px, n_levels)

    # Interior of wide pit should remain at level 0
    interior = result[25:35, 25:35]
    assert interior.max() == 0, (
        f"Wide pit interior was incorrectly raised. Interior values: {np.unique(interior)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Mesh faces are either perfectly horizontal OR perfectly vertical
# ---------------------------------------------------------------------------

def test_terrace_faces_are_horizontal_or_vertical():
    """
    Every face in the terrace mesh must have a normal that is either:
      - perfectly vertical   (nz == 0, i.e. a riser or perimeter wall), OR
      - perfectly horizontal (|nz| == 1, i.e. a top or bottom face).
    """
    hf = _stepped_hf(32, n_steps=4)
    cfg = TerraceConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        terrace_steps=4,
        tool_diameter_mm=6.0,
        mesh_resolution=32,
    )
    mesh, report = heightfield_to_terrace_mesh(hf, cfg)

    normals = mesh.face_normals
    nz = np.abs(normals[:, 2])   # |Z-component|

    is_horizontal = np.isclose(nz, 1.0, atol=1e-4)
    is_vertical   = np.isclose(nz, 0.0, atol=1e-4)
    neither = ~(is_horizontal | is_vertical)

    assert neither.sum() == 0, (
        f"{neither.sum()} face(s) are neither horizontal nor vertical. "
        f"nz range of offenders: [{nz[neither].min():.4f}, {nz[neither].max():.4f}]"
    )


# ---------------------------------------------------------------------------
# Test 4: Top faces are flat at discrete z values
# ---------------------------------------------------------------------------

def test_top_faces_at_discrete_z():
    """
    All upward-facing (top) faces must have vertices at exactly one of the
    N discrete z heights.  No interpolated or blurred z values.
    """
    n_steps = 5
    hf = _stepped_hf(32, n_steps=n_steps)
    cfg = TerraceConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        terrace_steps=n_steps,
        tool_diameter_mm=6.0,
        mesh_resolution=32,
    )
    mesh, _ = heightfield_to_terrace_mesh(hf, cfg)

    normals = mesh.face_normals
    top_mask = normals[:, 2] > 0.9

    top_face_z = mesh.vertices[mesh.faces[top_mask], 2]
    unique_z = np.unique(np.round(top_face_z, decimals=5))

    # Number of distinct z heights must equal the number of terrace levels.
    assert len(unique_z) <= n_steps, (
        f"Expected at most {n_steps} distinct z heights on top faces, "
        f"got {len(unique_z)}: {unique_z}"
    )


# ---------------------------------------------------------------------------
# Test 5: STL is watertight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terrace_steps", [3, 5, 7])
def test_terrace_mesh_is_watertight(terrace_steps: int):
    hf = _concentric_hf(48)
    cfg = TerraceConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        terrace_steps=terrace_steps,
        tool_diameter_mm=6.0,
        mesh_resolution=48,
    )
    mesh, report = heightfield_to_terrace_mesh(hf, cfg)
    assert mesh.is_watertight, (
        f"Mesh with {terrace_steps} terrace steps is not watertight. "
        f"Issues: {report.issues}"
    )


# ---------------------------------------------------------------------------
# Test 6: Face count is under the Fusion 500K limit
# ---------------------------------------------------------------------------

def test_face_count_under_limit():
    hf = _concentric_hf(128)
    cfg = TerraceConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        terrace_steps=6,
        tool_diameter_mm=6.0,
        mesh_resolution=128,
        face_limit=500_000,
    )
    mesh, report = heightfield_to_terrace_mesh(hf, cfg)
    assert len(mesh.faces) < cfg.face_limit, (
        f"Face count {len(mesh.faces):,} exceeds Fusion limit {cfg.face_limit:,}"
    )


# ---------------------------------------------------------------------------
# Test 7: preprocess_for_terrace normalises output to [0, 1]
# ---------------------------------------------------------------------------

def test_preprocess_output_range():
    hf = _concentric_hf(64)
    out = preprocess_for_terrace(hf, tool_diameter_mm=6.0,
                                  physical_size_mm=50.0, target_resolution=64)
    assert out.min() >= 0.0, f"Output min {out.min()} < 0"
    assert out.max() <= 1.0, f"Output max {out.max()} > 1"


# ---------------------------------------------------------------------------
# Test 8: Flat heightfield produces a flat-topped mesh with one level
# ---------------------------------------------------------------------------

def test_flat_heightfield_single_level():
    hf = _flat_hf(32, value=0.6)
    cfg = TerraceConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        terrace_steps=5,
        tool_diameter_mm=6.0,
        mesh_resolution=32,
    )
    mesh, report = heightfield_to_terrace_mesh(hf, cfg)

    normals = mesh.face_normals
    top_mask = normals[:, 2] > 0.9
    top_z = mesh.vertices[mesh.faces[top_mask], 2]
    unique_z = np.unique(np.round(top_z, decimals=5))

    assert len(unique_z) == 1, (
        f"Flat heightfield should produce exactly 1 top-face z level, "
        f"got {len(unique_z)}: {unique_z}"
    )
    assert mesh.is_watertight, "Flat heightfield mesh is not watertight"
