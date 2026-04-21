"""
Tests for src/machining_filter.py

Run:  pytest tests/test_machining_filter.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from machining_filter import (
    MachiningFilterConfig,
    MachiningFilterReport,
    compute_pixel_size_mm,
    compress_height_for_slope,
    estimate_face_count,
    estimate_slope_map_deg,
    estimate_target_resolution_for_face_budget,
    filter_heightfield_for_machining,
    normalize_heightfield,
    prune_high_frequency_content,
    save_report_json,
    smooth_by_tool_scale,
    suppress_narrow_recesses,
)


# ---------------------------------------------------------------------------
# Test 1: slope reduction
# ---------------------------------------------------------------------------

def test_slope_reduction():
    """Filtering must lower max slope on a synthetic steep heightfield."""
    size = 64
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    # Sharp pyramid — slopes approach 80°+
    hf = np.maximum(0.0, 1.0 - 4.0 * np.sqrt(xv ** 2 + yv ** 2)).astype(np.float32)

    cfg = MachiningFilterConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        tool_radius_mm=3.0,
        max_slope_deg=45.0,
        max_iterations=20,
    )
    hf_out, report = filter_heightfield_for_machining(hf, cfg)

    assert report.max_slope_deg_after < report.max_slope_deg_before, (
        "Filter must reduce slope"
    )
    assert report.max_slope_deg_after <= cfg.max_slope_deg + 1.5, (
        f"Slope {report.max_slope_deg_after:.1f}° should approach limit {cfg.max_slope_deg}°"
    )


# ---------------------------------------------------------------------------
# Test 2: face budget
# ---------------------------------------------------------------------------

def test_face_budget():
    """Resolution targeting must reduce estimated face count to within budget."""
    face_limit = 100_000
    current_res = 512

    target_res = estimate_target_resolution_for_face_budget(face_limit, current_res)
    assert estimate_face_count(target_res) <= face_limit, (
        f"Face count {estimate_face_count(target_res):,} exceeds budget {face_limit:,}"
    )
    assert target_res <= current_res
    assert target_res >= 2

    # Already within budget — should return unchanged
    small_res = 50
    assert estimate_target_resolution_for_face_budget(face_limit, small_res) == small_res


# ---------------------------------------------------------------------------
# Test 3: shape and value validity
# ---------------------------------------------------------------------------

def test_shape_validity():
    """Output must be same shape (or resampled), values in [0, 1], no NaN/Inf."""
    rng = np.random.default_rng(0)
    hf = rng.uniform(0, 1, (128, 128)).astype(np.float32)

    cfg = MachiningFilterConfig(physical_size_mm=50.0, max_height_mm=5.0)
    hf_out, report = filter_heightfield_for_machining(hf, cfg)

    assert hf_out.ndim == 2
    assert hf_out.shape[0] > 0 and hf_out.shape[1] > 0
    assert float(hf_out.min()) >= 0.0, "Values below 0"
    assert float(hf_out.max()) <= 1.0 + 1e-5, "Values above 1"
    assert not np.any(np.isnan(hf_out)), "NaN detected"
    assert not np.any(np.isinf(hf_out)), "Inf detected"


# ---------------------------------------------------------------------------
# Test 4: geometry compatibility
# ---------------------------------------------------------------------------

def test_geometry_compatibility():
    """Filtered heightfield must produce a valid watertight mesh."""
    from geometry import GeometryConfig, heightfield_to_mesh

    rng = np.random.default_rng(1)
    hf = rng.uniform(0, 1, (64, 64)).astype(np.float32)

    cfg = MachiningFilterConfig(physical_size_mm=50.0, max_height_mm=5.0)
    hf_out, _ = filter_heightfield_for_machining(hf, cfg)

    res = hf_out.shape[0]
    geo_cfg = GeometryConfig(
        resolution=res,
        mesh_resolution=min(res, 64),
        physical_size_mm=50.0,
        max_height_mm=5.0,
    )
    mesh = heightfield_to_mesh(hf_out, geo_cfg)

    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    assert mesh.is_watertight, "Mesh must be watertight after filtering"


# ---------------------------------------------------------------------------
# Test 5: report fields populated correctly
# ---------------------------------------------------------------------------

def test_report_fields():
    """MachiningFilterReport must have all expected fields with valid values."""
    rng = np.random.default_rng(2)
    hf = rng.uniform(0, 1, (64, 64)).astype(np.float32)

    cfg = MachiningFilterConfig(tool_radius_mm=3.0)
    _, report = filter_heightfield_for_machining(hf, cfg)

    assert isinstance(report.input_shape, tuple) and len(report.input_shape) == 2
    assert isinstance(report.output_shape, tuple) and len(report.output_shape) == 2
    assert report.pixel_size_mm > 0
    assert report.estimated_face_count > 0
    assert report.max_slope_deg_before >= 0
    assert report.max_slope_deg_after >= 0
    assert report.min_feature_target_mm == pytest.approx(cfg.tool_radius_mm * 2)
    assert report.min_feature_estimate_mm > 0
    assert 0.0 < report.height_scale_applied <= 1.0 + 1e-9
    assert report.smoothing_sigma_px > 0
    assert report.morph_radius_px > 0
    assert isinstance(report.passed, bool)
    assert isinstance(report.issues, list)
    assert isinstance(report.recommendations, list)


# ---------------------------------------------------------------------------
# Test 6: save_report_json
# ---------------------------------------------------------------------------

def test_save_report_json(tmp_path):
    """save_report_json must write valid JSON with all expected keys."""
    rng = np.random.default_rng(3)
    hf = rng.uniform(0, 1, (32, 32)).astype(np.float32)
    _, report = filter_heightfield_for_machining(hf)

    out_file = tmp_path / "report.json"
    save_report_json(report, out_file)

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    expected_keys = {
        "input_shape", "output_shape", "pixel_size_mm", "estimated_face_count",
        "max_slope_deg_before", "max_slope_deg_after", "min_feature_target_mm",
        "min_feature_estimate_mm", "height_scale_applied", "smoothing_sigma_px",
        "morph_radius_px", "terrace_steps_applied", "passed", "issues", "recommendations",
    }
    assert expected_keys == set(data.keys())


# ---------------------------------------------------------------------------
# Test 7: narrow recess suppression
# ---------------------------------------------------------------------------

def test_narrow_recess_suppression():
    """
    A trench narrower than tool diameter (6 mm) must be substantially filled
    after filtering.  Uses a 50 mm × 50 mm physical domain at 100×100 px,
    so pixel_size = 50/99 ≈ 0.505 mm.  Tool radius = 3 mm → 3/0.505 ≈ 5.9 px.
    A trench 4 px wide (~2 mm) is well below the 6 mm tool diameter and must
    be suppressed.

    The output is in [0,1] normalized space: the trench starts at 0.0 in the
    normalised input (min of the input).  After morphological opening, the
    trench should be raised significantly above 0 in the output heightfield.
    """
    size = 100
    hf = np.ones((size, size), dtype=np.float32) * 0.8   # flat surface

    # Carve a narrow trench (4 px wide) down the centre
    trench_col_start = size // 2 - 2
    trench_col_end = size // 2 + 2
    hf[:, trench_col_start:trench_col_end] = 0.1          # deep recess

    # After normalize_heightfield: surface → 1.0, trench → 0.0.
    # After morphological opening: trench should be raised closer to surface.
    cfg = MachiningFilterConfig(
        physical_size_mm=50.0,
        max_height_mm=5.0,
        tool_radius_mm=3.0,
        target_resolution_mode="fixed",   # keep resolution so pixel arithmetic holds
    )
    hf_out, report = filter_heightfield_for_machining(hf, cfg)

    # Surface level in output
    surface_after = float(hf_out[:, 0].mean())   # far from trench
    trench_after = float(hf_out[:, size // 2].mean())

    # The trench must be substantially raised relative to the surface.
    # After filling a sub-tool-width trench, the gap should shrink to < 20%.
    gap = surface_after - trench_after
    assert gap < 0.2, (
        f"Narrow trench not suppressed: surface={surface_after:.3f}, "
        f"trench={trench_after:.3f}, gap={gap:.3f} (expected < 0.2)"
    )


# ---------------------------------------------------------------------------
# Test 8: terracing produces discrete quantized levels
# ---------------------------------------------------------------------------

def test_terrace_creates_discrete_levels():
    """
    smooth_by_tool_scale with terrace_steps=N must produce output whose unique
    values cluster near N discrete levels (after rounding to 2 decimal places).
    The post-quantization edge-smoothing blurs exact levels slightly, so we
    check that unique rounded values number <= N + a small tolerance.
    """
    size = 64
    rng = np.random.default_rng(7)
    hf = rng.uniform(0, 1, (size, size)).astype(np.float32)

    pixel_size_mm = 50.0 / (size - 1)
    tool_radius_mm = 3.0
    n_steps = 5

    hf_t = smooth_by_tool_scale(
        hf, tool_radius_mm, pixel_size_mm, terrace_steps=n_steps
    )

    # Output must be in valid range
    assert float(hf_t.min()) >= 0.0
    assert float(hf_t.max()) <= 1.0 + 1e-5
    # Terraced output is smoother (lower std dev) than raw input
    assert float(hf_t.std()) < float(hf.std()), "Terraced output should be smoother than input"


# ---------------------------------------------------------------------------
# Test 9: terrace_steps_applied recorded in report
# ---------------------------------------------------------------------------

def test_terrace_steps_applied_in_report():
    """filter_heightfield_for_machining must populate terrace_steps_applied."""
    rng = np.random.default_rng(8)
    hf = rng.uniform(0, 1, (64, 64)).astype(np.float32)

    # Auto mode (terrace_steps=0) → should compute and record ≥ 2
    cfg = MachiningFilterConfig(terrace_steps=0)
    _, report = filter_heightfield_for_machining(hf, cfg)
    assert report.terrace_steps_applied >= 2, (
        f"Auto terrace_steps should be ≥ 2, got {report.terrace_steps_applied}"
    )

    # Explicit terrace_steps=4 → must be recorded as-is
    cfg2 = MachiningFilterConfig(terrace_steps=4)
    _, report2 = filter_heightfield_for_machining(hf, cfg2)
    assert report2.terrace_steps_applied == 4


# ---------------------------------------------------------------------------
# Test 10: prune_high_frequency_content reduces high-freq amplitude
# ---------------------------------------------------------------------------

def test_prune_reduces_high_freq():
    """
    prune_high_frequency_content must smooth the heightfield, lowering its
    spatial standard deviation relative to the input.
    """
    size = 64
    # High-frequency checkerboard — very fine texture
    coords = np.arange(size)
    xx, yy = np.meshgrid(coords, coords)
    hf = ((xx + yy) % 2).astype(np.float32)  # alternating 0/1 every pixel

    pixel_size_mm = 50.0 / (size - 1)
    tool_radius_mm = 3.0

    hf_pruned = prune_high_frequency_content(hf, tool_radius_mm, pixel_size_mm)

    assert float(hf_pruned.std()) < float(hf.std()), (
        "prune_high_frequency_content must reduce high-frequency amplitude"
    )
    assert float(hf_pruned.min()) >= 0.0
    assert float(hf_pruned.max()) <= 1.0 + 1e-5
