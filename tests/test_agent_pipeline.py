"""Tests for the agent pipeline: toolkit, analyzer, and tool dispatch."""
import numpy as np
import pytest

import heightmap_toolkit as htk
from heightmap_analyzer import analyze_heightmap, analysis_to_text, compare_analyses, analysis_to_json
from agent_tools import execute_tool, quick_machinability_check, TOOL_SCHEMAS


@pytest.fixture
def sample_hf():
    """512x512 float32 heightmap in [0, 1]."""
    rng = np.random.RandomState(42)
    hf = rng.rand(512, 512).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    hf = gaussian_filter(hf, sigma=8)
    lo, hi = hf.min(), hf.max()
    return ((hf - lo) / (hi - lo)).astype(np.float32)


@pytest.fixture
def small_hf():
    """64x64 float32 heightmap for fast tests."""
    rng = np.random.RandomState(0)
    hf = rng.rand(64, 64).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    hf = gaussian_filter(hf, sigma=3)
    lo, hi = hf.min(), hf.max()
    return ((hf - lo) / (hi - lo)).astype(np.float32)


# ── Identity tests (neutral params return unchanged) ────────────────────────

class TestIdentity:
    def test_boost_contrast_gamma_1(self, small_hf):
        out = htk.boost_contrast(small_hf, gamma=1.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_enhance_ridges_strength_0(self, small_hf):
        out = htk.enhance_ridges(small_hf, strength=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_add_perlin_noise_amplitude_0(self, small_hf):
        out = htk.add_perlin_noise(small_hf, amplitude=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_directional_warp_strength_0(self, small_hf):
        out = htk.directional_warp(small_hf, strength=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_anisotropic_emphasis_strength_0(self, small_hf):
        out = htk.anisotropic_emphasis(small_hf, strength=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_bend_contours_curvature_0(self, small_hf):
        out = htk.bend_contours(small_hf, curvature=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_blend_pattern_amplitude_0(self, small_hf):
        out = htk.blend_pattern(small_hf, amplitude=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_texture_overlay_amplitude_0(self, small_hf):
        out = htk.texture_overlay(small_hf, amplitude=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_blend_two_alpha_0(self, small_hf):
        other = np.zeros_like(small_hf)
        out = htk.blend_two(small_hf, other, alpha=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)

    def test_blend_two_alpha_1(self, small_hf):
        other = np.zeros_like(small_hf)
        out = htk.blend_two(small_hf, other, alpha=1.0)
        np.testing.assert_allclose(out, other, atol=1e-5)

    def test_mask_apply_feather_0(self, small_hf):
        mask = np.zeros_like(small_hf)
        modified = np.ones_like(small_hf)
        out = htk.mask_apply(small_hf, modified, mask, feather_px=0.0)
        np.testing.assert_allclose(out, small_hf, atol=1e-5)


# ── Output validity ─────────────────────────────────────────────────────────

class TestOutputValidity:
    """All ops must return float32, same shape, values in [0, 1]."""

    def _check(self, result, original):
        assert result.shape == original.shape, f"Shape {result.shape} != {original.shape}"
        assert result.dtype == np.float32, f"Dtype {result.dtype} != float32"
        assert result.min() >= 0.0, f"Min {result.min()} < 0"
        assert result.max() <= 1.0, f"Max {result.max()} > 1"

    def test_boost_contrast(self, small_hf):
        self._check(htk.boost_contrast(small_hf, gamma=2.0), small_hf)

    def test_bandpass_filter(self, small_hf):
        self._check(htk.bandpass_filter(small_hf, 0.05, 0.4), small_hf)

    def test_enhance_ridges(self, small_hf):
        self._check(htk.enhance_ridges(small_hf, strength=1.0), small_hf)

    def test_add_perlin_noise(self, small_hf):
        self._check(htk.add_perlin_noise(small_hf, amplitude=0.3), small_hf)

    def test_height_selective_transform(self, small_hf):
        self._check(htk.height_selective_transform(small_hf, valley_boost=0.2, ridge_boost=-0.1), small_hf)

    def test_height_redistribute(self, small_hf):
        self._check(htk.height_redistribute(small_hf, target_distribution="uniform"), small_hf)

    def test_directional_warp(self, small_hf):
        self._check(htk.directional_warp(small_hf, angle_deg=45, strength=10), small_hf)

    def test_anisotropic_emphasis(self, small_hf):
        self._check(htk.anisotropic_emphasis(small_hf, angle_deg=90, strength=0.8), small_hf)

    def test_bend_contours(self, small_hf):
        self._check(htk.bend_contours(small_hf, curvature=0.8), small_hf)

    def test_blend_pattern_waves(self, small_hf):
        self._check(htk.blend_pattern(small_hf, "waves", amplitude=0.3), small_hf)

    def test_blend_pattern_concentric(self, small_hf):
        self._check(htk.blend_pattern(small_hf, "concentric", amplitude=0.2), small_hf)

    def test_blend_pattern_crosshatch(self, small_hf):
        self._check(htk.blend_pattern(small_hf, "crosshatch", amplitude=0.15), small_hf)

    def test_texture_overlay_stochastic(self, small_hf):
        self._check(htk.texture_overlay(small_hf, "stochastic", amplitude=0.2), small_hf)

    def test_texture_overlay_fibers(self, small_hf):
        self._check(htk.texture_overlay(small_hf, "fibers", amplitude=0.15), small_hf)

    def test_height_redistribute_bimodal(self, small_hf):
        self._check(htk.height_redistribute(small_hf, "bimodal"), small_hf)

    def test_height_redistribute_gaussian(self, small_hf):
        self._check(htk.height_redistribute(small_hf, "gaussian"), small_hf)


# ── Composability ───────────────────────────────────────────────────────────

class TestComposability:
    """Chaining 5 ops should still produce valid output."""

    def test_chain_five_ops(self, small_hf):
        hf = small_hf.copy()
        hf = htk.boost_contrast(hf, gamma=1.5)
        hf = htk.enhance_ridges(hf, strength=0.4)
        hf = htk.directional_warp(hf, angle_deg=30, strength=5)
        hf = htk.add_perlin_noise(hf, amplitude=0.08)
        hf = htk.blend_pattern(hf, "waves", amplitude=0.1)
        assert hf.shape == small_hf.shape
        assert hf.dtype == np.float32
        assert 0.0 <= hf.min() and hf.max() <= 1.0

    def test_chain_modifies_content(self, small_hf):
        hf = small_hf.copy()
        hf = htk.boost_contrast(hf, gamma=2.5)
        hf = htk.enhance_ridges(hf, strength=1.0)
        hf = htk.directional_warp(hf, angle_deg=60, strength=15)
        assert not np.allclose(hf, small_hf, atol=0.01)


# ── Analyzer ────────────────────────────────────────────────────────────────

class TestAnalyzer:
    def test_analyze_returns_all_fields(self, sample_hf):
        report = analyze_heightmap(sample_hf)
        assert hasattr(report, "roughness")
        assert hasattr(report, "directionality")
        assert hasattr(report, "frequency")
        assert hasattr(report, "glcm_contrast")
        assert hasattr(report, "glcm_homogeneity")
        assert hasattr(report, "gradient_coherence")
        assert hasattr(report, "histogram_bimodality")

    def test_analysis_to_text_contains_metrics(self, sample_hf):
        report = analyze_heightmap(sample_hf)
        text = analysis_to_text(report)
        assert "Roughness" in text
        assert "Directionality" in text
        assert "GLCM" in text

    def test_compare_analyses(self, sample_hf):
        before = analyze_heightmap(sample_hf)
        modified = htk.boost_contrast(sample_hf, gamma=2.0)
        after = analyze_heightmap(modified)
        diff = compare_analyses(before, after)
        assert "→" in diff or "->" in diff or "↑" in diff or "↓" in diff

    def test_analysis_to_json(self, sample_hf):
        import json
        report = analyze_heightmap(sample_hf)
        j = json.loads(analysis_to_json(report))
        assert "roughness" in j
        assert isinstance(j["roughness"], float)


# ── Machinability check ─────────────────────────────────────────────────────

class TestMachinability:
    def test_flat_surface_machinable(self):
        hf = np.full((64, 64), 0.5, dtype=np.float32)
        check = quick_machinability_check(hf)
        assert check["max_slope_deg"] < 1.0
        # height_range is 0, so likely_machinable is False (range must be > 0.1)
        assert not check["likely_machinable"]

    def test_gentle_slope_machinable(self, small_hf):
        check = quick_machinability_check(small_hf)
        assert check["max_slope_deg"] < 80
        assert check["height_range"] > 0.1

    def test_extreme_slope_not_machinable(self):
        hf = np.zeros((64, 64), dtype=np.float32)
        hf[:32, :] = 0.0
        hf[32:, :] = 1.0
        check = quick_machinability_check(hf)
        # Step function has steep gradient at boundary
        assert check["max_slope_deg"] > 20


# ── Tool execution ──────────────────────────────────────────────────────────

class TestToolExecution:
    def test_analyze_tool(self, small_hf):
        hf, desc, done = execute_tool("analyze_heightmap", {}, small_hf, small_hf, analyze_heightmap(small_hf))
        assert hf is small_hf or np.array_equal(hf, small_hf)
        assert "Roughness" in desc
        assert not done

    def test_boost_contrast_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool("boost_contrast", {"gamma": 2.0}, small_hf, small_hf, analysis)
        assert hf.shape == small_hf.shape
        assert not done

    def test_accept_heightmap_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool("accept_heightmap", {"summary": "looks good"}, small_hf, small_hf, analysis)
        assert done
        assert "Accepted" in desc

    def test_evaluate_result_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool("evaluate_result", {"intent_keywords": ["rough"]}, small_hf, small_hf, analysis)
        assert not done
        assert "roughness" in desc.lower() or "rough" in desc.lower() or "INFO" in desc

    def test_unknown_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool("nonexistent_tool", {}, small_hf, small_hf, analysis)
        assert "Unknown" in desc
        assert not done


# ── Tool schemas ────────────────────────────────────────────────────────────

class TestSchemas:
    def test_all_schemas_have_required_fields(self):
        for schema in TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert "type" in schema["input_schema"]

    def test_schema_names_unique(self):
        names = [s["name"] for s in TOOL_SCHEMAS]
        assert len(names) == len(set(names))

    def test_accept_heightmap_in_schemas(self):
        names = {s["name"] for s in TOOL_SCHEMAS}
        assert "accept_heightmap" in names
        assert "evaluate_result" in names
        assert "analyze_heightmap" in names


# ── Bandpass filter edge cases ──────────────────────────────────────────────

class TestBandpass:
    def test_full_pass(self, small_hf):
        out = htk.bandpass_filter(small_hf, low_cutoff=0.0, high_cutoff=1.0)
        np.testing.assert_allclose(out, small_hf, atol=0.05)

    def test_narrow_band(self, small_hf):
        out = htk.bandpass_filter(small_hf, low_cutoff=0.1, high_cutoff=0.2)
        assert out.shape == small_hf.shape
        assert out.dtype == np.float32


# ── Determinism ─────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_perlin_noise_deterministic(self, small_hf):
        a = htk.add_perlin_noise(small_hf, amplitude=0.1, seed=123)
        b = htk.add_perlin_noise(small_hf, amplitude=0.1, seed=123)
        np.testing.assert_array_equal(a, b)

    def test_texture_deterministic(self, small_hf):
        a = htk.texture_overlay(small_hf, "stochastic", seed=99)
        b = htk.texture_overlay(small_hf, "stochastic", seed=99)
        np.testing.assert_array_equal(a, b)


# ── Creative Freedom Operations ─────────────────────────────────────────────

class TestCreativeOps:
    def _check(self, result, original):
        assert result.shape == original.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_replace_region_ellipse(self, small_hf):
        out = htk.replace_region(small_hf, target_value=0.8, shape="ellipse")
        self._check(out, small_hf)

    def test_replace_region_rectangle(self, small_hf):
        out = htk.replace_region(small_hf, target_value=0.2, shape="rectangle")
        self._check(out, small_hf)

    def test_replace_region_diamond(self, small_hf):
        out = htk.replace_region(small_hf, target_value=0.5, shape="diamond")
        self._check(out, small_hf)

    def test_generate_mask_ellipse(self):
        mask = htk.generate_mask(shape="ellipse", resolution=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_generate_mask_invert(self):
        mask = htk.generate_mask(shape="ellipse", resolution=64)
        inv = htk.generate_mask(shape="ellipse", invert=True, resolution=64)
        np.testing.assert_allclose(mask + inv, 1.0, atol=0.01)

    def test_generate_mask_rectangle(self):
        mask = htk.generate_mask(shape="rectangle", resolution=64)
        assert mask.shape == (64, 64)

    def test_height_zone_remap(self, small_hf):
        out = htk.height_zone_remap(small_hf, zone_low=0.0, zone_high=0.3,
                                     target_low=0.7, target_high=1.0)
        self._check(out, small_hf)

    def test_height_zone_remap_invert(self, small_hf):
        out = htk.height_zone_remap(small_hf, zone_low=0.0, zone_high=0.5,
                                     target_low=0.5, target_high=1.0)
        self._check(out, small_hf)

    def test_surface_warp_identity(self, small_hf):
        out = htk.surface_warp(small_hf, control_points=[])
        np.testing.assert_array_equal(out, small_hf)

    def test_surface_warp_with_points(self, small_hf):
        out = htk.surface_warp(small_hf, control_points=[(0.5, 0.5, 0.2, 0.3)])
        self._check(out, small_hf)

    def test_procedural_perlin(self):
        out = htk.procedural_generate("perlin", size=64)
        assert out.shape == (64, 64)
        assert out.dtype == np.float32

    def test_procedural_voronoi(self):
        out = htk.procedural_generate("voronoi", size=64)
        assert out.shape == (64, 64)

    def test_procedural_brick(self):
        out = htk.procedural_generate("brick", size=64)
        assert out.shape == (64, 64)

    def test_procedural_hex_grid(self):
        out = htk.procedural_generate("hex_grid", size=64)
        assert out.shape == (64, 64)

    def test_symmetry_x(self, small_hf):
        out = htk.symmetry_apply(small_hf, axis="x")
        assert out.shape == small_hf.shape
        out2 = htk.symmetry_apply(out, axis="x")
        np.testing.assert_array_equal(out2, small_hf)

    def test_symmetry_y(self, small_hf):
        out = htk.symmetry_apply(small_hf, axis="y")
        assert out.shape == small_hf.shape

    def test_symmetry_xy(self, small_hf):
        out = htk.symmetry_apply(small_hf, axis="xy")
        assert out.shape == small_hf.shape

    def test_symmetry_tile_quadrant(self, small_hf):
        out = htk.symmetry_apply(small_hf, axis="tile_quadrant")
        assert out.shape == small_hf.shape
        assert out.dtype == np.float32


# ── Frequency-Aware Operations ──────────────────────────────────────────────

class TestFrequencyAware:
    def _check(self, result, original):
        assert result.shape == original.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_freq_preserve_lowpass(self, small_hf):
        out = htk.freq_preserve_lowpass(small_hf, cutoff=0.15)
        self._check(out, small_hf)

    def test_freq_preserve_lowpass_smooths(self, small_hf):
        low = htk.freq_preserve_lowpass(small_hf, cutoff=0.15)
        gy_orig, gx_orig = np.gradient(small_hf)
        gy_low, gx_low = np.gradient(low)
        orig_grad = np.sqrt(gx_orig**2 + gy_orig**2).mean()
        low_grad = np.sqrt(gx_low**2 + gy_low**2).mean()
        assert low_grad < orig_grad

    def test_freq_stepped_convert(self, small_hf):
        out = htk.freq_stepped_convert(small_hf, n_levels=4)
        self._check(out, small_hf)

    def test_freq_stepped_quantizes(self, small_hf):
        out = htk.freq_stepped_convert(small_hf, n_levels=4)
        n_unique = len(np.unique(np.round(out.flatten(), 3)))
        n_orig = len(np.unique(np.round(small_hf.flatten(), 3)))
        assert n_unique < n_orig

    def test_freq_stepped_with_dither(self, small_hf):
        out = htk.freq_stepped_convert(small_hf, n_levels=4, dither=True)
        self._check(out, small_hf)

    def test_freq_band_boost_amplify(self, small_hf):
        out = htk.freq_band_boost(small_hf, band_low=0.0, band_high=0.15, gain=2.0)
        self._check(out, small_hf)

    def test_freq_band_boost_suppress(self, small_hf):
        out = htk.freq_band_boost(small_hf, band_low=0.0, band_high=0.15, gain=0.5)
        self._check(out, small_hf)

    def test_freq_band_boost_identity(self, small_hf):
        out = htk.freq_band_boost(small_hf, band_low=0.0, band_high=0.15, gain=1.0)
        np.testing.assert_allclose(out, small_hf, atol=0.01)


# ── Stored Masks Workflow ───────────────────────────────────────────────────

class TestStoredMasks:
    def test_generate_then_mask_apply(self, small_hf):
        mask = htk.generate_mask(shape="ellipse", resolution=64, feather_px=8.0)
        modified = np.ones_like(small_hf) * 0.9
        out = htk.mask_apply(small_hf, modified, mask, feather_px=0.0)
        assert out.shape == small_hf.shape
        assert out.dtype == np.float32

    def test_mask_apply_selective(self, small_hf):
        mask = np.zeros_like(small_hf)
        mask[16:48, 16:48] = 1.0
        modified = np.ones_like(small_hf)
        out = htk.mask_apply(small_hf, modified, mask, feather_px=0.0)
        np.testing.assert_allclose(out[:16, :], small_hf[:16, :], atol=1e-5)


# ── New Tool Schemas ────────────────────────────────────────────────────────

class TestNewSchemas:
    def test_new_tool_names_present(self):
        names = {s["name"] for s in TOOL_SCHEMAS}
        new_tools = [
            "replace_region", "generate_mask", "height_zone_remap",
            "surface_warp", "procedural_generate", "symmetry_apply",
            "freq_preserve_lowpass", "freq_stepped_convert", "freq_band_boost",
        ]
        for tool in new_tools:
            assert tool in names, f"Missing schema for {tool}"

    def test_total_schema_count(self):
        assert len(TOOL_SCHEMAS) == 25


# ── New Tool Execution ─────────────────────────────────────────────────────

class TestNewToolExecution:
    def test_replace_region_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool(
            "replace_region", {"target_value": 0.8, "shape": "ellipse"},
            small_hf, small_hf, analysis,
        )
        assert hf.shape == small_hf.shape
        assert not done

    def test_generate_mask_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        stored = {}
        hf, desc, done = execute_tool(
            "generate_mask", {"shape": "ellipse"},
            small_hf, small_hf, analysis, stored_masks=stored,
        )
        assert "last" in stored
        assert hf is small_hf

    def test_freq_preserve_lowpass_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool(
            "freq_preserve_lowpass", {"cutoff": 0.15},
            small_hf, small_hf, analysis,
        )
        assert hf.shape == small_hf.shape
        assert not done

    def test_freq_stepped_convert_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool(
            "freq_stepped_convert", {"n_levels": 4},
            small_hf, small_hf, analysis,
        )
        assert hf.shape == small_hf.shape
        assert not done

    def test_symmetry_apply_tool(self, small_hf):
        analysis = analyze_heightmap(small_hf)
        hf, desc, done = execute_tool(
            "symmetry_apply", {"axis": "x"},
            small_hf, small_hf, analysis,
        )
        assert hf.shape == small_hf.shape
