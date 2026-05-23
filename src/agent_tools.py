"""
Tool definitions and execution dispatch for LLM function-calling API.

Each tool maps to heightmap_toolkit operations. The LLM selects tools
and provides parameters; this module validates and executes them.
"""
from __future__ import annotations

import numpy as np

import heightmap_toolkit as htk
from heightmap_analyzer import analyze_heightmap, analysis_to_text, compare_analyses


TOOL_SCHEMAS = [
    {
        "name": "analyze_heightmap",
        "description": "Analyze the current heightmap. Returns texture metrics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "boost_contrast",
        "description": "Power-law contrast. gamma>1 sharpens, gamma<1 smooths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gamma": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                "midtone_offset": {"type": "number", "minimum": -0.5, "maximum": 0.5},
            },
            "required": ["gamma"],
        },
    },
    {
        "name": "bandpass_filter",
        "description": "FFT bandpass: keep spatial frequencies in [low, high].",
        "input_schema": {
            "type": "object",
            "properties": {
                "low_cutoff": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "high_cutoff": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["low_cutoff", "high_cutoff"],
        },
    },
    {
        "name": "enhance_ridges",
        "description": "LoG ridge enhancement. Amplifies linear features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strength": {"type": "number", "minimum": 0.0, "maximum": 4.0},
                "scale_px": {"type": "number", "minimum": 2.0, "maximum": 64.0},
            },
            "required": ["strength"],
        },
    },
    {
        "name": "add_perlin_noise",
        "description": "Add coherent Perlin noise for organic variation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amplitude": {"type": "number", "minimum": 0.0, "maximum": 0.8},
                "frequency": {"type": "number", "minimum": 1.0, "maximum": 32.0},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": ["amplitude"],
        },
    },
    {
        "name": "height_selective_transform",
        "description": "Different transforms for valleys vs ridges.",
        "input_schema": {
            "type": "object",
            "properties": {
                "valley_boost": {"type": "number", "minimum": -0.6, "maximum": 0.6},
                "ridge_boost": {"type": "number", "minimum": -0.6, "maximum": 0.6},
                "low_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "high_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [],
        },
    },
    {
        "name": "directional_warp",
        "description": "Sinusoidal contour bending for flowing ridges effect.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
                "strength": {"type": "number", "minimum": 0.0, "maximum": 50.0},
                "wavelength_px": {"type": "number", "minimum": 8.0, "maximum": 128.0},
            },
            "required": ["angle_deg"],
        },
    },
    {
        "name": "anisotropic_emphasis",
        "description": "Sharpen along angle, blur perpendicular.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_deg": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["angle_deg"],
        },
    },
    {
        "name": "bend_contours",
        "description": "Bend contours toward/away from a center point.",
        "input_schema": {
            "type": "object",
            "properties": {
                "curvature": {"type": "number", "minimum": -3.0, "maximum": 3.0},
                "center_x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "center_y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["curvature"],
        },
    },
    {
        "name": "blend_pattern",
        "description": "Blend procedural pattern (waves/concentric/radial/crosshatch).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern_type": {"type": "string", "enum": ["waves", "concentric", "radial", "crosshatch"]},
                "blend_mode": {"type": "string", "enum": ["add", "multiply", "overlay"]},
                "amplitude": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "frequency": {"type": "number", "minimum": 1.0, "maximum": 32.0},
                "angle_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
            },
            "required": ["pattern_type"],
        },
    },
    {
        "name": "texture_overlay",
        "description": "Fine texture: stochastic/fibers/cracks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "texture_type": {"type": "string", "enum": ["stochastic", "fibers", "cracks"]},
                "amplitude": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                "feature_size_px": {"type": "number", "minimum": 3.0, "maximum": 30.0},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": ["texture_type"],
        },
    },
    {
        "name": "mask_apply",
        "description": "Apply modifications only within a feathered mask. Use generate_mask first to create a mask.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feather_px": {"type": "number", "minimum": 0.0, "maximum": 64.0},
            },
            "required": [],
        },
    },
    {
        "name": "generate_mask",
        "description": "Generate a spatial mask from geometric primitives. Stored for use with mask_apply.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shape": {"type": "string", "enum": ["ellipse", "rectangle", "ring", "gradient_x", "gradient_y"]},
                "center_x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "center_y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "size_x": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                "size_y": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                "invert": {"type": "boolean"},
                "feather_px": {"type": "number", "minimum": 0.0, "maximum": 64.0},
            },
            "required": [],
        },
    },
    {
        "name": "height_zone_remap",
        "description": "Remap a specific height range to a new range. Can invert topography.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone_low": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "zone_high": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "target_low": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "target_high": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "blend_width": {"type": "number", "minimum": 0.01, "maximum": 0.3},
            },
            "required": [],
        },
    },
    # ── Frequency-Aware Tools ───────────────────────────────────────────────
    {
        "name": "freq_preserve_lowpass",
        "description": "Extract low-frequency base (large-area shape) via FFT lowpass. Use as foundation before adding stepped texture.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cutoff": {"type": "number", "minimum": 0.05, "maximum": 0.5},
            },
            "required": [],
        },
    },
    {
        "name": "freq_stepped_convert",
        "description": "Convert high-frequency undulations to machinable stepped stripes. Core frequency-aware operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_levels": {"type": "integer", "minimum": 2, "maximum": 12},
                "freq_low": {"type": "number", "minimum": 0.05, "maximum": 0.5},
                "freq_high": {"type": "number", "minimum": 0.3, "maximum": 1.0},
                "dither": {"type": "boolean"},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": [],
        },
    },
    {
        "name": "freq_band_boost",
        "description": "Boost or attenuate a specific frequency band. gain>1 amplifies, gain<1 suppresses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "band_low": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "band_high": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "gain": {"type": "number", "minimum": 0.1, "maximum": 5.0},
            },
            "required": [],
        },
    },
    # ── Region-Aware Directional Tools ─────────────────────────────────────
    {
        "name": "image_guided_ridge_restore",
        "description": "Restore ridge structures from original image where heightmap is too flat. "
                       "Use when analysis shows low height_std — the diffusion model often loses "
                       "fine ridge/valley detail that the original image carries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_blend_strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.8,
                    "description": "How much image structure to blend in [0, 0.8]. Higher = more restoration.",
                },
                "ridge_threshold": {
                    "type": "number",
                    "minimum": 0.005,
                    "maximum": 0.1,
                    "description": "Local std below which to restore ridges. Lower = more selective.",
                },
                "smooth_boundary_px": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 32.0,
                    "description": "Feathering width at blend boundaries in pixels.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "directional_step_convert",
        "description": "Convert parallel-stripe regions (Type B) into directional stepped ridges. "
                       "Quantizes by coordinate perpendicular to grain direction instead of by height. "
                       "Creates stepped ridges parallel to grain for CNC machining along grain direction. "
                       "Only apply to Type B regions detected in analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_deg": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 180.0,
                    "description": "Grain/ridge direction in degrees. Use region_b_angle_deg from analysis.",
                },
                "n_steps": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 12,
                    "description": "Number of discrete step levels across grain direction.",
                },
                "use_region_mask": {
                    "type": "boolean",
                    "description": "If true, automatically use the Type B region mask from analysis. "
                                   "If false, apply to entire heightfield.",
                },
            },
            "required": ["angle_deg"],
        },
    },
    # ── Meta Tools ──────────────────────────────────────────────────────────
    {
        "name": "evaluate_result",
        "description": "Evaluate current heightmap against user intent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["intent_keywords"],
        },
    },
    {
        "name": "accept_heightmap",
        "description": "Accept current heightmap as final result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
    {
        "name": "propose_edit_plan",
        "description": "Propose bounded enhancement parameters for the preprocessed heightmap. "
                       "All values are clamped to safe ranges before application.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ridge_boost": {
                    "type": "number",
                    "description": "Ridge enhancement strength, 0.0 to 0.35",
                },
                "contrast_boost": {
                    "type": "number",
                    "description": "Contrast boost amount, 0.0 to 0.25",
                },
                "texture_amount": {
                    "type": "number",
                    "description": "Texture overlay intensity, 0.0 to 0.10",
                },
                "smoothing_sigma": {
                    "type": "number",
                    "description": "Smoothing sigma, 0.0 to 1.5",
                },
                "target_regions": {
                    "type": "string",
                    "description": "Which regions to enhance: 'global', 'ridges', 'valleys', 'high', 'low'",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the enhancement plan",
                },
            },
            "required": ["summary"],
        },
    },
]


_DANGEROUS_TOOLS = frozenset({
    "procedural_generate", "blend_two", "surface_warp",
    "height_redistribute", "replace_region", "symmetry_apply",
})


def execute_tool(
    tool_name: str,
    tool_input: dict,
    current_hf: np.ndarray,
    original_hf: np.ndarray,
    analysis_before,
    stored_masks: dict | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Execute a toolkit operation. Returns (heightmap, description, is_final)."""
    if stored_masks is None:
        stored_masks = {}

    # Reject dangerous tools with explicit warning
    if tool_name in _DANGEROUS_TOOLS:
        return current_hf, f"REJECTED: {tool_name} is not permitted in bounded mode. Use propose_edit_plan.", False

    if tool_name == "analyze_heightmap":
        report = analyze_heightmap(current_hf)
        return current_hf, analysis_to_text(report), False

    elif tool_name == "boost_contrast":
        result = htk.boost_contrast(current_hf, **tool_input)
        return result, f"Applied boost_contrast(gamma={tool_input.get('gamma', 1.5)})", False

    elif tool_name == "bandpass_filter":
        result = htk.bandpass_filter(current_hf, **tool_input)
        return result, f"Applied bandpass_filter({tool_input})", False

    elif tool_name == "enhance_ridges":
        result = htk.enhance_ridges(current_hf, **tool_input)
        return result, f"Applied enhance_ridges(strength={tool_input.get('strength', 0.5)})", False

    elif tool_name == "add_perlin_noise":
        result = htk.add_perlin_noise(current_hf, **tool_input)
        return result, f"Applied add_perlin_noise({tool_input})", False

    elif tool_name == "height_selective_transform":
        result = htk.height_selective_transform(current_hf, **tool_input)
        return result, f"Applied height_selective_transform({tool_input})", False

    elif tool_name == "directional_warp":
        result = htk.directional_warp(current_hf, **tool_input)
        return result, f"Applied directional_warp({tool_input})", False

    elif tool_name == "anisotropic_emphasis":
        result = htk.anisotropic_emphasis(current_hf, **tool_input)
        return result, f"Applied anisotropic_emphasis({tool_input})", False

    elif tool_name == "bend_contours":
        result = htk.bend_contours(current_hf, **tool_input)
        return result, f"Applied bend_contours({tool_input})", False

    elif tool_name == "blend_pattern":
        result = htk.blend_pattern(current_hf, **tool_input)
        return result, f"Applied blend_pattern({tool_input})", False

    elif tool_name == "texture_overlay":
        result = htk.texture_overlay(current_hf, **tool_input)
        return result, f"Applied texture_overlay({tool_input})", False

    elif tool_name == "mask_apply":
        if "last" not in stored_masks:
            raise ValueError(
                "mask_apply requires a stored mask. Call generate_mask first. "
                "Global fallback is not permitted."
            )
        mask = stored_masks["last"]
        result = htk.mask_apply(original_hf, current_hf, mask, **tool_input)
        return result, f"Applied mask_apply(feather={tool_input.get('feather_px', 8.0)}, mask=stored)", False

    # ── Mask Tools ──────────────────────────────────────────────────────────
    elif tool_name == "generate_mask":
        mask = htk.generate_mask(**tool_input)
        stored_masks["last"] = mask
        return current_hf, f"Generated mask: shape={tool_input.get('shape', 'ellipse')}, stored for mask_apply", False

    elif tool_name == "height_zone_remap":
        result = htk.height_zone_remap(current_hf, **tool_input)
        return result, f"Applied height_zone_remap({tool_input})", False

    # ── Frequency-Aware Tools ───────────────────────────────────────────────
    elif tool_name == "freq_preserve_lowpass":
        result = htk.freq_preserve_lowpass(current_hf, **tool_input)
        return result, f"Applied freq_preserve_lowpass(cutoff={tool_input.get('cutoff', 0.15)})", False

    elif tool_name == "freq_stepped_convert":
        result = htk.freq_stepped_convert(current_hf, **tool_input)
        return result, f"Applied freq_stepped_convert({tool_input})", False

    elif tool_name == "freq_band_boost":
        result = htk.freq_band_boost(current_hf, **tool_input)
        return result, f"Applied freq_band_boost({tool_input})", False

    # ── Region-Aware Directional Tools ─────────────────────────────────────
    elif tool_name == "image_guided_ridge_restore":
        image_gray = stored_masks.get("_image_gray") if stored_masks else None
        if image_gray is None:
            return current_hf, "ERROR: image_guided_ridge_restore requires image_gray data (not available in this context)", False
        result = htk.image_guided_ridge_restore(
            current_hf,
            image_gray,
            raw_blend_strength=float(tool_input.get("raw_blend_strength", 0.5)),
            ridge_threshold=float(tool_input.get("ridge_threshold", 0.02)),
            smooth_boundary_px=float(tool_input.get("smooth_boundary_px", 8.0)),
        )
        return result, (
            f"Applied image_guided_ridge_restore("
            f"strength={tool_input.get('raw_blend_strength', 0.5):.2f}, "
            f"threshold={tool_input.get('ridge_threshold', 0.02):.4f})"
        ), False

    elif tool_name == "directional_step_convert":
        use_mask = tool_input.get("use_region_mask", True)
        region_mask = None
        if use_mask and analysis_before is not None:
            region_mask = getattr(analysis_before, "region_b_mask", None)
        result = htk.directional_step_convert(
            current_hf,
            angle_deg=float(tool_input.get("angle_deg", 0.0)),
            n_steps=int(tool_input.get("n_steps", 8)),
            mask=region_mask,
        )
        return result, (
            f"Applied directional_step_convert(angle={tool_input.get('angle_deg', 0.0):.1f}deg, "
            f"n_steps={tool_input.get('n_steps', 8)}, "
            f"mask={'region_b' if region_mask is not None else 'none'})"
        ), False

    # ── Meta Tools ──────────────────────────────────────────────────────────
    elif tool_name == "evaluate_result":
        after = analyze_heightmap(current_hf)
        comparison = compare_analyses(analysis_before, after)
        keywords = tool_input.get("intent_keywords", [])
        feedback = _evaluate_against_intent(analysis_before, after, keywords)
        return current_hf, f"{comparison}\n\n{feedback}", False

    elif tool_name == "accept_heightmap":
        summary = tool_input.get("summary", "Accepted")
        return current_hf, f"Accepted: {summary}", True

    elif tool_name == "propose_edit_plan":
        summary = tool_input.get("summary", "Plan proposed")
        stored_masks["_plan_proposal"] = {
            "ridge_boost": tool_input.get("ridge_boost", 0.0),
            "contrast_boost": tool_input.get("contrast_boost", 0.0),
            "texture_amount": tool_input.get("texture_amount", 0.0),
            "smoothing_sigma": tool_input.get("smoothing_sigma", 0.0),
            "target_regions": tool_input.get("target_regions", "global"),
            "summary": summary,
        }
        return current_hf, f"Plan proposed: {summary}", True

    else:
        return current_hf, f"Unknown tool: {tool_name}", False


def _evaluate_against_intent(before, after, keywords: list[str]) -> str:
    """Evaluate whether the transformation moved in the right direction."""
    feedback = []
    for kw in keywords:
        kw = kw.lower()
        if kw in ("rough", "roughness"):
            if after.roughness > before.roughness:
                feedback.append(f"GOOD: Roughness {before.roughness:.2f} -> {after.roughness:.2f}")
            else:
                feedback.append(f"WARN: Roughness decreased {before.roughness:.2f} -> {after.roughness:.2f}")
        elif kw in ("smooth", "soft"):
            if after.roughness < before.roughness:
                feedback.append(f"GOOD: Roughness decreased {before.roughness:.2f} -> {after.roughness:.2f}")
        elif kw in ("flowing", "directional"):
            if after.directionality > before.directionality:
                feedback.append(f"GOOD: Directionality {before.directionality:.2f} -> {after.directionality:.2f}")
            else:
                feedback.append(f"WARN: Directionality decreased")
        elif kw in ("deep", "channel"):
            if after.height_range > before.height_range:
                feedback.append(f"GOOD: Height range {before.height_range:.3f} -> {after.height_range:.3f}")
        elif kw in ("organic", "natural"):
            feedback.append(f"INFO: Bimodality={after.histogram_bimodality:.2f}, skewness={after.histogram_skewness:.2f}")
    if not feedback:
        feedback.append("INFO: No specific intent keywords matched")
    return "\n".join(feedback)


def quick_machinability_check(hf: np.ndarray) -> dict:
    """Fast pre-check before terrace processing."""
    gy, gx = np.gradient(hf.astype(np.float32))
    max_slope = float(np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2).max())))
    height_range = float(hf.max() - hf.min())
    return {
        "max_slope_deg": max_slope,
        "height_range": height_range,
        "likely_machinable": max_slope < 80 and height_range > 0.1,
    }


def extract_plan_parameters(
    tool_calls: list[dict],
    stored_masks: dict,
) -> dict:
    """Extract bounded edit parameters from tool calls and stored state.

    Primary path: read from stored_masks["_plan_proposal"] if propose_edit_plan was called.
    Fallback: scan tool calls for parameter values.

    Returns dict with keys: ridge_boost, contrast_boost, texture_amount,
    smoothing_sigma, target_regions, notes.
    """
    # Safe ranges
    CLAMP = {
        "ridge_boost": (0.0, 0.35),
        "contrast_boost": (0.0, 0.25),
        "texture_amount": (0.0, 0.10),
        "smoothing_sigma": (0.0, 1.5),
    }

    plan: dict = {
        "ridge_boost": 0.0,
        "contrast_boost": 0.0,
        "texture_amount": 0.0,
        "smoothing_sigma": 0.0,
        "target_regions": "global",
        "directional_step": None,  # {"angle_deg": ..., "n_steps": ..., "use_region_mask": ...}
        "notes": [],
    }

    # Primary: extract from propose_edit_plan if it was called
    proposal = stored_masks.get("_plan_proposal")
    if proposal and isinstance(proposal, dict):
        for key in ("ridge_boost", "contrast_boost", "texture_amount", "smoothing_sigma"):
            raw = proposal.get(key, 0.0)
            lo, hi = CLAMP[key]
            plan[key] = min(max(float(raw), lo), hi)
        plan["target_regions"] = proposal.get("target_regions", "global")
        plan["notes"].append(f"Plan from propose_edit_plan: {proposal.get('summary', '')}")
        return plan

    # Fallback: scan tool calls for parameter values
    for tc in tool_calls:
        name = tc.get("name", "")
        inp = tc.get("input", {})

        if name in _DANGEROUS_TOOLS:
            plan["notes"].append(f"WARNING: dangerous tool '{name}' was requested but rejected")
            continue

        if name == "enhance_ridges":
            raw = inp.get("strength", 0.0)
            plan["ridge_boost"] = min(max(float(raw) / 10.0, 0.0), 0.35)
            plan["notes"].append(f"ridge_boost={plan['ridge_boost']:.3f} from enhance_ridges(strength={raw})")

        elif name == "boost_contrast":
            raw = inp.get("gamma", 1.0)
            plan["contrast_boost"] = min(max((float(raw) - 1.0) / 8.0, 0.0), 0.25)
            plan["notes"].append(f"contrast_boost={plan['contrast_boost']:.3f} from boost_contrast(gamma={raw})")

        elif name in ("texture_overlay", "blend_pattern"):
            raw = inp.get("amplitude", 0.0)
            plan["texture_amount"] = min(max(float(raw) * 0.25, 0.0), 0.10)
            plan["notes"].append(f"texture_amount={plan['texture_amount']:.3f} from {name}(amplitude={raw})")

        elif name in ("bandpass_filter", "freq_preserve_lowpass"):
            cutoff = inp.get("cutoff", inp.get("high_cutoff", 0.5))
            plan["smoothing_sigma"] = min(max(float(cutoff) * 3.0, 0.0), 1.5)
            plan["notes"].append(f"smoothing_sigma={plan['smoothing_sigma']:.3f} from {name}")

        elif name == "generate_mask":
            plan["notes"].append(f"mask generated: shape={inp.get('shape', 'unknown')}")

        elif name == "directional_step_convert":
            plan["directional_step"] = {
                "angle_deg": float(inp.get("angle_deg", 0.0)),
                "n_steps": int(inp.get("n_steps", 8)),
                "use_region_mask": inp.get("use_region_mask", True),
            }
            plan["notes"].append(
                f"directional_step_convert: angle={inp.get('angle_deg', 0.0):.1f}deg, "
                f"n_steps={inp.get('n_steps', 8)}"
            )

    return plan
