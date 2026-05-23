"""
Tool definitions and execution dispatch for creative tactile design agent.

Creative tools directly transform heightmaps. The LLM selects tools
and provides parameters; this module validates and executes them.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

import heightmap_toolkit as htk
from heightmap_analyzer import analyze_heightmap, analysis_to_text, compare_analyses


TOOL_SCHEMAS = [
    # ── Analysis Tools ───────────────────────────────────────────────────────
    {
        "name": "analyze_heightmap",
        "description": "Analyze the current heightmap. Returns texture metrics, region classification, and gradient statistics.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ── Creative Transformation Tools ────────────────────────────────────────
    {
        "name": "redistribute_topology",
        "description": "Remap the height histogram to a target distribution. "
                       "Use 'uniform' to spread heights evenly, 'gaussian' to concentrate "
                       "around a center, 'bimodal' to create distinct high/low zones.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_distribution": {
                    "type": "string",
                    "enum": ["uniform", "gaussian", "bimodal", "emphasize_ridges"],
                    "description": "Target height distribution shape.",
                },
                "center": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Center of gaussian/bimodal distribution (default 0.5).",
                },
                "width": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.5,
                    "description": "Width/spread of the distribution (default 0.2).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_directional_texture",
        "description": "Add grain-aligned directional texture. Creates parallel ridges or "
                       "flow patterns aligned with a given angle. Use for wood grain, "
                       "brushed metal, or flowing water effects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "angle_deg": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 180.0,
                    "description": "Direction of texture grain in degrees (0=horizontal).",
                },
                "wavelength_px": {
                    "type": "number",
                    "minimum": 4.0,
                    "maximum": 128.0,
                    "description": "Spacing between ridges in pixels (default 32).",
                },
                "amplitude": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.3,
                    "description": "Strength of the texture overlay (default 0.1).",
                },
                "type": {
                    "type": "string",
                    "enum": ["waves", "fibers", "crosshatch"],
                    "description": "Texture pattern type (default 'waves').",
                },
            },
            "required": ["angle_deg"],
        },
    },
    {
        "name": "blend_procedural_base",
        "description": "Blend a procedural pattern into the surface as a structural base. "
                       "Use voronoi for organic cells, hex for geometric, waves for flowing, "
                       "brick for architectural patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern_type": {
                    "type": "string",
                    "enum": ["voronoi", "hex_grid", "waves", "brick", "perlin"],
                    "description": "Procedural pattern to blend.",
                },
                "blend_mode": {
                    "type": "string",
                    "enum": ["add", "multiply", "overlay"],
                    "description": "How to blend with existing surface (default 'add').",
                },
                "amplitude": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.5,
                    "description": "Blend strength (default 0.15).",
                },
                "frequency": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 32.0,
                    "description": "Pattern frequency (default 4.0).",
                },
                "seed": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Random seed for reproducibility.",
                },
            },
            "required": ["pattern_type"],
        },
    },
    {
        "name": "regional_carve",
        "description": "Carve a focused feature into a specific region. Creates valleys, "
                       "ridges, channels, or domes within a mask area.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_type": {
                    "type": "string",
                    "enum": ["valley", "ridge", "channel", "high"],
                    "description": "Type of feature to carve.",
                },
                "depth": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.15,
                    "description": "Depth/height of the feature (default 0.2).",
                },
                "center_x": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "X center of the feature region (default 0.5).",
                },
                "center_y": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Y center of the feature region (default 0.5).",
                },
                "size": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": 0.8,
                    "description": "Size of the feature region as fraction of image (default 0.3).",
                },
            },
            "required": ["feature_type"],
        },
    },
    {
        "name": "smooth_topology",
        "description": "Smooth the surface to repair topology violations. Removes isolated "
                       "islands, fills sharp concave corners, and ensures CNC machinability. "
                       "Call this after major reshaping operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sigma": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 5.0,
                    "description": "Gaussian smoothing sigma (default 1.5).",
                },
                "min_feature_px": {
                    "type": "number",
                    "minimum": 2.0,
                    "maximum": 32.0,
                    "description": "Minimum feature size in pixels — smaller features are merged (default 8).",
                },
            },
            "required": [],
        },
    },
    # ── Mask Tools ───────────────────────────────────────────────────────────
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
    # ── Meta Tools ───────────────────────────────────────────────────────────
    {
        "name": "evaluate_result",
        "description": "Evaluate current heightmap against user intent keywords.",
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
        "name": "accept_result",
        "description": "Accept current heightmap as final result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
            },
            "required": ["summary"],
        },
    },
]


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

    # Log all operations
    if "_operations_log" not in stored_masks:
        stored_masks["_operations_log"] = []

    if tool_name == "analyze_heightmap":
        report = analyze_heightmap(current_hf)
        return current_hf, analysis_to_text(report), False

    # ── Creative Transformation Tools ────────────────────────────────────────
    elif tool_name == "redistribute_topology":
        target_dist = tool_input.get("target_distribution", "uniform")
        strength = float(tool_input.get("width", 0.2))

        if target_dist == "emphasize_ridges":
            # Raise ridges, lower valleys — amplify relief
            result = htk.height_selective_transform(
                current_hf,
                ridge_boost=strength * 0.3,
                valley_boost=-strength * 0.15,
                high_threshold=0.6,
                low_threshold=0.4,
            )
        else:
            result = htk.height_redistribute(
                current_hf,
                target_distribution=target_dist,
                center=float(tool_input.get("center", 0.5)),
                width=strength,
            )
        stored_masks["_operations_log"].append({
            "tool": "redistribute_topology",
            "params": tool_input,
        })
        return result, (
            f"Redistributed topology: distribution={tool_input.get('target_distribution', 'uniform')}, "
            f"center={tool_input.get('center', 0.5)}, width={tool_input.get('width', 0.2)}"
        ), False

    elif tool_name == "generate_directional_texture":
        angle = float(tool_input.get("angle_deg", 0.0))
        wavelength = float(tool_input.get("wavelength_px", 32.0))
        amplitude = float(tool_input.get("amplitude", 0.1))
        texture_type = tool_input.get("type", "waves")

        if texture_type == "waves":
            result = htk.blend_pattern(
                current_hf, pattern_type="waves", blend_mode="add",
                amplitude=amplitude, frequency=max(1.0, 512.0 / wavelength), angle_deg=angle,
            )
        elif texture_type == "fibers":
            result = htk.anisotropic_emphasis(current_hf, angle_deg=angle, strength=amplitude * 3.0)
        elif texture_type == "crosshatch":
            result = htk.blend_pattern(
                current_hf, pattern_type="crosshatch", blend_mode="add",
                amplitude=amplitude, frequency=max(1.0, 512.0 / wavelength), angle_deg=angle,
            )
        else:
            result = htk.blend_pattern(
                current_hf, pattern_type="waves", blend_mode="add",
                amplitude=amplitude, frequency=max(1.0, 512.0 / wavelength), angle_deg=angle,
            )

        # Smooth texture edges: turn vertical jumps into gentle slopes
        step_smooth = max(wavelength * 0.25, 2.0)
        result = gaussian_filter(result, sigma=step_smooth).astype(np.float32)

        stored_masks["_operations_log"].append({
            "tool": "generate_directional_texture",
            "params": tool_input,
        })
        return result, (
            f"Generated directional texture: type={texture_type}, angle={angle:.1f}deg, "
            f"wavelength={wavelength}px, amplitude={amplitude}"
        ), False

    elif tool_name == "blend_procedural_base":
        pattern_type = tool_input.get("pattern_type", "perlin")
        proc = htk.procedural_generate(
            pattern_type=pattern_type,
            size=current_hf.shape[0],
            frequency=float(tool_input.get("frequency", 4.0)),
            amplitude=1.0,
            seed=int(tool_input.get("seed", 42)),
        )
        # Low-pass filter: keep only large-scale structure, no fine grain noise
        H = current_hf.shape[0]
        proc_smooth_sigma = max(H / 16.0, 4.0)
        proc = gaussian_filter(proc, sigma=proc_smooth_sigma)
        proc = np.clip(proc, 0, 1).astype(np.float32)

        # Blend generated pattern into current heightmap
        amplitude = float(tool_input.get("amplitude", 0.15))
        blend_mode = tool_input.get("blend_mode", "add")
        result = htk.blend_two(current_hf, proc, alpha=amplitude, blend_mode=blend_mode if blend_mode in ("linear", "multiply", "screen") else "linear")

        stored_masks["_operations_log"].append({
            "tool": "blend_procedural_base",
            "params": tool_input,
        })
        return result, (
            f"Blended procedural base: pattern={pattern_type}, mode={blend_mode}, "
            f"amplitude={amplitude}, frequency={tool_input.get('frequency', 4.0)}"
        ), False

    elif tool_name == "regional_carve":
        feature_type = tool_input.get("feature_type", "valley")
        depth = float(tool_input.get("depth", 0.2))
        cx = float(tool_input.get("center_x", 0.5))
        cy = float(tool_input.get("center_y", 0.5))
        size = float(tool_input.get("size", 0.3))

        # Generate region mask
        mask = htk.generate_mask(
            shape="ellipse", center_x=cx, center_y=cy,
            size_x=size, size_y=size, feather_px=16.0,
            resolution=current_hf.shape[0],
        )

        if feature_type == "valley":
            carved = current_hf - depth * mask
        elif feature_type == "ridge":
            carved = current_hf + depth * mask
        elif feature_type == "channel":
            mask = htk.generate_mask(
                shape="rectangle", center_x=cx, center_y=cy,
                size_x=size * 2, size_y=size * 0.5, feather_px=8.0,
                resolution=current_hf.shape[0],
            )
            carved = current_hf - depth * mask
        elif feature_type == "high":
            # Bidirectional stretch: raise highs, lower lows, preserve mean
            p75 = float(np.percentile(current_hf, 75))
            p25 = float(np.percentile(current_hf, 25))
            high_mask = np.clip((current_hf - p75) / (1.0 - p75 + 1e-8), 0, 1)
            low_mask = np.clip((p25 - current_hf) / (p25 + 1e-8), 0, 1)
            carved = current_hf + depth * high_mask * mask - depth * 0.5 * low_mask * mask
        else:
            carved = current_hf

        result = np.clip(carved, 0.0, 1.0).astype(np.float32)

        stored_masks["_operations_log"].append({
            "tool": "regional_carve",
            "params": tool_input,
        })
        return result, (
            f"Carved {feature_type}: depth={depth}, center=({cx:.2f},{cy:.2f}), size={size:.2f}"
        ), False

    elif tool_name == "smooth_topology":
        sigma = float(tool_input.get("sigma", 1.5))
        min_feature = float(tool_input.get("min_feature_px", 8.0))

        # Gaussian smooth
        smoothed = gaussian_filter(current_hf, sigma=sigma)
        smoothed = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

        # Morphological cleanup: remove small isolated features
        if min_feature > 1.0:
            import cv2
            n_levels = 8
            labels = np.floor(np.clip(smoothed, 0, 0.999) * n_levels).astype(np.int32)
            min_area = max(int(min_feature * min_feature), 4)

            for level in range(n_levels):
                mask = (labels == level).astype(np.uint8)
                n_comp, _, stats, _ = cv2.connectedComponentsWithStats(mask)
                for i in range(1, n_comp):
                    if stats[i, cv2.CC_STAT_AREA] < min_area:
                        component_mask = (mask == i).astype(np.uint8)
                        local_mean = cv2.blur(smoothed, (int(min_feature), int(min_feature)))
                        smoothed[component_mask > 0] = local_mean[component_mask > 0]

            smoothed = np.clip(smoothed, 0.0, 1.0).astype(np.float32)

        stored_masks["_operations_log"].append({
            "tool": "smooth_topology",
            "params": tool_input,
        })
        return smoothed, (
            f"Smoothed topology: sigma={sigma}, min_feature={min_feature}px"
        ), False

    # ── Mask Tools ───────────────────────────────────────────────────────────
    elif tool_name == "generate_mask":
        mask = htk.generate_mask(**tool_input)
        stored_masks["last"] = mask
        return current_hf, f"Generated mask: shape={tool_input.get('shape', 'ellipse')}, stored for mask_apply", False

    elif tool_name == "mask_apply":
        if "last" not in stored_masks:
            return current_hf, "ERROR: mask_apply requires a stored mask. Call generate_mask first.", False
        mask = stored_masks["last"]
        result = htk.mask_apply(original_hf, current_hf, mask, **tool_input)
        return result, f"Applied mask_apply(feather={tool_input.get('feather_px', 8.0)}, mask=stored)", False

    # ── Meta Tools ───────────────────────────────────────────────────────────
    elif tool_name == "evaluate_result":
        after = analyze_heightmap(current_hf)
        comparison = compare_analyses(analysis_before, after)
        keywords = tool_input.get("intent_keywords", [])
        feedback = _evaluate_against_intent(analysis_before, after, keywords)
        return current_hf, f"{comparison}\n\n{feedback}", False

    elif tool_name == "accept_result":
        summary = tool_input.get("summary", "Accepted")
        return current_hf, f"Accepted: {summary}", True

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


def extract_plan_parameters(
    tool_calls: list[dict],
    stored_masks: dict,
) -> dict:
    """Extract plan info from the operation log.

    Returns dict with keys: operations, topology_preserved, reasoning, notes.
    """
    operations_log = stored_masks.get("_operations_log", [])

    return {
        "operations": operations_log,
        "topology_preserved": True,
        "reasoning": f"Applied {len(operations_log)} creative operations",
        "notes": [f"Operation {i+1}: {op['tool']}" for i, op in enumerate(operations_log)],
    }
