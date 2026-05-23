"""
Unified pipeline: Image -> Raw Heightmap -> Preprocess -> Agent Plan -> Validate -> Machinable STL.

Bounded Agent Transform architecture:
  raw_hf -> preprocess_for_terrace() -> base_hf
         -> agent plans bounded parameters -> AgentEditPlan
         -> apply_agent_plan(base_hf, plan) -> modified_hf
         -> validate against base_hf -> accept or fallback
         -> heightfield_to_terrace_mesh() -> STL
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from diffusion_pipeline import generate_heightfield, DiffusionConfig
from agent_planner import run_agent_pipeline, AgentConfig, AgentResult, AgentEditPlan
from heightmap_analyzer import analyze_heightmap, analysis_to_json
from terrace_geometry import (
    preprocess_for_terrace,
    heightfield_to_terrace_mesh,
    TerraceConfig,
    save_stl,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1]."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.full_like(x, 0.5, dtype=np.float32)


def image_guided_blend(
    image_path: str,
    raw_hf: np.ndarray,
    image_weight: float = 0.8,
) -> np.ndarray:
    """Blend image gray-level into diffusion output to recover dynamic range.

    Simple approach that preserves the image's natural brightness structure:
      1. Percentile soft-stretch (p2-p98) — full dynamic range, no block boundaries.
      2. Mild denoise (sigma=1.5) — pixel noise only, all wood grain preserved.
      3. Blend with diffusion output.

    Args:
        image_path: Path to the source image.
        raw_hf: Diffusion-generated heightfield.
        image_weight: Weight for image gray component (default 0.8).

    Returns:
        Blended heightfield, float32 [0, 1].
    """
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    gray = cv2.resize(gray, (raw_hf.shape[1], raw_hf.shape[0]), interpolation=cv2.INTER_AREA)

    # 1. Percentile soft normalization (no block boundaries, no halo)
    lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
    gray_norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # 2. Mild denoise (only pixel noise, all wood grain structure preserved)
    gray_clean = gaussian_filter(gray_norm, sigma=1.5)

    # 3. Blend with diffusion output
    blended = image_weight * gray_clean + (1.0 - image_weight) * _normalize(raw_hf)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


@dataclass
class PipelineConfig:
    image_path: str
    user_intent: str
    output_dir: str = "outputs/agent_run"
    diffusion_steps: int = 50
    diffusion_seed: int = 20
    agent_model: str = "claude-sonnet-4-20250514"
    agent_max_iterations: int = 3
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_diameter_mm: float = 6.0
    terrace_steps: int = 0


@dataclass
class PipelineResult:
    raw_heightmap: np.ndarray
    base_heightmap: np.ndarray
    agent_plan: AgentEditPlan
    final_heightmap: np.ndarray
    stl_path: Path
    agent_result: AgentResult
    validation_accepted: bool
    validation_reason: str


def apply_agent_plan(
    base_hf: np.ndarray,
    raw_hf: np.ndarray,
    plan: AgentEditPlan,
    alpha: float = 0.35,
) -> np.ndarray:
    """Apply bounded edit plan to the base heightfield.

    Clamps all plan values to safe ranges, builds a bounded candidate,
    and alpha-blends against the stable base.

    Args:
        base_hf: Preprocessed stable heightfield (source of truth).
        raw_hf: Original raw heightfield (for reference only).
        plan: AgentEditPlan with bounded parameters.
        alpha: Maximum blend factor. Default 0.35.

    Returns:
        Modified heightfield as float32 [0, 1].
    """
    import heightmap_toolkit as htk

    alpha = min(max(alpha, 0.0), 0.35)

    # Clamp plan values
    ridge_boost = min(max(plan.ridge_boost, 0.0), 0.35)
    contrast_boost = min(max(plan.contrast_boost, 0.0), 0.25)
    texture_amount = min(max(plan.texture_amount, 0.0), 0.10)
    smoothing_sigma = min(max(plan.smoothing_sigma, 0.0), 1.5)

    logger.info(
        "apply_agent_plan: ridge=%.3f contrast=%.3f texture=%.3f smooth=%.3f alpha=%.2f target=%s",
        ridge_boost, contrast_boost, texture_amount, smoothing_sigma, alpha, plan.target_regions,
    )

    # Build candidate by applying bounded modifications to base_hf
    candidate = base_hf.copy()

    # Generate target region mask
    region_mask = _build_region_mask(base_hf, plan.target_regions)

    # 1. Ridge enhancement (scale strength to toolkit range)
    if ridge_boost > 0.001:
        toolkit_strength = ridge_boost * 10.0  # map [0,0.35] -> [0,3.5]
        candidate = htk.enhance_ridges(candidate, strength=toolkit_strength)

    # 2. Contrast boost (scale to gamma range)
    if contrast_boost > 0.001:
        gamma = 1.0 + contrast_boost * 8.0  # map [0,0.25] -> [1.0,3.0]
        candidate = htk.boost_contrast(candidate, gamma=gamma)

    # 3. Texture overlay
    if texture_amount > 0.001:
        toolkit_amp = texture_amount * 4.0  # map [0,0.10] -> [0,0.40]
        candidate = htk.texture_overlay(candidate, amplitude=toolkit_amp)

    # 4. Smoothing (mild gaussian)
    if smoothing_sigma > 0.01:
        from scipy.ndimage import gaussian_filter
        candidate = gaussian_filter(candidate, sigma=smoothing_sigma)
        candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)

    # 4b. Directional step conversion for Type B regions (if agent requested it)
    if plan.directional_step_angle_deg is not None:
        import heightmap_toolkit as htk
        from heightmap_analyzer import analyze_heightmap as _analyze
        # Re-analyze to get the region mask on the current candidate
        cand_analysis = _analyze(candidate)
        ds_mask = cand_analysis.region_b_mask if plan.directional_step_use_region_mask else None
        candidate = htk.directional_step_convert(
            candidate,
            angle_deg=plan.directional_step_angle_deg,
            n_steps=plan.directional_step_n_steps or 8,
            mask=ds_mask,
            feather_px=12.0,
        )
        logger.info(
            "apply_agent_plan: directional_step_convert applied (angle=%.1f, n_steps=%s, mask=%s)",
            plan.directional_step_angle_deg,
            plan.directional_step_n_steps or 8,
            "region_b" if ds_mask is not None else "none",
        )

    # 5. Apply region mask: modifications only where mask is active
    if region_mask is not None:
        candidate = base_hf * (1.0 - region_mask) + candidate * region_mask

    # 6. Alpha blend against base
    final = (1.0 - alpha) * base_hf + alpha * candidate

    # 7. Clip to [0, 1]
    final = np.clip(final, 0.0, 1.0).astype(np.float32)

    logger.info(
        "apply_agent_plan result: min=%.4f max=%.4f std=%.4f mean=%.4f",
        float(final.min()), float(final.max()), float(final.std()), float(final.mean()),
    )

    return final


def _build_region_mask(hf: np.ndarray, target_regions: str) -> np.ndarray | None:
    """Build a binary mask based on target region specification."""
    if target_regions == "global":
        return None  # No mask — apply everywhere

    if target_regions == "ridges":
        # Upper quartile of heights
        threshold = np.percentile(hf, 75)
        mask = (hf >= threshold).astype(np.float32)
    elif target_regions == "valleys":
        # Lower quartile of heights
        threshold = np.percentile(hf, 25)
        mask = (hf <= threshold).astype(np.float32)
    elif target_regions == "high":
        # Upper half
        threshold = np.percentile(hf, 50)
        mask = (hf >= threshold).astype(np.float32)
    elif target_regions == "low":
        # Lower half
        threshold = np.percentile(hf, 50)
        mask = (hf <= threshold).astype(np.float32)
    else:
        return None

    # Feather the mask edges
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=3.0)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def validate_agent_modified_heightfield(
    base_hf: np.ndarray,
    modified_hf: np.ndarray,
    std_min: float = 0.04,
    std_change_max: float = 1.5,
    min_unique_labels: int = 3,
) -> tuple[bool, str]:
    """Validate that the agent-modified heightfield is safe for terrace meshing.

    All checks are relative to base_hf.

    Returns:
        (accepted, reason) tuple.
    """
    base_std = float(base_hf.std())
    mod_std = float(modified_hf.std())

    logger.info(
        "validate: base_std=%.4f mod_std=%.4f base_range=%.4f mod_range=%.4f",
        base_std, mod_std,
        float(base_hf.max() - base_hf.min()),
        float(modified_hf.max() - modified_hf.min()),
    )

    # Check 1: Not almost flat
    if mod_std < std_min:
        reason = f"REJECT: modified heightfield is nearly flat (std={mod_std:.4f} < {std_min})"
        logger.warning(reason)
        return False, reason

    # Check 2: Std change not too drastic (relative to base)
    if base_std > 1e-6:
        std_ratio = abs(mod_std - base_std) / base_std
        if std_ratio > std_change_max:
            reason = (
                f"REJECT: std change too drastic "
                f"({base_std:.4f} -> {mod_std:.4f}, ratio={std_ratio:.2f} > {std_change_max})"
            )
            logger.warning(reason)
            return False, reason

    # Check 3: No label collapse (quantize to 8 levels, check unique count)
    labels = np.floor(np.clip(modified_hf, 0, 0.999) * 8).astype(np.int32)
    n_unique = len(np.unique(labels))
    if n_unique < min_unique_labels:
        reason = f"REJECT: label collapse (only {n_unique} unique labels < {min_unique_labels})"
        logger.warning(reason)
        return False, reason

    # Check 4: No excessive tiny components (relative to base)
    import cv2
    base_labels = np.floor(np.clip(base_hf, 0, 0.999) * 8).astype(np.int32)
    for level in range(1, 8):
        mod_mask = (labels == level).astype(np.uint8)
        base_mask = (base_labels == level).astype(np.uint8)
        if mod_mask.sum() == 0:
            continue
        total_pixels = mod_mask.size
        tiny_threshold = total_pixels * 0.001

        # Count tiny components in modified
        n_comp_mod, _, stats_mod, _ = cv2.connectedComponentsWithStats(mod_mask)
        tiny_mod = sum(1 for i in range(1, n_comp_mod) if stats_mod[i, cv2.CC_STAT_AREA] < tiny_threshold)

        # Count tiny components in base
        n_comp_base, _, stats_base, _ = cv2.connectedComponentsWithStats(base_mask)
        tiny_base = sum(1 for i in range(1, n_comp_base) if stats_base[i, cv2.CC_STAT_AREA] < tiny_threshold)

        # Reject if tiny components increased by more than 50% relative to base
        if tiny_mod > max(tiny_base * 1.5, tiny_base + 10):
            reason = (
                f"REJECT: excessive tiny components at level {level} "
                f"({tiny_mod} vs base {tiny_base})"
            )
            logger.warning(reason)
            return False, reason

    # Check 5: High-frequency noise not amplified too much (relative to base)
    base_gy, base_gx = np.gradient(base_hf)
    mod_gy, mod_gx = np.gradient(modified_hf)
    base_grad_mean = float(np.sqrt(base_gx**2 + base_gy**2).mean())
    mod_grad_mean = float(np.sqrt(mod_gx**2 + mod_gy**2).mean())

    if base_grad_mean > 1e-6:
        grad_ratio = mod_grad_mean / base_grad_mean
        if grad_ratio > 5.0:
            reason = (
                f"REJECT: high-frequency noise amplified too much "
                f"(gradient ratio={grad_ratio:.2f}x)"
            )
            logger.warning(reason)
            return False, reason

    logger.info("validate: ACCEPTED")
    return True, "accepted"


def normalize_for_terrace_quantize(
    hf: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    preserve_margin: float = 0.02,
) -> np.ndarray:
    """Expand a validated heightfield before terrace quantization.

    NOT an agent editing operation. Remaps the already validated final
    heightfield so terrace quantization uses more available levels.

    Args:
        hf: Heightfield, expected roughly in [0, 1].
        p_low: Lower percentile mapped near 0.
        p_high: Upper percentile mapped near 1.
        preserve_margin: Keep a small margin to avoid exact hard clipping at 0/1.

    Returns:
        np.ndarray in [0, 1].
    """
    hf = hf.astype(np.float32)
    lo, hi = np.percentile(hf, [p_low, p_high])
    dynamic = hi - lo
    if dynamic < 1e-6:
        return np.full_like(hf, 0.5, dtype=np.float32)
    out = (hf - lo) / dynamic
    out = np.clip(out, 0.0, 1.0)
    if preserve_margin > 0:
        out = preserve_margin + (1.0 - 2.0 * preserve_margin) * out
    return out.astype(np.float32)


def _debug_heightfield(name: str, hf: np.ndarray) -> None:
    print(f"[{name}] min={hf.min():.6f}, max={hf.max():.6f}, mean={hf.mean():.6f}, std={hf.std():.6f}")
    print(f"[{name}] p1,p5,p50,p95,p99={np.percentile(hf, [1, 5, 50, 95, 99]).tolist()}")


def _debug_labels(name: str, labels: np.ndarray) -> None:
    u, c = np.unique(labels, return_counts=True)
    print(f"[{name}] unique={u.tolist()}")
    print(f"[{name}] counts={dict(zip(u.tolist(), c.tolist()))}")


def run_full_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the complete pipeline: preprocess -> agent plan -> apply -> validate -> STL."""
    out_dir = ROOT / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate raw heightfield
    img = np.array(Image.open(config.image_path).convert("RGB"))
    diff_config = DiffusionConfig(
        num_inference_steps=config.diffusion_steps,
        seed=config.diffusion_seed,
    )
    raw_hf = generate_heightfield(img, diff_config)
    np.save(out_dir / "heightfield_raw.npy", raw_hf)

    # 1b. Image-guided blend: bypass diffusion's weak dynamic range
    blended_hf = image_guided_blend(config.image_path, raw_hf)
    np.save(out_dir / "heightfield_blended.npy", blended_hf)

    # 2. Preprocess FIRST (produces stable base)
    base_hf = preprocess_for_terrace(
        blended_hf,
        tool_diameter_mm=config.tool_diameter_mm,
        physical_size_mm=config.physical_size_mm,
    )
    np.save(out_dir / "heightfield_base.npy", base_hf)

    # 3. Agent planning (agent directly executes tools, returns final_hf)
    agent_config = AgentConfig(
        model=config.agent_model,
        max_iterations=config.agent_max_iterations,
    )
    # Prepare image_gray for image_guided_ridge_restore tool
    gray = np.array(Image.open(config.image_path).convert("L"), dtype=np.float32) / 255.0
    image_gray = cv2.resize(gray, (base_hf.shape[1], base_hf.shape[0]), interpolation=cv2.INTER_AREA)

    agent_result = run_agent_pipeline(
        base_hf, raw_hf, config.user_intent, agent_config,
        image_gray=image_gray,
    )

    # 4. Use agent's final heightmap if available, otherwise apply plan
    if agent_result.final_hf is not None:
        modified_hf = agent_result.final_hf
    else:
        modified_hf = apply_agent_plan(base_hf, raw_hf, agent_result.plan)
    np.save(out_dir / "heightfield_modified.npy", modified_hf)

    # 5. Validate
    accepted, reason = validate_agent_modified_heightfield(base_hf, modified_hf)
    final_hf = modified_hf if accepted else base_hf
    if not accepted:
        logger.warning("Agent modification rejected, falling back to base_hf: %s", reason)
    np.save(out_dir / "heightfield_final.npy", final_hf)

    # 6. Normalize for terrace quantization (expands range, not an agent edit)
    tc = TerraceConfig(
        physical_size_mm=config.physical_size_mm,
        max_height_mm=config.max_height_mm,
        tool_diameter_mm=config.tool_diameter_mm,
        terrace_steps=config.terrace_steps if config.terrace_steps > 0 else 5,
    )
    terrace_hf = normalize_for_terrace_quantize(final_hf, p_low=1.0, p_high=99.0, preserve_margin=0.02)
    np.save(out_dir / "heightfield_terrace_input.npy", terrace_hf)
    _debug_heightfield("terrace_hf", terrace_hf)
    _debug_labels("terrace_hf quantize", np.floor(np.clip(terrace_hf, 0, 0.999) * tc.terrace_steps).astype(np.int32))

    # 7. Generate terrace mesh
    mesh, report = heightfield_to_terrace_mesh(terrace_hf, tc)
    stl_path = out_dir / "terrace.stl"
    save_stl(mesh, stl_path)

    # 7. Save analysis
    agent_result.analysis_after = analyze_heightmap(final_hf)
    (out_dir / "analysis_before.json").write_text(
        analysis_to_json(agent_result.analysis_before)
    )
    (out_dir / "analysis_after.json").write_text(
        analysis_to_json(agent_result.analysis_after)
    )

    return PipelineResult(
        raw_heightmap=raw_hf,
        base_heightmap=base_hf,
        agent_plan=agent_result.plan,
        final_heightmap=final_hf,
        stl_path=stl_path,
        agent_result=agent_result,
        validation_accepted=accepted,
        validation_reason=reason,
    )


def run_agent_only(
    heightmap_path: str,
    user_intent: str,
    output_dir: str = "outputs/agent_run",
) -> PipelineResult:
    """Run only the agent transformation on an existing heightmap."""
    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_hf = np.load(heightmap_path).astype(np.float32)

    # Preprocess first
    base_hf = preprocess_for_terrace(raw_hf)
    np.save(out_dir / "heightfield_base.npy", base_hf)

    # Agent planning
    agent_result = run_agent_pipeline(base_hf, raw_hf, user_intent)

    # Use agent's final heightmap if available, otherwise apply plan
    if agent_result.final_hf is not None:
        modified_hf = agent_result.final_hf
    else:
        modified_hf = apply_agent_plan(base_hf, raw_hf, agent_result.plan)
    np.save(out_dir / "heightfield_modified.npy", modified_hf)

    # Validate
    accepted, reason = validate_agent_modified_heightfield(base_hf, modified_hf)
    final_hf = modified_hf if accepted else base_hf
    np.save(out_dir / "heightfield_final.npy", final_hf)

    # Normalize for terrace quantization
    tc = TerraceConfig()
    terrace_hf = normalize_for_terrace_quantize(final_hf, p_low=1.0, p_high=99.0, preserve_margin=0.02)
    np.save(out_dir / "heightfield_terrace_input.npy", terrace_hf)
    _debug_heightfield("terrace_hf", terrace_hf)
    _debug_labels("terrace_hf quantize", np.floor(np.clip(terrace_hf, 0, 0.999) * tc.terrace_steps).astype(np.int32))

    # Generate mesh
    mesh, report = heightfield_to_terrace_mesh(terrace_hf, tc)
    stl_path = out_dir / "terrace.stl"
    save_stl(mesh, stl_path)

    agent_result.analysis_after = analyze_heightmap(final_hf)

    return PipelineResult(
        raw_heightmap=raw_hf,
        base_heightmap=base_hf,
        agent_plan=agent_result.plan,
        final_heightmap=final_hf,
        stl_path=stl_path,
        agent_result=agent_result,
        validation_accepted=accepted,
        validation_reason=reason,
    )
