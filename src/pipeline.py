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

    Preserves the image's natural brightness structure (including panel-level
    gradients). Large-scale brightness equalization is handled later in
    normalize_for_terrace_quantize via CLAHE.

      1. Percentile soft-stretch (p2-p98) — full dynamic range.
      2. Mild denoise (sigma=1.5) — pixel noise only, all structure preserved.
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
    """Fallback: return base_hf unchanged.

    In creative mode, the agent directly transforms the heightmap via tools
    and returns the result in AgentResult.final_hf. This function only
    serves as a fallback when the agent fails to produce a result.
    """
    logger.info("apply_agent_plan: creative mode fallback, returning base_hf unchanged")
    return base_hf.copy()


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
    std_min: float = 0.05,
    min_unique_labels: int = 3,
    grad_ratio_max: float = 6.0,
) -> tuple[bool, str]:
    """Validate that the agent-modified heightfield is safe for terrace meshing.

    Simplified validation for creative mode — only checks hard CNC constraints.

    Returns:
        (accepted, reason) tuple.
    """
    mod_std = float(modified_hf.std())

    logger.info(
        "validate: mod_std=%.4f mod_range=%.4f",
        mod_std,
        float(modified_hf.max() - modified_hf.min()),
    )

    # Check 1: Not almost flat
    if mod_std < std_min:
        reason = f"REJECT: modified heightfield is nearly flat (std={mod_std:.4f} < {std_min})"
        logger.warning(reason)
        return False, reason

    # Check 2: No label collapse (quantize to 8 levels, check unique count)
    labels = np.floor(np.clip(modified_hf, 0, 0.999) * 8).astype(np.int32)
    n_unique = len(np.unique(labels))
    if n_unique < min_unique_labels:
        reason = f"REJECT: label collapse (only {n_unique} unique labels < {min_unique_labels})"
        logger.warning(reason)
        return False, reason

    # Check 3: High-frequency noise not amplified too much (relative to base)
    base_gy, base_gx = np.gradient(base_hf)
    mod_gy, mod_gx = np.gradient(modified_hf)
    base_grad_mean = float(np.sqrt(base_gx**2 + base_gy**2).mean())
    mod_grad_mean = float(np.sqrt(mod_gx**2 + mod_gy**2).mean())

    if base_grad_mean > 1e-6:
        grad_ratio = mod_grad_mean / base_grad_mean
        if grad_ratio > grad_ratio_max:
            reason = (
                f"REJECT: high-frequency noise amplified too much "
                f"(gradient ratio={grad_ratio:.2f}x > {grad_ratio_max})"
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

    Uses CLAHE (tileGridSize=2x2) to equalize left/right brightness
    imbalance before global percentile normalization. This eliminates
    systematic height bias (e.g. left mean 0.567 vs right mean 0.461)
    that would otherwise compress one side into low terrace levels.

    Args:
        hf: Heightfield, expected roughly in [0, 1].
        p_low: Lower percentile mapped near 0.
        p_high: Upper percentile mapped near 1.
        preserve_margin: Keep a small margin to avoid exact hard clipping at 0/1.

    Returns:
        np.ndarray in [0, 1].
    """
    hf = hf.astype(np.float32)

    # CLAHE: 4 large tiles (2x2) equalize regional brightness without visible grid artifacts
    hf_u8 = (np.clip(hf, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
    hf_eq = clahe.apply(hf_u8).astype(np.float32) / 255.0

    # Global percentile normalization
    lo, hi = np.percentile(hf_eq, [p_low, p_high])
    dynamic = hi - lo
    if dynamic < 1e-6:
        return np.full_like(hf, 0.5, dtype=np.float32)
    out = (hf_eq - lo) / dynamic
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

    # 3. Agent planning (creative mode — agent directly executes tools)
    agent_config = AgentConfig(
        model=config.agent_model,
        max_iterations=config.agent_max_iterations,
    )
    # Prepare image_gray for image_guided_ridge_restore tool
    gray = np.array(Image.open(config.image_path).convert("L"), dtype=np.float32) / 255.0
    image_gray = cv2.resize(gray, (base_hf.shape[1], base_hf.shape[0]), interpolation=cv2.INTER_AREA)

    # Compute machining constraints for agent
    constraints = {
        "tool_diameter_mm": config.tool_diameter_mm,
        "tool_radius_mm": config.tool_diameter_mm / 2.0,
        "physical_size_mm": config.physical_size_mm,
        "max_height_mm": config.max_height_mm,
        "terrace_steps": config.terrace_steps if config.terrace_steps > 0 else 5,
    }

    agent_result = run_agent_pipeline(
        base_hf, raw_hf, config.user_intent, agent_config,
        image_gray=image_gray,
        constraints=constraints,
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

    # Agent planning (creative mode)
    agent_result = run_agent_pipeline(base_hf, raw_hf, user_intent, constraints={})

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
