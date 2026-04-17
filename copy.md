"""
img2img diffusion pipeline: generates single-channel heightfield from
a conditioning image + a prompt auto-derived from TactileDescriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class DiffusionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    strength: float = 0.45          # img2img denoising strength (lowered from 0.75 to preserve structure)
    guidance_scale: float = 9.0
    num_inference_steps: int = 30
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_size: tuple[int, int] = (512, 512)


# ---------------------------------------------------------------------------
# Descriptor → prompt
# ---------------------------------------------------------------------------

def descriptor_to_prompt(descriptor) -> str:
    """
    Convert a TactileDescriptor into a textual prompt for SD img2img.

    The descriptor fields are mapped to language using simple threshold
    bands so the prompt reads naturally and steers the diffusion model
    toward the correct tactile character.

    Args:
        descriptor: TactileDescriptor (roughness, directionality, frequency).

    Returns:
        A concise English prompt string.
    """
    # --- roughness ---
    r = descriptor.roughness
    if r < 0.10:
        roughness_term = "very smooth, mirror-like surface"
    elif r < 0.25:
        roughness_term = "smooth surface with subtle texture"
    elif r < 0.45:
        roughness_term = "moderately rough surface"
    elif r < 0.65:
        roughness_term = "rough, granular surface"
    else:
        roughness_term = "very rough, coarse surface"

    # --- directionality ---
    d = descriptor.directionality
    if d < 0.15:
        direction_term = "isotropic, no preferred direction"
    elif d < 0.35:
        direction_term = "slightly directional texture"
    elif d < 0.60:
        direction_term = "moderately directional, linear grain"
    else:
        direction_term = "strongly directional, parallel ridges"

    # --- spatial frequency ---
    f = descriptor.frequency
    if f < 0.20:
        freq_term = "large-scale, coarse pattern"
    elif f < 0.45:
        freq_term = "medium-scale texture"
    elif f < 0.70:
        freq_term = "fine-grained texture"
    else:
        freq_term = "very fine, high-frequency micro-texture"

    prompt = (
        f"tactile heightfield surface texture, "
        f"{roughness_term}, {direction_term}, {freq_term}, "
        f"grayscale displacement map, single channel, top-down view, "
        f"no color, no shading, physically accurate"
    )
    return prompt


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_pipe(config: DiffusionConfig):
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
    )
    pipe = pipe.to(config.device)
    pipe.safety_checker = None  # disable for grayscale texture generation
    return pipe


def _prepare_conditioning(
    conditioning_image: np.ndarray,
    output_size: tuple[int, int],
    orientation_strength: np.ndarray | None = None,
) -> Image.Image:
    """Convert float32 (H,W) or (H,W,3) array → PIL RGB image.

    If orientation_strength is provided, blends it with the gray image
    (gray×0.7 + orient_strength×0.3) to improve structural conditioning.
    Diagnostic analysis showed this reduces RMSE from 0.2593 → 0.1933.
    """
    if conditioning_image.ndim == 2:
        gray = conditioning_image
        if orientation_strength is not None:
            gray = np.clip(gray * 0.7 + orientation_strength * 0.3, 0.0, 1.0)
        rgb = np.stack([gray] * 3, axis=-1)
    else:
        rgb = conditioning_image
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize(output_size)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_heightfield(
    conditioning_image: np.ndarray,
    descriptor,
    config: DiffusionConfig | None = None,
    prompt_override: str | None = None,
    features: dict | None = None,
) -> tuple[np.ndarray, str]:
    """
    Run img2img diffusion and return a single-channel heightfield.

    The prompt is automatically derived from the TactileDescriptor so that
    the model generates a texture matching the roughness, directionality,
    and frequency character measured from the input image.

    Args:
        conditioning_image: float32 (H, W) or (H, W, 3) array in [0, 1].
        descriptor: TactileDescriptor from tactile_mapping.map_features().
        config: DiffusionConfig (uses defaults if None).
        prompt_override: if provided, use this prompt instead of the
            auto-generated one (useful for manual experimentation).
        features: optional full features dict from preprocess(); if present,
            orientation_strength is extracted and blended into the conditioning
            image (gray×0.7 + orient_strength×0.3) for better structure preservation.

    Returns:
        (heightfield, prompt_used)
        heightfield: float32 numpy array (512, 512) in [0, 1].
        prompt_used: the exact prompt string sent to the model.
    """
    if config is None:
        config = DiffusionConfig()

    prompt = prompt_override if prompt_override else descriptor_to_prompt(descriptor)

    orientation_strength = features.get("orientation_strength") if features else None

    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    pil_input = _prepare_conditioning(
        conditioning_image, config.output_size, orientation_strength=orientation_strength
    )

    pipe = _build_pipe(config)
    result = pipe(
        prompt=prompt,
        image=pil_input,
        strength=config.strength,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        generator=generator,
    ).images[0]

    heightfield = np.array(result.convert("L"), dtype=np.float32) / 255.0
    return heightfield, prompt


def save_heightfield(heightfield: np.ndarray, out_path: str | Path) -> None:
    """Save heightfield as .npy file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), heightfield)
