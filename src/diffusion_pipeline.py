"""
Heightfield generation pipeline: input image → single-channel heightfield.

Uses the locally trained conditional DDPM checkpoint at
``<project_root>/models/improved`` by default. No text prompt, no
TactileDescriptor — the trained model learns the diffuse → height mapping
end-to-end from paired data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


_DEFAULT_TRAINED_CKPT = Path(__file__).resolve().parents[1] / "models" / "improved"


@dataclass
class DiffusionConfig:
    trained_model_path: str = str(_DEFAULT_TRAINED_CKPT)
    num_inference_steps: int = 50
    seed: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_size: tuple[int, int] = (512, 512)


def _ensure_checkpoint(path: str) -> Path:
    p = Path(path)
    if not (p.is_dir() and (p / "config.json").exists()):
        raise FileNotFoundError(
            f"Trained checkpoint not found at: {p}\n"
            f"Train the model first with: python src/training/train.py"
        )
    return p


_TRAINED_CACHE: dict = {}


def _load_model(config: DiffusionConfig):
    from training.infer import TrainedHeightfieldModel

    ckpt = str(_ensure_checkpoint(config.trained_model_path))
    key = (ckpt, config.device)
    model = _TRAINED_CACHE.get(key)
    if model is None:
        model = TrainedHeightfieldModel(ckpt, device=config.device)
        _TRAINED_CACHE[key] = model
    return model


def generate_heightfield(
    image: np.ndarray,
    config: DiffusionConfig | None = None,
) -> np.ndarray:
    """
    Generate a heightfield from an input image using the trained DDPM.

    Args:
        image: (H, W, 3) RGB or (H, W) grayscale array, uint8 or float32 in [0, 1].
        config: DiffusionConfig; uses defaults if None.

    Returns:
        float32 numpy array of shape ``config.output_size`` in [0, 1].
    """
    if config is None:
        config = DiffusionConfig()

    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    else:
        rgb = image

    model = _load_model(config)
    return model.generate(
        rgb,
        num_steps=config.num_inference_steps,
        seed=config.seed,
        output_size=config.output_size,
    )


def save_heightfield(heightfield: np.ndarray, out_path: str | Path) -> None:
    """Save heightfield as .npy file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), heightfield)
