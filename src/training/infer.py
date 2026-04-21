"""
Inference helper for the locally trained conditional DDPM.

Typical use inside the main pipeline:

    from training.infer import TrainedHeightfieldModel
    trained = TrainedHeightfieldModel(checkpoint_dir, device="cuda")
    heightfield = trained.generate(diffuse_rgb, num_steps=50, seed=42)

The model is fully convolutional so checkpoints trained at 256 work at any
multiple-of-8 resolution (512 used by the rest of the pipeline).
"""
from __future__ import annotations

from pathlib import Path

import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cnc_params import FEATURE_SIGMA, MAX_SLOPE_PX  # noqa: E402


def _enforce_machinability(h: np.ndarray) -> np.ndarray:
    h = gaussian_filter(h, sigma=FEATURE_SIGMA)
    slope_limit = MAX_SLOPE_PX * 0.85
    for _ in range(20):
        gx = np.gradient(h, axis=1)
        gy = np.gradient(h, axis=0)
        slope = np.sqrt(gx * gx + gy * gy)
        mask = slope > slope_limit
        if not mask.any():
            break
        blurred = gaussian_filter(h, sigma=1.5)
        h = np.where(mask, blurred, h)
    lo, hi = float(h.min()), float(h.max())
    if hi - lo > 1e-6:
        h = (h - lo) / (hi - lo)
    else:
        h = np.zeros_like(h)
    return h.astype(np.float32)


def _round_to_multiple(x: int, m: int = 8) -> int:
    return max(m, (x // m) * m)


class TrainedHeightfieldModel:
    def __init__(self, checkpoint_dir: str | Path, device: str | None = None) -> None:
        from diffusers import UNet2DModel, DDIMScheduler

        self.checkpoint_dir = Path(checkpoint_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = UNet2DModel.from_pretrained(self.checkpoint_dir).to(self.device)
        self.model.eval()
        self.scheduler = DDIMScheduler.from_pretrained(self.checkpoint_dir)
        self.scheduler.config.clip_sample = False

    @torch.no_grad()
    def generate(
        self,
        diffuse_rgb: np.ndarray,
        num_steps: int = 50,
        eta: float = 0.0,
        seed: int | None = None,
        output_size: tuple[int, int] | None = None,
        enforce_machinability: bool = False,
    ) -> np.ndarray:
        """
        diffuse_rgb: (H, W, 3) uint8 OR float32 in [0, 1].
        Returns (H, W) float32 heightfield in [0, 1].
        """
        if diffuse_rgb.ndim == 2:
            diffuse_rgb = np.stack([diffuse_rgb] * 3, axis=-1)
        if diffuse_rgb.dtype == np.uint8:
            d = diffuse_rgb.astype(np.float32) / 127.5 - 1.0
        else:
            d = diffuse_rgb.astype(np.float32)
            if d.max() <= 1.0 + 1e-3:
                d = d * 2.0 - 1.0

        H, W = d.shape[:2]
        if output_size is not None:
            out_h, out_w = output_size
        else:
            out_h, out_w = H, W
        H8 = _round_to_multiple(out_h, 8)
        W8 = _round_to_multiple(out_w, 8)
        if (H8, W8) != (H, W):
            d = cv2.resize(d, (W8, H8), interpolation=cv2.INTER_AREA)

        diffuse_t = torch.from_numpy(d).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
        else:
            generator = None
        height_t = torch.randn((1, 1, H8, W8), device=self.device, generator=generator)

        self.scheduler.set_timesteps(num_steps, device=self.device)
        for t in self.scheduler.timesteps:
            model_in = torch.cat([height_t, diffuse_t], dim=1)
            pred = self.model(model_in, t).sample
            step_kwargs = {}
            if "eta" in self.scheduler.step.__code__.co_varnames:
                step_kwargs["eta"] = eta
            height_t = self.scheduler.step(pred, t, height_t, **step_kwargs).prev_sample

        h = height_t.squeeze().cpu().numpy()
        h = (h + 1.0) / 2.0
        h = np.clip(h, 0.0, 1.0).astype(np.float32)
        if (H8, W8) != (out_h, out_w):
            h = cv2.resize(h, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if enforce_machinability:
            h = _enforce_machinability(h)
        return h
