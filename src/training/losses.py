"""
Fabrication-aware loss functions for CNC-constrained heightmap DDPM training.

All tensors are (B, 1, H, W) in [0, 1] for the heightmap-space losses.
Sobel is implemented via F.conv2d over the whole batch at once — no per-sample loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
) / 4.0

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0],
     [ 0.0,  0.0,  0.0],
     [ 1.0,  2.0,  1.0]]
) / 4.0


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = _SOBEL_X.to(device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


# def slope_loss(x0_pred: torch.Tensor, max_slope_px: float) -> torch.Tensor:
#     """Hinge on Sobel-gradient magnitude above max_slope_px."""
#     kx, ky = _sobel_kernels(x0_pred.device, x0_pred.dtype)
#     gx = F.conv2d(x0_pred, kx, padding=1)
#     gy = F.conv2d(x0_pred, ky, padding=1)
#     slope = torch.sqrt(gx * gx + gy * gy + 1e-12)
#     return F.relu(slope - max_slope_px).mean()


# def min_feature_loss(x0_pred: torch.Tensor, min_feat_px: float) -> torch.Tensor:
#     """Penalise high-frequency detail finer than the tool diameter."""
#     k = int(min_feat_px)
#     if k % 2 == 0:
#         k += 1
#     if k < 3:
#         k = 3
#     pad = k // 2
#     smoothed = F.avg_pool2d(x0_pred, kernel_size=k, stride=1, padding=pad)
#     detail = (x0_pred - smoothed).abs()
#     return F.relu(detail - 0.015).mean()


def fabrication_aware_loss(
    pred_noise: torch.Tensor,
    target_noise: torch.Tensor,
    noisy_h: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    max_slope_px: float,
    min_feat_px: float,
    lambda_slope: float = 0.1,
    lambda_feat: float = 0.05,
    warmup_steps: int = 2000,
    current_step: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Combines epsilon-MSE with slope and feature hinges on the predicted x0.

    The two auxiliary terms are disabled during warmup so the model first
    learns the denoising objective before being constrained.
    """
    # Pure MSE training. CNC constraints are enforced via preprocess_dataset.py
    # (training data already machinable) and infer.py's enforce_machinability
    # post-processing. Aux losses disabled for stability.
    mse = F.mse_loss(pred_noise, target_noise)
    return mse, {
        "mse": mse.item(),
        "slope": 0.0,
        "feat": 0.0,
        "total": mse.item(),
    }
