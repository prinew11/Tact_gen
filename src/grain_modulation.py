"""
grain_modulation.py

Anisotropic contour smoothing guided by wood grain direction.

Instead of warping the heightfield (which has little effect after
quantisation), this module operates on the quantised label map and
smooths contour boundaries anisotropically — along grain direction,
not across it. This directly bends iso-height contours to follow
the wood grain.

Usage:
    from grain_modulation import smooth_labels_by_grain
    labels = smooth_labels_by_grain(labels, image_gray, n_levels)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def remove_illumination(
    gray: np.ndarray,
    sigma_illumination: float = 32.0,
) -> np.ndarray:
    gray = gray.astype(np.float32)
    illumination = gaussian_filter(gray, sigma=sigma_illumination)
    corrected = gray - illumination
    lo, hi = float(corrected.min()), float(corrected.max())
    if hi - lo < 1e-7:
        return np.zeros_like(gray)
    return ((corrected - lo) / (hi - lo)).astype(np.float32)


def compute_grain_angle(
    gray: np.ndarray,
    sigma_noise: float = 1.0,
    sigma_integration: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        grain_angle : direction along grain (radians), shape (H, W)
        coherence   : anisotropy [0, 1], shape (H, W)
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if sigma_noise > 0:
        gx = gaussian_filter(gx, sigma=sigma_noise)
        gy = gaussian_filter(gy, sigma=sigma_noise)

    Jxx = gaussian_filter(gx * gx, sigma=sigma_integration)
    Jyy = gaussian_filter(gy * gy, sigma=sigma_integration)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_integration)

    tr   = Jxx + Jyy
    disc = np.sqrt(np.maximum((Jxx - Jyy) ** 2 * 0.25 + Jxy ** 2, 0.0))
    l1   = tr * 0.5 + disc
    l2   = tr * 0.5 - disc

    coherence = np.where(
        tr > 1e-8,
        ((l1 - l2) / (l1 + l2 + 1e-8)) ** 2,
        0.0,
    ).astype(np.float32)

    # gradient angle -> grain angle = gradient + 90 deg
    gradient_angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy + 1e-8)
    grain_angle = gradient_angle + np.pi / 2.0

    return grain_angle, coherence


def _anisotropic_smooth_labels(
    labels: np.ndarray,
    grain_angle: np.ndarray,
    coherence: np.ndarray,
    along_sigma: float = 6.0,
    across_sigma: float = 0.5,
) -> np.ndarray:
    """
    Smooth label map anisotropically along grain direction.

    For each level, the binary mask is smoothed with an elongated
    kernel aligned to the local grain direction. This rounds contour
    edges along the grain while keeping them sharp across the grain.

    along_sigma  : blur radius along grain (px) — larger = more following
    across_sigma : blur radius across grain (px) — keep small to preserve steps
    """
    H, W = labels.shape
    result = labels.astype(np.float32).copy()

    # Work on float label map, smooth, then re-quantise
    # Use steerable filter: decompose into along/across components
    # Mean angle for the whole image (sufficient for parallel grain)
    mean_angle = float(grain_angle[coherence > 0.3].mean()) \
        if (coherence > 0.3).any() else 0.0

    print(f"  [grain] mean grain angle: {np.degrees(mean_angle):.1f}deg")
    print(f"  [grain] mean coherence:   {coherence.mean():.3f}")

    # Build anisotropic kernel: elongated along grain direction
    # Kernel size based on along_sigma
    ksize = int(along_sigma * 4) | 1  # ensure odd
    ksize = max(ksize, 3)

    # 1D Gaussian kernels
    k_along  = cv2.getGaussianKernel(ksize, along_sigma)
    k_across = cv2.getGaussianKernel(max(int(across_sigma * 4) | 1, 3),
                                      across_sigma)
    kernel_2d = k_along @ k_across.T  # elongated along Y

    # Rotate kernel to align with grain direction
    # grain_angle is measured from X axis; rotate kernel accordingly
    angle_deg = float(np.degrees(mean_angle))
    rot_mat = cv2.getRotationMatrix2D(
        (ksize // 2, ksize // 2), -angle_deg, 1.0
    )
    kernel_rotated = cv2.warpAffine(
        kernel_2d.astype(np.float32),
        rot_mat,
        (ksize, ksize),
        flags=cv2.INTER_LINEAR,
    )
    # Normalise kernel
    kernel_rotated = kernel_rotated / (kernel_rotated.sum() + 1e-8)

    # Smooth the float label map with the anisotropic kernel
    labels_float = labels.astype(np.float32)
    smoothed = cv2.filter2D(labels_float, -1, kernel_rotated,
                             borderType=cv2.BORDER_REFLECT)

    # Re-quantise: round back to nearest integer level
    n_levels = int(labels.max()) + 1
    result = np.clip(np.round(smoothed), 0, n_levels - 1).astype(np.int32)

    return result


def smooth_labels_by_grain(
    labels: np.ndarray,
    image_gray: np.ndarray,
    n_levels: int,
    along_sigma: float = 8.0,
    across_sigma: float = 0.5,
    sigma_noise: float = 1.0,
    sigma_integration: float = 16.0,
) -> np.ndarray:
    """
    Main entry point.

    Smooths quantised label contours anisotropically along wood grain.
    Contour edges parallel to grain are smoothed (rounded to follow grain);
    edges perpendicular to grain are kept sharp.

    Args:
        labels      : int32 array, shape (H, W), values 0..n_levels-1
        image_gray  : float32 [0,1] grayscale image, any size
        n_levels    : number of discrete height levels
        along_sigma : blur along grain direction (px). 6-12 recommended.
        across_sigma: blur across grain direction (px). Keep <= 1.0.

    Returns:
        Smoothed label map, int32, same shape as input.
    """
    H, W = labels.shape

    if image_gray.shape != (H, W):
        image_gray = cv2.resize(
            image_gray.astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_AREA,
        )

    # Remove illumination before analysis
    image_clean = remove_illumination(image_gray, sigma_illumination=32.0)

    grain_angle, coherence = compute_grain_angle(
        image_clean, sigma_noise, sigma_integration
    )

    return _anisotropic_smooth_labels(
        labels, grain_angle, coherence,
        along_sigma=along_sigma,
        across_sigma=across_sigma,
    )
