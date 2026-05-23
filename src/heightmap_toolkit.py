"""
Heightmap manipulation toolkit for creative transformations.

All operations:
  - Input:  float32 (H, W) in [0, 1]
  - Output: float32 (H, W) in [0, 1] (clipped internally)
  - Deterministic given same parameters
  - Composable (output of any op feeds any other op)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


# ── helpers ──────────────────────────────────────────────────────────────────

def _validate(hf: np.ndarray) -> np.ndarray:
    """Ensure input is float32, 2-D, and in [0, 1]."""
    hf = np.asarray(hf, dtype=np.float32)
    if hf.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {hf.ndim}-D")
    return np.clip(hf, 0.0, 1.0)


def _perlin_2d(shape: tuple[int, int], scale: float, seed: int) -> np.ndarray:
    """Generate 2D Perlin-like noise via multi-octave gradient noise."""
    rng = np.random.RandomState(seed)
    h, w = shape
    result = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    for _ in range(int(max(1, scale))):
        gy, gx = h // max(1, int(freq)) + 2, w // max(1, int(freq)) + 2
        gradients_y = rng.randn(gy + 1, gx + 1).astype(np.float32)
        gradients_x = rng.randn(gy + 1, gx + 1).astype(np.float32)
        ys = np.linspace(0, gy - 1, h, dtype=np.float32)
        xs = np.linspace(0, gx - 1, w, dtype=np.float32)
        yi, xi = np.floor(ys).astype(int), np.floor(xs).astype(int)
        yi = np.clip(yi, 0, gy - 1)
        xi = np.clip(xi, 0, gx - 1)
        yf, xf = ys - yi, xs - xi
        yf = yf[:, None] * np.ones((1, w), dtype=np.float32)
        xf = np.ones((h, 1), dtype=np.float32) * xf[None, :]
        def _dot(gy_arr, gx_arr, dy, dx):
            return gy_arr * dy + gx_arr * dx
        n00 = _dot(gradients_y[yi[:, None], xi[None, :]],
                    gradients_x[yi[:, None], xi[None, :]], yf, xf)
        n10 = _dot(gradients_y[yi[:, None] + 1, xi[None, :]],
                    gradients_x[yi[:, None] + 1, xi[None, :]], yf - 1, xf)
        n01 = _dot(gradients_y[yi[:, None], xi[None, :] + 1],
                    gradients_x[yi[:, None], xi[None, :] + 1], yf, xf - 1)
        n11 = _dot(gradients_y[yi[:, None] + 1, xi[None, :] + 1],
                    gradients_x[yi[:, None] + 1, xi[None, :] + 1], yf - 1, xf - 1)
        tx = xf * xf * (3 - 2 * xf)
        ty = yf * yf * (3 - 2 * yf)
        nx0 = n00 + tx * (n10 - n00)
        nx1 = n01 + tx * (n11 - n01)
        result += amplitude * (nx0 + ty * (nx1 - nx0))
        amplitude *= 0.5
        freq *= 2.0
    lo, hi = result.min(), result.max()
    if hi - lo > 1e-8:
        result = (result - lo) / (hi - lo)
    return result


# ── Category 1: Frequency Operations ────────────────────────────────────────

def boost_contrast(
    hf: np.ndarray,
    gamma: float = 1.5,
    midtone_offset: float = 0.0,
) -> np.ndarray:
    """Power-law contrast. gamma < 1 brightens, gamma > 1 darkens midtones."""
    hf = _validate(hf)
    midtone = np.clip(0.5 + midtone_offset, 0.01, 0.99)
    normalized = np.clip(hf, 0.0, 1.0)
    powered = np.power(normalized / midtone, gamma) * midtone
    return np.clip(powered, 0.0, 1.0).astype(np.float32)


def bandpass_filter(
    hf: np.ndarray,
    low_cutoff: float = 0.02,
    high_cutoff: float = 0.5,
    rolloff: float = 2.0,
) -> np.ndarray:
    """FFT bandpass: keep only spatial frequencies in [low, high]."""
    hf = _validate(hf)
    h, w = hf.shape
    fft = np.fft.fft2(hf)
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    max_r = min(cy, cx)
    low_r = low_cutoff * max_r
    high_r = high_cutoff * max_r
    mask = np.ones((h, w), dtype=np.float32)
    if high_r > low_r:
        low_falloff = np.exp(-0.5 * ((radius - low_r) / max(rolloff, 0.1)) ** 2)
        high_falloff = np.exp(-0.5 * ((radius - high_r) / max(rolloff, 0.1)) ** 2)
        mask = np.where(radius < low_r, low_falloff,
               np.where(radius > high_r, high_falloff, np.ones_like(radius)))
    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * mask)))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def enhance_ridges(
    hf: np.ndarray,
    strength: float = 0.5,
    scale_px: float = 8.0,
) -> np.ndarray:
    """Laplacian-of-Gaussian ridge enhancement."""
    hf = _validate(hf)
    sigma = max(scale_px / 3.0, 0.5)
    log = gaussian_filter(hf, sigma=sigma)
    laplace = gaussian_filter(hf, sigma=sigma, order=2)
    result = hf - strength * laplace
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def add_perlin_noise(
    hf: np.ndarray,
    amplitude: float = 0.1,
    frequency: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    """Add coherent Perlin noise."""
    hf = _validate(hf)
    noise = _perlin_2d(hf.shape, scale=frequency, seed=seed)
    result = hf + amplitude * (noise - 0.5)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 2: Regional Operations ─────────────────────────────────────────

def height_selective_transform(
    hf: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
    valley_boost: float = 0.0,
    ridge_boost: float = 0.0,
    transition_width: float = 0.1,
) -> np.ndarray:
    """Apply different transforms to valleys vs ridges."""
    hf = _validate(hf)
    tw = max(transition_width, 0.01)
    valley_mask = 0.5 * (1 - np.tanh((hf - low_threshold) / tw))
    ridge_mask = 0.5 * (1 + np.tanh((hf - high_threshold) / tw))
    result = hf + valley_boost * valley_mask + ridge_boost * ridge_mask
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def height_redistribute(
    hf: np.ndarray,
    target_distribution: str = "uniform",
    center: float = 0.5,
    width: float = 0.2,
) -> np.ndarray:
    """Remap height histogram to a target distribution."""
    hf = _validate(hf)
    flat = hf.flatten()
    sorted_vals = np.sort(flat)
    n = len(sorted_vals)
    cdf = np.searchsorted(sorted_vals, flat) / n
    cdf = np.clip(cdf, 0.0, 1.0)
    if target_distribution == "uniform":
        result = cdf
    elif target_distribution == "gaussian":
        from scipy.stats import norm
        result = norm.ppf(np.clip(cdf, 0.001, 0.999), loc=center, scale=width)
    elif target_distribution == "bimodal":
        w = max(width, 0.05)
        left = center - w
        right = center + w
        result = np.where(cdf < 0.5,
                          left + cdf * 2 * w,
                          right + (cdf - 0.5) * 2 * (1 - right))
    else:
        result = cdf
    return np.clip(result, 0.0, 1.0).astype(np.float32).reshape(hf.shape)


# ── Category 3: Directional Operations ───────────────────────────────────────

def directional_warp(
    hf: np.ndarray,
    angle_deg: float = 0.0,
    strength: float = 5.0,
    wavelength_px: float = 32.0,
) -> np.ndarray:
    """Warp heightmap along sinusoidal displacement in given direction."""
    hf = _validate(hf)
    h, w = hf.shape
    angle_rad = np.radians(angle_deg)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    proj = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    wavelength = max(wavelength_px, 4.0)
    displacement = strength * np.sin(2 * np.pi * proj / wavelength)
    dx = displacement * np.cos(angle_rad + np.pi / 2)
    dy = displacement * np.sin(angle_rad + np.pi / 2)
    src_y = yy + dy
    src_x = xx + dx
    coords = np.array([src_y.ravel(), src_x.ravel()])
    result = map_coordinates(hf, coords, order=1, mode='reflect')
    return np.clip(result.reshape(h, w), 0.0, 1.0).astype(np.float32)


def anisotropic_emphasis(
    hf: np.ndarray,
    angle_deg: float = 0.0,
    strength: float = 0.5,
) -> np.ndarray:
    """Sharpen features aligned with angle_deg, blur perpendicular."""
    hf = _validate(hf)
    angle_rad = np.radians(angle_deg)
    sigma_along = max(4.0 / max(strength, 0.01), 1.0)
    sigma_across = max(1.0, 0.5)
    blurred_along = _directional_blur(hf, angle_rad, sigma_along, sigma_across)
    result = hf + strength * (hf - blurred_along)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _directional_blur(
    img: np.ndarray, angle: float, sigma_along: float, sigma_across: float,
) -> np.ndarray:
    """Blur more along one direction than the other."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    along = xx * cos_a + yy * sin_a
    across = -xx * sin_a + yy * cos_a
    along_smooth = gaussian_filter(img, sigma=sigma_along)
    across_smooth = gaussian_filter(img, sigma=sigma_across)
    t = np.clip(sigma_along / (sigma_along + sigma_across + 1e-8), 0, 1)
    return t * along_smooth + (1 - t) * across_smooth


def bend_contours(
    hf: np.ndarray,
    curvature: float = 0.5,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> np.ndarray:
    """Bend iso-height contours toward/away from a center point."""
    hf = _validate(hf)
    h, w = hf.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = center_x * w, center_y * h
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
    angle = np.arctan2(dy, dx)
    bend_amount = curvature * dist * 0.05
    new_xx = xx + bend_amount * np.cos(angle)
    new_yy = yy + bend_amount * np.sin(angle)
    coords = np.array([new_yy.ravel(), new_xx.ravel()])
    result = map_coordinates(hf, coords, order=1, mode='reflect')
    return np.clip(result.reshape(h, w), 0.0, 1.0).astype(np.float32)


# ── Category 4: Pattern Blending ────────────────────────────────────────────

def blend_pattern(
    hf: np.ndarray,
    pattern_type: str = "waves",
    blend_mode: str = "add",
    amplitude: float = 0.1,
    frequency: float = 4.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Blend a procedural pattern into the heightmap."""
    hf = _validate(hf)
    h, w = hf.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32) / max(h, w)
    angle_rad = np.radians(angle_deg)
    proj = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
    freq = max(frequency, 0.1)
    if pattern_type == "waves":
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi * freq * proj)
    elif pattern_type == "concentric":
        dist = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi * freq * dist)
    elif pattern_type == "radial":
        angle_map = np.arctan2(yy - 0.5, xx - 0.5)
        pattern = 0.5 + 0.5 * np.sin(freq * angle_map)
    elif pattern_type == "crosshatch":
        p1 = np.sin(2 * np.pi * freq * proj)
        proj2 = xx * np.cos(angle_rad + np.pi / 2) + yy * np.sin(angle_rad + np.pi / 2)
        p2 = np.sin(2 * np.pi * freq * proj2)
        pattern = 0.5 + 0.25 * (p1 + p2)
    else:
        pattern = np.zeros_like(hf)
    pattern = pattern.astype(np.float32)
    if blend_mode == "add":
        result = hf + amplitude * (pattern - 0.5)
    elif blend_mode == "multiply":
        result = hf * (1.0 + amplitude * (pattern - 0.5))
    elif blend_mode == "overlay":
        mask = hf < 0.5
        result = np.where(mask,
                          2 * hf * (0.5 + amplitude * (pattern - 0.5)),
                          1 - 2 * (1 - hf) * (0.5 - amplitude * (pattern - 0.5)))
    else:
        result = hf
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def texture_overlay(
    hf: np.ndarray,
    texture_type: str = "stochastic",
    amplitude: float = 0.05,
    feature_size_px: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    """Add fine texture detail simulating surface material properties."""
    hf = _validate(hf)
    h, w = hf.shape
    rng = np.random.RandomState(seed)
    if texture_type == "stochastic":
        noise = rng.randn(h, w).astype(np.float32)
        sigma = max(feature_size_px / 3.0, 0.5)
        texture = gaussian_filter(noise, sigma=sigma)
    elif texture_type == "fibers":
        angle = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[:h, :w].astype(np.float32) / max(h, w)
        along = xx * np.cos(angle) + yy * np.sin(angle)
        freq = max(1.0 / max(feature_size_px / max(h, w), 0.01), 0.1)
        texture = np.sin(2 * np.pi * freq * along + rng.randn() * 6.28)
        texture = texture * (0.5 + 0.5 * rng.rand(h, w).astype(np.float32))
    elif texture_type == "cracks":
        noise = rng.randn(h, w).astype(np.float32)
        from scipy.ndimage import laplace
        texture = np.abs(laplace(gaussian_filter(noise, sigma=max(feature_size_px / 4, 0.5))))
    else:
        texture = np.zeros((h, w), dtype=np.float32)
    lo, hi = texture.min(), texture.max()
    if hi - lo > 1e-8:
        texture = (texture - lo) / (hi - lo)
    result = hf + amplitude * (texture - 0.5)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 5: Composition Operations ──────────────────────────────────────

def blend_two(
    hf_a: np.ndarray,
    hf_b: np.ndarray,
    alpha: float = 0.5,
    blend_mode: str = "linear",
) -> np.ndarray:
    """Blend two heightmaps."""
    hf_a = _validate(hf_a)
    hf_b = _validate(hf_b)
    if hf_a.shape != hf_b.shape:
        raise ValueError(f"Shape mismatch: {hf_a.shape} vs {hf_b.shape}")
    alpha = np.clip(alpha, 0.0, 1.0)
    if blend_mode == "linear":
        result = (1 - alpha) * hf_a + alpha * hf_b
    elif blend_mode == "multiply":
        result = hf_a * (1 - alpha + alpha * hf_b)
    elif blend_mode == "screen":
        result = 1 - (1 - hf_a) * (1 - alpha * hf_b)
    else:
        result = (1 - alpha) * hf_a + alpha * hf_b
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def mask_apply(
    hf_original: np.ndarray,
    hf_modified: np.ndarray,
    mask: np.ndarray,
    feather_px: float = 8.0,
) -> np.ndarray:
    """Apply modifications only within a feathered mask region."""
    hf_original = _validate(hf_original)
    hf_modified = _validate(hf_modified)
    mask = np.asarray(mask, dtype=np.float32)
    if hf_original.shape != hf_modified.shape:
        raise ValueError(f"Shape mismatch: {hf_original.shape} vs {hf_modified.shape}")
    if mask.shape != hf_original.shape:
        raise ValueError(f"Mask shape mismatch: {mask.shape} vs {hf_original.shape}")
    mask = np.clip(mask, 0.0, 1.0)
    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px / 3.0)
        mask = np.clip(mask, 0.0, 1.0)
    result = hf_original * (1 - mask) + hf_modified * mask
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 6: Creative Freedom ────────────────────────────────────────────

def replace_region(
    hf: np.ndarray,
    target_value: float = 0.5,
    shape: str = "ellipse",
    center_x: float = 0.5,
    center_y: float = 0.5,
    size_x: float = 0.3,
    size_y: float = 0.3,
    feather_px: float = 16.0,
) -> np.ndarray:
    """Replace a region with a flat value (plateau, pit, or flat zone)."""
    hf = _validate(hf)
    h, w = hf.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = center_x * w, center_y * h
    sx, sy = max(size_x * w, 1.0), max(size_y * h, 1.0)

    if shape == "ellipse":
        dist = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0)
    elif shape == "rectangle":
        dx = np.abs(xx - cx) / sx
        dy = np.abs(yy - cy) / sy
        inside = (dx <= 0.5) & (dy <= 0.5)
        mask = inside.astype(np.float32)
    elif shape == "diamond":
        dist = np.abs(xx - cx) / sx + np.abs(yy - cy) / sy
        mask = np.clip(1.0 - dist, 0.0, 1.0)
    else:
        dist = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0)

    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px / 3.0)
        mask = np.clip(mask, 0.0, 1.0)

    result = hf * (1 - mask) + target_value * mask
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def generate_mask(
    shape: str = "ellipse",
    center_x: float = 0.5,
    center_y: float = 0.5,
    size_x: float = 0.3,
    size_y: float = 0.3,
    invert: bool = False,
    feather_px: float = 16.0,
    resolution: int = 512,
) -> np.ndarray:
    """Generate a spatial mask from geometric primitives. Returns float32 [0,1]."""
    h, w = resolution, resolution
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = center_x * w, center_y * h
    sx, sy = max(size_x * w, 1.0), max(size_y * h, 1.0)

    if shape == "ellipse":
        dist = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0)
    elif shape == "rectangle":
        dx = np.abs(xx - cx) / sx
        dy = np.abs(yy - cy) / sy
        inside = (dx <= 0.5) & (dy <= 0.5)
        mask = inside.astype(np.float32)
    elif shape == "ring":
        dist = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        mask = np.clip(np.abs(dist - 0.5) * 2, 0.0, 1.0)
    elif shape == "gradient_x":
        mask = xx / w
    elif shape == "gradient_y":
        mask = yy / h
    else:
        dist = np.sqrt(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0)

    if invert:
        mask = 1.0 - mask

    if feather_px > 0:
        mask = gaussian_filter(mask, sigma=feather_px / 3.0)
        mask = np.clip(mask, 0.0, 1.0)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def height_zone_remap(
    hf: np.ndarray,
    zone_low: float = 0.0,
    zone_high: float = 0.3,
    target_low: float = 0.7,
    target_high: float = 1.0,
    blend_width: float = 0.05,
) -> np.ndarray:
    """Remap a specific height range to a new range. Can invert topography."""
    hf = _validate(hf)
    bw = max(blend_width, 0.001)
    zone_mask = 0.5 * (1 + np.tanh((hf - zone_low) / bw)) * \
                0.5 * (1 + np.tanh((zone_high - hf) / bw))
    if zone_high > zone_low:
        zone_frac = np.clip((hf - zone_low) / (zone_high - zone_low), 0.0, 1.0)
    else:
        zone_frac = np.zeros_like(hf)
    remapped = target_low + zone_frac * (target_high - target_low)
    result = hf * (1 - zone_mask) + remapped * zone_mask
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def surface_warp(
    hf: np.ndarray,
    control_points: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Large-scale deformation via sparse control-point displacement.

    control_points: list of (x_frac, y_frac, height_delta, radius_frac)
    """
    hf = _validate(hf)
    if control_points is None or len(control_points) == 0:
        return hf.copy()

    h, w = hf.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    displacement = np.zeros((h, w), dtype=np.float32)

    for cpx, cpy, delta, radius in control_points:
        px, py = cpx * w, cpy * h
        r = max(radius * max(h, w), 1.0)
        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        weight = np.exp(-0.5 * (dist / r) ** 2)
        displacement += delta * weight

    result = hf + displacement
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def procedural_generate(
    pattern_type: str = "perlin",
    size: int = 512,
    frequency: float = 4.0,
    amplitude: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a standalone procedural pattern from scratch."""
    shape = (size, size)
    if pattern_type == "perlin":
        result = _perlin_2d(shape, scale=frequency, seed=seed)
    elif pattern_type == "voronoi":
        rng = np.random.RandomState(seed)
        n_points = max(int(frequency * 4), 4)
        pts_y = rng.rand(n_points).astype(np.float32) * size
        pts_x = rng.rand(n_points).astype(np.float32) * size
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        min_dist = np.full((size, size), np.inf, dtype=np.float32)
        for py, px in zip(pts_y, pts_x):
            dist = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
            min_dist = np.minimum(min_dist, dist)
        max_d = min_dist.max() + 1e-8
        result = min_dist / max_d
    elif pattern_type == "brick":
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        brick_h = max(size // int(frequency), 2)
        brick_w = brick_h * 2
        row = np.floor(yy / brick_h).astype(int)
        offset = (row % 2) * (brick_w // 2)
        col = np.floor((xx - offset) / brick_w).astype(int)
        result = ((row + col) % 2).astype(np.float32)
    elif pattern_type == "hex_grid":
        yy, xx = np.mgrid[:size, :size].astype(np.float32)
        spacing = max(size / frequency, 2.0)
        hex_h = spacing * np.sqrt(3) / 2
        row = np.floor(yy / hex_h).astype(int)
        col_offset = (row % 2) * spacing / 2
        col = np.floor((xx - col_offset) / spacing).astype(int)
        cy_pts = row * hex_h
        cx_pts = col * spacing + col_offset
        d1 = np.sqrt((yy - cy_pts) ** 2 + (xx - cx_pts) ** 2)
        result = np.clip(d1 / (spacing * 0.5), 0, 1).astype(np.float32)
    else:
        result = _perlin_2d(shape, scale=frequency, seed=seed)

    result = amplitude * result
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def symmetry_apply(
    hf: np.ndarray,
    axis: str = "x",
) -> np.ndarray:
    """Apply symmetry: mirror or tile the heightmap."""
    hf = _validate(hf)
    if axis == "x":
        result = np.flip(hf, axis=1).copy()
    elif axis == "y":
        result = np.flip(hf, axis=0).copy()
    elif axis == "xy":
        result = np.flip(np.flip(hf, axis=0), axis=1).copy()
    elif axis == "tile_quadrant":
        h, w = hf.shape
        q = hf[:h // 2, :w // 2].copy()
        top = np.concatenate([q, np.flip(q, axis=1)], axis=1)
        bottom = np.flip(top, axis=0)
        result = np.concatenate([top, bottom], axis=0)
        if result.shape != hf.shape:
            result = result[:h, :w]
    else:
        result = hf.copy()
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 7: Frequency-Aware Operations ──────────────────────────────────

def freq_preserve_lowpass(
    hf: np.ndarray,
    cutoff: float = 0.15,
) -> np.ndarray:
    """Extract low-frequency base (large-area shape) via FFT lowpass."""
    hf = _validate(hf)
    h, w = hf.shape
    fft = np.fft.fft2(hf)
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    max_r = min(cy, cx)
    cutoff_r = cutoff * max_r
    mask = np.exp(-0.5 * (radius / max(cutoff_r, 1.0)) ** 2).astype(np.float32)
    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * mask)))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def freq_stepped_convert(
    hf: np.ndarray,
    n_levels: int = 4,
    freq_low: float = 0.15,
    freq_high: float = 0.8,
    dither: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Convert high-frequency undulations to machinable stepped stripes.

    Separates HF component via bandpass, quantizes to n_levels discrete
    heights, then adds back to LF base. Fine texture becomes flat terraces.
    """
    hf = _validate(hf)
    h, w = hf.shape

    fft = np.fft.fft2(hf)
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    max_r = min(cy, cx)

    low_r = freq_low * max_r
    high_r = freq_high * max_r

    low_mask = np.exp(-0.5 * (radius / max(low_r, 1.0)) ** 2).astype(np.float32)
    high_rolloff = max((max_r - high_r) * 0.3, 1.0)
    high_mask = np.clip(1.0 - np.exp(-0.5 * ((radius - high_r) / high_rolloff) ** 2), 0, 1).astype(np.float32)
    band_mask = np.clip(1.0 - low_mask, 0, 1) * np.clip(1.0 - high_mask, 0, 1)
    band_mask = np.clip(band_mask, 0, 1).astype(np.float32)

    hf_component = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * band_mask)))
    lf_component = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * low_mask)))

    hf_norm = hf_component.copy()
    lo, hi = hf_norm.min(), hf_norm.max()
    if hi - lo > 1e-8:
        hf_norm = (hf_norm - lo) / (hi - lo)
    else:
        hf_norm = np.full_like(hf_norm, 0.5)

    n_levels = max(int(n_levels), 2)
    if dither:
        rng = np.random.RandomState(seed)
        noise = rng.uniform(-0.5 / n_levels, 0.5 / n_levels, hf_norm.shape).astype(np.float32)
        quantized = np.floor(hf_norm * n_levels + noise) / max(n_levels - 1, 1)
    else:
        quantized = np.floor(hf_norm * n_levels) / max(n_levels - 1, 1)

    quantized = np.clip(quantized, 0.0, 1.0).astype(np.float32)

    result = lf_component + (quantized - 0.5) * 0.5
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def freq_band_boost(
    hf: np.ndarray,
    band_low: float = 0.0,
    band_high: float = 0.15,
    gain: float = 1.5,
) -> np.ndarray:
    """Boost or attenuate a specific frequency band.

    gain > 1 amplifies, gain < 1 suppresses.
    """
    hf = _validate(hf)
    h, w = hf.shape
    fft = np.fft.fft2(hf)
    fft_shift = np.fft.fftshift(fft)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    max_r = min(cy, cx)

    low_r = band_low * max_r
    high_r = band_high * max_r
    rolloff = max((high_r - low_r) * 0.2, 1.0)

    band_mask = np.clip(
        0.5 * (1 + np.tanh((radius - low_r) / rolloff)) *
        0.5 * (1 + np.tanh((high_r - radius) / rolloff)),
        0, 1,
    ).astype(np.float32)

    scale = 1.0 + (gain - 1.0) * band_mask
    result = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shift * scale)))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 8: Region-Aware Directional Operations ────────────────────────

def directional_step_convert(
    hf: np.ndarray,
    angle_deg: float = 0.0,
    n_steps: int = 8,
    step_width_px: float = 16.0,
    mask: np.ndarray | None = None,
    feather_px: float = 12.0,
) -> np.ndarray:
    """Convert parallel-stripe regions into directional stepped ridges.

    Instead of quantizing by height (which flattens low-gradient stripes),
    quantizes by the coordinate PERPENDICULAR to the grain direction.
    This creates stepped ridges that run parallel to the grain —
    tool cuts along grain direction, steps are across it.

    Args:
        hf: Input heightfield float32 [0,1].
        angle_deg: Grain/ridge direction in degrees (0=horizontal).
        n_steps: Number of discrete step levels (4-12).
        step_width_px: Minimum width per step in pixels.
            Controls the spatial frequency of stepping.
        mask: Optional float32 [0,1] mask. Only areas where mask > 0.5
            are converted; other areas pass through unchanged.
            If None, the entire heightfield is converted.
        feather_px: Feathering width at mask boundaries.

    Returns:
        Modified heightfield, float32 [0,1].
    """
    hf = _validate(hf)
    h, w = hf.shape
    angle_rad = np.radians(angle_deg)

    # Build coordinate grid
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0

    # Project each pixel onto the axis PERPENDICULAR to grain direction
    # grain direction = angle_deg, so perpendicular = angle_deg + 90
    perp_angle = angle_rad + np.pi / 2.0
    perp_coord = (xx - cx) * np.cos(perp_angle) + (yy - cy) * np.sin(perp_angle)

    # Quantize the perpendicular coordinate into n_steps levels
    coord_min = float(perp_coord.min())
    coord_max = float(perp_coord.max())
    coord_range = coord_max - coord_min

    if coord_range < 1e-6 or n_steps < 2:
        return hf.copy()

    # Normalize perpendicular coordinate to [0, 1]
    perp_norm = (perp_coord - coord_min) / coord_range

    # Quantize into n_steps levels
    n_steps = max(int(n_steps), 2)
    stepped = np.floor(perp_norm * n_steps) / max(n_steps - 1, 1)
    stepped = np.clip(stepped, 0.0, 1.0).astype(np.float32)

    # Blend original height with stepped version:
    # Use the stepped coordinate as a modulator — it creates the
    # "across-grain staircase" while preserving the original height
    # variation within each step band.
    # Strategy: remap original height through the step function
    # so that within each step band, height variation is preserved
    # but bands have distinct levels.
    band_idx = np.floor(perp_norm * n_steps).astype(np.int32)
    band_idx = np.clip(band_idx, 0, n_steps - 1)

    # Normalize height within each band
    result = hf.copy()
    for i in range(n_steps):
        mask_band = (band_idx == i)
        if not mask_band.any():
            continue
        band_vals = hf[mask_band]
        b_min, b_max = float(band_vals.min()), float(band_vals.max())
        if b_max - b_min > 1e-8:
            # Normalize within band to [0, 1]
            band_norm = (band_vals - b_min) / (b_max - b_min)
        else:
            band_norm = np.zeros_like(band_vals)

        # Map to stepped range: each band occupies 1/n_steps of the height range
        step_low = i / max(n_steps - 1, 1)
        step_high = (i + 1) / max(n_steps - 1, 1)
        # Preserve within-band variation but scale to 60% of band height
        # (40% gap between bands ensures visible steps)
        band_height = step_high - step_low
        result[mask_band] = step_low + band_norm * band_height * 0.6

    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    # Apply mask if provided — blend with original at boundaries
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape == hf.shape:
            if feather_px > 0:
                mask = gaussian_filter(mask, sigma=feather_px / 3.0)
                mask = np.clip(mask, 0.0, 1.0)
            result = hf * (1.0 - mask) + result * mask

    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Category 9: Image-Guided Restoration ──────────────────────────────────

def image_guided_ridge_restore(
    hf: np.ndarray,
    image_gray: np.ndarray,
    raw_blend_strength: float = 0.5,
    ridge_threshold: float = 0.02,
    smooth_boundary_px: float = 8.0,
) -> np.ndarray:
    """Restore ridge structures from original image where heightmap is too flat.

    The diffusion model often produces near-flat regions that lose the fine
    ridge/valley structure present in the original image. This function
    identifies flat regions (low local std) and blends in the original
    image structure to restore lost detail.

    Args:
        hf: Input heightfield float32 [0,1].
        image_gray: Original image grayscale float32 [0,1], same shape as hf.
        raw_blend_strength: Maximum blend factor for image structure [0, 0.8].
        ridge_threshold: Local std below which to restore ridges.
        smooth_boundary_px: Feathering width at blend boundaries.

    Returns:
        Modified heightfield, float32 [0,1].
    """
    hf = _validate(hf)
    image_gray = np.asarray(image_gray, dtype=np.float32)
    if image_gray.shape != hf.shape:
        raise ValueError(f"Shape mismatch: hf={hf.shape} vs image_gray={image_gray.shape}")
    image_gray = np.clip(image_gray, 0.0, 1.0)

    raw_blend_strength = np.clip(raw_blend_strength, 0.0, 0.8)
    ridge_threshold = max(ridge_threshold, 0.001)

    # Compute local std using box filter (efficient approximation)
    win_size = max(int(hf.shape[0] // 32), 8)
    if win_size % 2 == 0:
        win_size += 1

    import cv2
    hf_sq = hf * hf
    local_mean = cv2.blur(hf, (win_size, win_size))
    local_mean_sq = cv2.blur(hf_sq, (win_size, win_size))
    local_var = local_mean_sq - local_mean * local_mean
    local_std = np.sqrt(np.maximum(local_var, 0.0)).astype(np.float32)

    # Blend factor: stronger where local_std is lower (flatter regions)
    blend_factor = raw_blend_strength * np.clip(1.0 - local_std / ridge_threshold, 0.0, 1.0)

    # Smooth the blend boundary
    if smooth_boundary_px > 0:
        blend_factor = gaussian_filter(blend_factor, sigma=smooth_boundary_px / 3.0)
        blend_factor = np.clip(blend_factor, 0.0, raw_blend_strength)

    # Blend image structure into heightmap
    result = hf * (1.0 - blend_factor) + image_gray * blend_factor
    return np.clip(result, 0.0, 1.0).astype(np.float32)
