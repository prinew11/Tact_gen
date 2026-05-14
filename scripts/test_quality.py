"""Quick quality check: verify 4 intents produce std >= 0.05."""
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TactileIntent:
    gamma: float = 1.0
    edge_sigma: float = 1.5
    morph_strength: float = 1.0
    terrace_steps: int = 8
    physical_size_mm: float = 150.0
    target_resolution: int = 512


def normalize_heightfield(hf):
    hf = hf.astype(np.float32)
    p_low, p_high = np.percentile(hf, 2), np.percentile(hf, 98)
    hf = np.clip(hf, p_low, p_high)
    return ((hf - p_low) / (p_high - p_low + 1e-8)).astype(np.float32)


def suppress_narrow_recesses(heightfield, tool_radius_px):
    r = max(int(math.ceil(tool_radius_px)), 1)
    inv = (1.0 - heightfield).astype(np.float32)
    rect = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    opened = grey_dilation(grey_erosion(inv, footprint=rect), footprint=rect)
    return np.clip(1.0 - opened, 0.0, 1.0).astype(np.float32)


def preprocess_for_terrace(heightfield, tool_diameter_mm=6.0, intent=None):
    if intent is None:
        intent = TactileIntent()
    physical_size_mm = intent.physical_size_mm
    target_resolution = intent.target_resolution
    hf = normalize_heightfield(heightfield)
    if hf.shape[0] != target_resolution or hf.shape[1] != target_resolution:
        hf = cv2.resize(hf, (target_resolution, target_resolution),
                        interpolation=cv2.INTER_AREA)
    px_size = physical_size_mm / (target_resolution - 1)
    tool_radius_mm = tool_diameter_mm / 2.0
    tool_radius_px = tool_radius_mm / px_size
    gamma_safe = float(np.clip(intent.gamma, 0.85, 1.8))
    hf = np.power(np.clip(hf, 0.0, 1.0), gamma_safe).astype(np.float32)
    edge_sigma_px = max(intent.edge_sigma / px_size, 0.5)
    hf = gaussian_filter(hf.astype(np.float32), sigma=edge_sigma_px)
    # Skip morph if morph_strength <= 0.5 (preserve fine rough texture)
    if intent.morph_strength > 0.5:
        effective_radius_px = tool_radius_px * intent.morph_strength
        max_radius_px = target_resolution * 0.05
        effective_radius_px = min(effective_radius_px, max_radius_px)
        hf = suppress_narrow_recesses(hf, effective_radius_px)
        print("  [morph] applied: radius={:.1f}px".format(effective_radius_px))
    else:
        print("  [morph] skipped: morph_strength={} <= 0.5".format(intent.morph_strength))

    # Step 3: edge_sigma
    edge_sigma_px = np.interp(intent.edge_sigma, [0.5, 4.5], [0.5, 3.0])
    hf = gaussian_filter(hf.astype(np.float32), sigma=edge_sigma_px)

    # Final anti-spike pass
    hf = gaussian_filter(hf, sigma=1.5)

    return np.clip(hf, 0.0, 1.0).astype(np.float32)


hmap = (ROOT.parent / "dataset_split" / "heightmap"
        / "Wood__dark_shiny_wood_1_01_height.png")
hf = np.array(Image.open(hmap).convert("L"), dtype=np.float32) / 255.0

test_intents = {
    "rough":   TactileIntent(gamma=2.0, edge_sigma=0.5, morph_strength=0.4,
                             terrace_steps=10, physical_size_mm=150.0),
    "soft":    TactileIntent(gamma=1.0, edge_sigma=4.5, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
    "hard":    TactileIntent(gamma=1.0, edge_sigma=0.5, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
    "organic": TactileIntent(gamma=1.3, edge_sigma=2.0, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
}

print("{:<10} {:>8}  {:>6}  {:>6}  {}".format("Intent", "std", "min", "max", "PASS?"))
print("-" * 45)
for name, intent in test_intents.items():
    out = preprocess_for_terrace(hf, intent=intent)
    std = out.std()
    passed = "PASS" if std >= 0.05 else "FAIL too flat"
    print("{:<10} {:.4f}  {:.3f}  {:.3f}  {}".format(
        name, std, out.min(), out.max(), passed))
