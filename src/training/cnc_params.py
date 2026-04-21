"""
CNC physical constants. Single source of truth — do not hardcode these
anywhere else in the codebase.
"""
from __future__ import annotations

import math

# --- Fixed physical parameters (workpiece + tooling) ---
WORKPIECE_MM: float = 100.0          # square side length
IMAGE_PX: int = 512                  # heightmap resolution
MAX_DEPTH_MM: float = 5.0            # carving depth
TOOL_DIAMETER_MM: float = 6.0        # ball end mill
MAX_SLOPE_DEG: float = 40.0          # 3-axis machining limit (5° safety margin)

# --- Derived parameters ---
MM_PER_PX: float = WORKPIECE_MM / IMAGE_PX
MIN_FEAT_PX: float = TOOL_DIAMETER_MM / MM_PER_PX
Z_SCALE_RATIO: float = MAX_DEPTH_MM / WORKPIECE_MM
MAX_SLOPE_PX: float = math.tan(math.radians(MAX_SLOPE_DEG)) * Z_SCALE_RATIO
FEATURE_SIGMA: float = MIN_FEAT_PX / (2.0 * math.pi)


print(
    f"[cnc_params] workpiece={WORKPIECE_MM}mm  px={IMAGE_PX}  "
    f"mm/px={MM_PER_PX:.4f}  min_feat={MIN_FEAT_PX:.2f}px  "
    f"max_slope_px={MAX_SLOPE_PX:.4f}  feature_sigma={FEATURE_SIGMA:.2f}px"
)
