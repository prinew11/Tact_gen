import sys
import numpy as np
import cv2

sys.path.append(r"d:\gitproject\Tact_gen\src")

from terrace_geometry import (
    TerraceConfig,
    _quantize,
    _enforce_min_recess_width,
    _resolve_checkerboard,
)

hf = np.load(r"d:\gitproject\Tact_gen\outputs\agent_run\heightfield_agent.npy")
hf = np.clip(hf, 0.0, 1.0).astype(np.float32)

tc = TerraceConfig(
    physical_size_mm=100.0,
    max_height_mm=20.0,
    tool_diameter_mm=6.0,
    terrace_steps=10,
    mesh_resolution=256,
)

# 和 heightfield_to_terrace_mesh() 保持一致：先 resize
if hf.shape[0] != tc.mesh_resolution or hf.shape[1] != tc.mesh_resolution:
    hf = cv2.resize(
        hf,
        (tc.mesh_resolution, tc.mesh_resolution),
        interpolation=cv2.INTER_AREA,
    )

px_size = tc.physical_size_mm / (tc.mesh_resolution - 1)
tool_radius_px = (tc.tool_diameter_mm / 2.0) / px_size

def show_counts(name, labels):
    vals, counts = np.unique(labels, return_counts=True)
    print("\n" + name)
    print("unique:", vals)
    print("counts:", dict(zip(vals.tolist(), counts.tolist())))

labels0 = _quantize(hf, tc.terrace_steps)
show_counts("after quantize", labels0)

labels1 = _enforce_min_recess_width(labels0, tool_radius_px, tc.terrace_steps)
show_counts("after enforce_min_recess_width", labels1)

labels2 = _resolve_checkerboard(labels1)
show_counts("after resolve_checkerboard", labels2)

print("\npx_size:", px_size)
print("tool_radius_px:", tool_radius_px)
print("tool_diameter_px:", tool_radius_px * 2)