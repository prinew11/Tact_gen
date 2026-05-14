"""
Test script: run preprocess_for_terrace() with 4 intent anchors,
generate preview images for visual comparison.

Usage:
    python scripts/test_intent.py
    python scripts/test_intent.py --heightmap path/to/any_heightmap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from terrace_geometry import TactileIntent, preprocess_for_terrace

DEFAULT_HEIGHTMAP = (
    ROOT.parent / "dataset_split" / "heightmap"
    / "Wood__dark_shiny_wood_1_01_height.png"
)

INTENTS = {
    "rough":   TactileIntent(gamma=2.0, edge_sigma=0.5, morph_strength=0.4,
                             terrace_steps=10, physical_size_mm=150.0),
    "soft":    TactileIntent(gamma=1.0, edge_sigma=4.5, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
    "hard":    TactileIntent(gamma=1.0, edge_sigma=0.5, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
    "organic": TactileIntent(gamma=1.3, edge_sigma=2.0, morph_strength=0.8,
                             terrace_steps=8,  physical_size_mm=150.0),
}

OUT_DIR = ROOT / "outputs" / "intent_preview"


def main():
    hmap_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HEIGHTMAP
    if not hmap_path.exists():
        print(f"ERROR: heightmap not found: {hmap_path}")
        sys.exit(1)

    hf = np.array(Image.open(hmap_path).convert("L"), dtype=np.float32) / 255.0
    print(f"Loaded: {hmap_path.name}  shape={hf.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, intent in INTENTS.items():
        hf_out, _ = preprocess_for_terrace(hf, intent=intent)
        out_u8 = (hf_out * 255).astype(np.uint8)
        out_path = OUT_DIR / f"preview_{name}.png"
        Image.fromarray(out_u8).save(str(out_path))
        print(f"  {name:8s}: min={hf_out.min():.3f}  max={hf_out.max():.3f}  "
              f"std={hf_out.std():.3f}  -> {out_path.name}")

    print(f"\nDone. Check {OUT_DIR}/ for 4 preview images.")


if __name__ == "__main__":
    main()
