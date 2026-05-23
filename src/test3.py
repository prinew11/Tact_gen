#!/usr/bin/env python3
"""
Debug Agent Transform heightfield dynamic range and terrace label usage.

Usage:
    python tests/debug_agent_heightfields.py outputs/agent_run

Expected files:
    outputs/agent_run/heightfield_base.npy
    outputs/agent_run/heightfield_modified.npy
    outputs/agent_run/heightfield_final.npy
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---- Config: match your Agent Transform terrace config ----

TERRACE_STEPS = 12          # 改成你实际用的 terrace_steps，例如 10 / 12 / 16
PHYSICAL_SIZE_MM = 100.0    # 改成你实际物理尺寸
TOOL_DIAMETER_MM = 6.0      # 当前问题里的 6mm tool


# ---- Debug helpers ----

def debug_hf(name: str, hf: np.ndarray) -> None:
    hf = np.asarray(hf, dtype=np.float32)

    print(f"\n========== {name} heightfield ==========")
    print(f"shape: {hf.shape}")
    print(f"min/max: {float(hf.min()):.6f} / {float(hf.max()):.6f}")
    print(f"mean/std: {float(hf.mean()):.6f} / {float(hf.std()):.6f}")
    print(
        "p1/p5/p50/p95/p99:",
        np.percentile(hf, [1, 5, 50, 95, 99]).round(6).tolist(),
    )

    dynamic_1_99 = float(np.percentile(hf, 99) - np.percentile(hf, 1))
    dynamic_5_95 = float(np.percentile(hf, 95) - np.percentile(hf, 5))

    print(f"dynamic p1-p99: {dynamic_1_99:.6f}")
    print(f"dynamic p5-p95: {dynamic_5_95:.6f}")


def debug_labels(name: str, labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    u, c = np.unique(labels, return_counts=True)
    total = labels.size

    print(f"\n---------- {name} labels ----------")
    print("unique:", u.tolist())
    print("counts:", dict(zip(u.tolist(), c.tolist())))
    print(
        "percent:",
        {int(k): round(float(v) * 100.0 / total, 3) for k, v in zip(u, c)},
    )


def fallback_quantize(hf: np.ndarray, terrace_steps: int) -> np.ndarray:
    """
    Fallback quantization if project private quantizer cannot be imported.

    This maps [0,1] to integer labels [0, terrace_steps - 1].
    """
    hf = np.asarray(hf, dtype=np.float32)
    hf = np.clip(hf, 0.0, 1.0)

    labels = np.floor(hf * terrace_steps).astype(np.int32)
    labels = np.clip(labels, 0, terrace_steps - 1)

    return labels


def get_project_quantizer() -> Any | None:
    """
    Try to use the project's actual quantization function if it exists.

    Different versions may name it differently.
    """
    try:
        import terrace_geometry  # type: ignore
    except Exception as exc:
        print(f"[WARN] Could not import terrace_geometry: {exc}")
        return None

    for fn_name in [
        "_quantize",
        "_quantize_heightfield",
        "quantize_heightfield",
    ]:
        fn = getattr(terrace_geometry, fn_name, None)
        if callable(fn):
            print(f"[INFO] Using terrace_geometry.{fn_name} for quantization")
            return fn

    print("[WARN] No project quantizer found; using fallback_quantize()")
    return None


def quantize(hf: np.ndarray, terrace_steps: int) -> np.ndarray:
    fn = get_project_quantizer()

    if fn is None:
        return fallback_quantize(hf, terrace_steps)

    # Try common signatures.
    for args in [
        (hf, terrace_steps),
        (hf, terrace_steps - 1),
    ]:
        try:
            labels = fn(*args)
            return np.asarray(labels)
        except TypeError:
            continue

    print("[WARN] Project quantizer signature did not match; using fallback_quantize()")
    return fallback_quantize(hf, terrace_steps)


def try_recess_enforcement(labels: np.ndarray) -> None:
    """
    Run raw and/or guarded recess enforcement if project functions exist.

    This helps detect:
        after quantize: [6,7,8,9]
        after recess: [9]
    """
    try:
        import terrace_geometry  # type: ignore
    except Exception as exc:
        print(f"[WARN] Could not import terrace_geometry for recess check: {exc}")
        return

    h, w = labels.shape[:2]
    px_size = PHYSICAL_SIZE_MM / float(max(h, w) - 1)
    tool_radius_px = (TOOL_DIAMETER_MM / 2.0) / px_size
    tool_diameter_px = TOOL_DIAMETER_MM / px_size

    print("\n---------- physical scale ----------")
    print(f"px_size_mm: {px_size:.6f}")
    print(f"tool_radius_px: {tool_radius_px:.3f}")
    print(f"tool_diameter_px: {tool_diameter_px:.3f}")

    raw_fn = getattr(terrace_geometry, "_enforce_min_recess_width", None)
    guarded_fn = getattr(terrace_geometry, "_enforce_min_recess_width_guarded", None)

    if callable(raw_fn):
        try:
            raw_after = raw_fn(labels.copy(), tool_radius_px, TERRACE_STEPS)
            debug_labels("after RAW _enforce_min_recess_width", raw_after)

            if len(np.unique(raw_after)) <= 1 and len(np.unique(labels)) > 1:
                print(
                    "[FAIL] Raw recess enforcement collapsed labels to one level. "
                    "This is the old failure mode."
                )
        except Exception as exc:
            print(f"[WARN] Raw recess enforcement failed: {exc}")
    else:
        print("[INFO] No raw _enforce_min_recess_width found.")

    if callable(guarded_fn):
        try:
            guarded_after = guarded_fn(labels.copy(), tool_radius_px, TERRACE_STEPS)
            debug_labels("after GUARDED recess enforcement", guarded_after)

            if len(np.unique(guarded_after)) <= 1 and len(np.unique(labels)) > 1:
                print("[FAIL] Guarded recess enforcement still collapsed labels.")
            else:
                print("[OK] Guarded recess enforcement preserved multiple levels.")
        except Exception as exc:
            print(f"[WARN] Guarded recess enforcement failed: {exc}")
    else:
        print("[INFO] No _enforce_min_recess_width_guarded found.")


def load_required_npy(run_dir: Path, filename: str) -> np.ndarray:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def main() -> None:
    if len(sys.argv) < 2:
        run_dir = Path("outputs/agent_run")
        print(f"[INFO] No run_dir provided. Using default: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])

    base_hf = load_required_npy(run_dir, "heightfield_base.npy")
    modified_hf = load_required_npy(run_dir, "heightfield_modified.npy")
    final_hf = load_required_npy(run_dir, "heightfield_final.npy")

    for name, hf in [
        ("base_hf", base_hf),
        ("modified_hf", modified_hf),
        ("final_hf", final_hf),
    ]:
        debug_hf(name, hf)

        labels = quantize(hf, TERRACE_STEPS)
        debug_labels(f"{name} after quantize", labels)

        if name == "final_hf":
            try_recess_enforcement(labels)

    print("\n========== interpretation hints ==========")
    print("1. If final_hf p1-p99 dynamic range is very small, quantize will use few levels.")
    print("2. If final_hf after quantize has only 2-4 labels, the issue is height range compression.")
    print("3. If quantize has many labels but recess enforcement collapses to one label, the issue is recess handling.")
    print("4. If modified_hf looks fine but final_hf equals base_hf, validation likely rejected the agent result.")


if __name__ == "__main__":
    main()