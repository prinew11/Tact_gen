"""
Pipeline orchestrator: runs all stages end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    input_image: str
    output_dir: str = "outputs"
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_radius_mm: float = 3.0        # 6 mm diameter ball-end mill
    material_hint: str | None = None
    apply_machining_filter: bool = True


def run_pipeline(config: PipelineConfig) -> None:
    from preprocessing import preprocess
    from tactile_mapping import map_features
    from diffusion_pipeline import DiffusionConfig, generate_heightfield, save_heightfield
    from memory_store import MemoryStore
    from fabrication_corrector import (
        CorrectionConfig,
        correct_heightfield,
        print_correction_report,
    )
    from geometry import GeometryConfig, heightfield_to_mesh, save_stl
    from mockup import generate_mockup
    from fabrication import FabricationConfig, check_mesh, print_report

    out = Path(config.output_dir)
    store = MemoryStore()

    print("=== Stage 1: Preprocessing ===")
    features = preprocess(config.input_image)

    print("=== Stage 2: Tactile mapping ===")
    descriptor = map_features(features)
    print(f"  {descriptor}")

    print("=== Stage 3: Diffusion (trained model) ===")
    diff_cfg = DiffusionConfig()
    heightfield = generate_heightfield(features["rgb"], diff_cfg)
    hf_raw_path = out / "heightfields" / "heightfield_raw.npy"
    save_heightfield(heightfield, hf_raw_path)
    print(f"  Raw heightfield saved: {hf_raw_path}")

    print("=== Stage 3.5: Memory-driven fabrication correction ===")
    corr_cfg = CorrectionConfig(
        physical_size_mm=config.physical_size_mm,
        max_height_mm=config.max_height_mm,
        tool_radius_mm=config.tool_radius_mm,
    )
    heightfield, corr_report = correct_heightfield(
        heightfield, corr_cfg, store=store, material_hint=config.material_hint
    )
    print_correction_report(corr_report)
    hf_path = out / "heightfields" / "heightfield.npy"
    save_heightfield(heightfield, hf_path)
    print(f"  Corrected heightfield saved: {hf_path}")

    if config.apply_machining_filter:
        print("=== Stage 3.8: Machining filter ===")
        from machining_filter import (
            MachiningFilterConfig,
            filter_heightfield_for_machining,
            save_report_json,
            save_heightfield as save_machinable,
        )
        mf_cfg = MachiningFilterConfig(
            physical_size_mm=config.physical_size_mm,
            max_height_mm=config.max_height_mm,
            tool_radius_mm=config.tool_radius_mm,
        )
        heightfield, mf_report = filter_heightfield_for_machining(heightfield, mf_cfg)
        hf_machinable_path = out / "heightfields" / "heightfield_machinable.npy"
        save_machinable(heightfield, hf_machinable_path)
        save_report_json(mf_report, out / "heightfields" / "machining_filter_report.json")
        print(f"  Passed: {mf_report.passed}")
        print(
            f"  Slope: {mf_report.max_slope_deg_before:.1f}° → "
            f"{mf_report.max_slope_deg_after:.1f}°"
        )
        print(f"  Height scale: {mf_report.height_scale_applied:.3f}")
        for issue in mf_report.issues:
            print(f"  WARNING: {issue}")

    print("=== Stage 4: Geometry (STL) ===")
    geo_cfg = GeometryConfig(
        physical_size_mm=config.physical_size_mm,
        max_height_mm=config.max_height_mm,
    )
    mesh = heightfield_to_mesh(heightfield, geo_cfg)
    stl_path = out / "stl_fabrication" / "tactile.stl"
    save_stl(mesh, stl_path)

    print("=== Stage 5: Fabrication check ===")
    fab_cfg = FabricationConfig(tool_radius_mm=config.tool_radius_mm)
    report = check_mesh(mesh, fab_cfg)
    print_report(report)

    print("=== Stage 6: Mockup (OBJ preview) ===")
    mockup_path = out / "mockup" / "preview.obj"
    generate_mockup(
        heightfield,
        mockup_path,
        physical_size_mm=config.physical_size_mm,
        max_height_mm=config.max_height_mm,
    )

    print("\nPipeline complete.")
    if not report.passes:
        print("WARNING: Mesh failed fabrication checks — review before CAM.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <input_image> [material_hint]")
        sys.exit(1)

    material = sys.argv[2] if len(sys.argv) > 2 else None
    cfg = PipelineConfig(input_image=sys.argv[1], material_hint=material)
    run_pipeline(cfg)
