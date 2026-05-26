"""
Gradio UI entry point: test each pipeline module independently.
Run:  python src/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# ensure src/ is importable
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


# ===== helpers =============================================================

def _arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255] for Gradio image display."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


# ===== 1. Preprocessing ====================================================

def run_preprocessing(image: np.ndarray | None):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        import cv2
        from preprocessing import load_image_gray, extract_edges, extract_frequency

        # Gradio gives RGB uint8 (H, W, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        edges = extract_edges(gray)
        freq = extract_frequency(gray)

        info = (
            f"**Preprocessing passed**\n\n"
            f"- Output shape: {gray.shape}\n"
            f"- Gray range: [{gray.min():.3f}, {gray.max():.3f}]\n"
            f"- Edges non-zero pixels: {(edges > 0).sum():,}\n"
            f"- Frequency mean: {freq.mean():.4f}"
        )
        return _arr_to_uint8(gray), _arr_to_uint8(edges), _arr_to_uint8(freq), info
    except Exception as e:
        raise gr.Error(f"Preprocessing failed: {e}")


# ===== 2. Tactile Mapping ===================================================

def run_tactile_mapping(image: np.ndarray | None):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        import cv2
        from preprocessing import extract_edges, extract_frequency
        from tactile_mapping import map_features, TactileDescriptor

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        features = {
            "gray": gray,
            "edges": extract_edges(gray),
            "frequency": extract_frequency(gray),
        }
        desc = map_features(features)

        info = (
            f"**Tactile Mapping passed**\n\n"
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Roughness | {desc.roughness:.4f} |\n"
            f"| Directionality | {desc.directionality:.4f} |\n"
            f"| Frequency | {desc.frequency:.4f} |"
        )
        return info
    except Exception as e:
        raise gr.Error(f"Tactile Mapping failed: {e}")


# ===== 2.5 Smart Crop =======================================================

def _local_roughness_map(gray: np.ndarray, win: int) -> np.ndarray:
    """Local std of intensity — high where texture varies fine."""
    import cv2
    mean = cv2.boxFilter(gray, -1, (win, win))
    sq_mean = cv2.boxFilter(gray * gray, -1, (win, win))
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    return np.sqrt(var)


def _local_directionality_map(gray: np.ndarray, win: int) -> np.ndarray:
    """Structure-tensor coherence — high where edges align in one direction."""
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.boxFilter(gx * gx, -1, (win, win))
    Jyy = cv2.boxFilter(gy * gy, -1, (win, win))
    Jxy = cv2.boxFilter(gx * gy, -1, (win, win))
    tr = Jxx + Jyy
    disc = np.sqrt(np.maximum((Jxx - Jyy) ** 2 * 0.25 + Jxy * Jxy, 0.0))
    l1, l2 = tr * 0.5 + disc, tr * 0.5 - disc
    return ((l1 - l2) / (l1 + l2 + 1e-8)).astype(np.float32)


def _local_frequency_map(gray: np.ndarray) -> np.ndarray:
    """|Laplacian| — proxy for high spatial frequency content per pixel."""
    import cv2
    return np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    rng = float(x.max() - lo)
    return ((x - lo) / rng).astype(np.float32) if rng > 1e-8 else np.zeros_like(x, np.float32)


def run_smart_crop(image: np.ndarray | None, crop_fraction: float,
                   w_rough: float, w_dir: float, w_freq: float,
                   mode: str, manual_cx_frac: float, manual_cy_frac: float):
    """Smart crop v3: weighted local features + auto/manual crop position."""
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        side = int(min(h, w) * float(crop_fraction))
        side = max(side, 16)
        side = min(side, min(h, w))

        # Local feature window for roughness/directionality estimation —
        # smaller than the crop window so the score has spatial structure.
        local_win = max(side // 8, 5)
        if local_win % 2 == 0:
            local_win += 1

        R = _normalize01(_local_roughness_map(gray, local_win))
        D = _normalize01(_local_directionality_map(gray, local_win))
        F = _normalize01(_local_frequency_map(gray))

        # Normalize weights to sum to 1 so absolute slider values are decoupled
        wsum = max(float(w_rough) + float(w_dir) + float(w_freq), 1e-6)
        wr, wd, wf = w_rough / wsum, w_dir / wsum, w_freq / wsum
        score_pixel = wr * R + wd * D + wf * F

        # Window-mean to find the best crop center
        score = cv2.boxFilter(score_pixel, ddepth=-1, ksize=(side, side))

        half = side // 2
        cx_lo, cx_hi = half, w - (side - half)
        cy_lo, cy_hi = half, h - (side - half)

        if mode == "Manual":
            cx = int(round(float(manual_cx_frac) * w))
            cy = int(round(float(manual_cy_frac) * h))
            cx = max(cx_lo, min(cx_hi - 1, cx))
            cy = max(cy_lo, min(cy_hi - 1, cy))
        else:
            valid_mask = np.zeros_like(score, dtype=bool)
            valid_mask[cy_lo:cy_hi, cx_lo:cx_hi] = True
            score_masked = np.where(valid_mask, score, -1.0)
            cy, cx = np.unravel_index(int(np.argmax(score_masked)), score.shape)

        y0, y1 = cy - half, cy - half + side
        x0, x1 = cx - half, cx - half + side
        cropped = image[y0:y1, x0:x1].copy()

        # Heatmap overlay
        heat = (_normalize01(score) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image.astype(np.uint8), 0.55, heat_color, 0.45, 0)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 80, 80), 3)

        out_dir = OUT / "cropped"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cropped_input.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        # Per-region descriptor values to help the user understand why
        crop_R = float(R[y0:y1, x0:x1].mean())
        crop_D = float(D[y0:y1, x0:x1].mean())
        crop_F = float(F[y0:y1, x0:x1].mean())

        info = (
            f"**Smart Crop v3 done** ({mode})\n\n"
            f"- Input size: {w}x{h}\n"
            f"- Crop size: {side}x{side} ({crop_fraction*100:.0f}% of min side)\n"
            f"- Crop center: ({cx}, {cy})  Normalized: ({cx/w:.3f}, {cy/h:.3f})\n"
            f"- Range: x[{x0}:{x1}], y[{y0}:{y1}]\n\n"
            f"**Weights (normalized)**: rough={wr:.2f}, dir={wd:.2f}, freq={wf:.2f}\n\n"
            f"**Region averages**\n"
            f"- Roughness: {crop_R:.3f}\n"
            f"- Directionality: {crop_D:.3f}\n"
            f"- Frequency (|Laplacian|): {crop_F:.3f}\n"
            f"- Score: {float(score[cy, cx]):.3f}\n\n"
            f"Saved: `{out_path.relative_to(ROOT)}`"
        )
        return overlay, cropped, info
    except Exception as e:
        raise gr.Error(f"Smart Crop failed: {e}")


# ===== 3. Diffusion Pipeline ===============================================

def run_diffusion(image: np.ndarray | None, steps: int):
    if image is None:
        raise gr.Error("Please upload an image first.")
    try:
        import cv2
        from diffusion_pipeline import DiffusionConfig, generate_heightfield

        rgb = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        config = DiffusionConfig(num_inference_steps=int(steps))
        t0 = time.time()
        hf = generate_heightfield(rgb, config)
        elapsed = time.time() - t0

        hf_raw_path = OUT / "heightfields" / "heightfield_raw.npy"
        hf_raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(hf_raw_path), hf)

        info = "\n".join([
            f"**Diffusion passed (local trained model)**\n",
            f"- Checkpoint: `{config.trained_model_path}`",
            f"- Device: {config.device}",
            f"- Sampling steps: {config.num_inference_steps}",
            f"- Elapsed: {elapsed:.1f}s",
            f"- Heightfield range: [{hf.min():.3f}, {hf.max():.3f}]",
            f"- Saved: `{hf_raw_path.relative_to(ROOT)}`",
        ])
        return _arr_to_uint8(hf), info
    except FileNotFoundError as e:
        raise gr.Error(f"Diffusion failed: {e}")
    except Exception as e:
        raise gr.Error(f"Diffusion failed: {e}")


# ===== Helpers ==============================================================

def _render_heightmap_3d(hf: np.ndarray, physical_size: float, max_height: float,
                         base_thickness: float = 0.0, title: str = "") -> np.ndarray:
    """3D surface preview of a heightfield."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = hf.shape
    xs = np.linspace(0, physical_size, w)
    ys = np.linspace(0, physical_size, h)
    X, Y = np.meshgrid(xs, ys)
    Z = hf * max_height + base_thickness

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="terrain", linewidth=0, antialiased=False, rcount=100, ccount=100)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_zlim(0, max_height + base_thickness if base_thickness else max_height)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


# ===== 4. Geometry (machining filter + STL) =================================

def _load_best_heightfield(heightfield_file) -> tuple[np.ndarray, str]:
    """Load heightfield: uploaded > raw > synthetic."""
    if heightfield_file is not None:
        return np.load(heightfield_file), "uploaded"
    raw = OUT / "heightfields" / "heightfield.npy"
    if raw.exists():
        return np.load(str(raw)), str(raw.relative_to(ROOT))
    return _make_test_heightfield(), "synthetic test"


# ===== 4. Saliency-Guided Geometry ==========================================

def run_saliency_geometry(
    heightfield_file,
    physical_size: float, max_height: float, tool_radius: float,
    steps_high: int, steps_low: int,
    thresh_high: float, thresh_low: float,
    mesh_resolution: int,
    fft_stride: int, fft_energy_threshold: float, weight_blur_sigma: float,
    structure_sigma: float, tool_angle: float, tool_tolerance: float,
    stripe_expansion_factor: float = 0.0,
    stripe_boost_strength: float = 2.5,
):
    """Multi-scale FFT + structure tensor + height range → machinability-weighted terrace."""
    try:
        from terrace_geometry import (
            MachiningFilterConfig, TerraceConfig, SaliencyConfig,
            run_saliency_pipeline, save_stl as terrace_save_stl,
        )

        hf, hf_source = _load_best_heightfield(heightfield_file)
        hf_raw_path = OUT / "heightfields" / "heightfield_raw.npy"
        hf_raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(hf_raw_path), hf)

        t0 = time.time()
        mesh, hf_processed, mach_report, ter_report, sal_report = run_saliency_pipeline(
            raw_heightmap_path=str(hf_raw_path),
            config=MachiningFilterConfig(
                physical_size_mm=float(physical_size),
                max_height_mm=float(max_height),
                tool_radius_mm=float(tool_radius),
                max_slope_deg=40.0,
            ),
            terrace_config=TerraceConfig(
                physical_size_mm=float(physical_size),
                max_height_mm=float(max_height),
                base_thickness_mm=3.0,
                tool_diameter_mm=float(tool_radius) * 2.0,
                mesh_resolution=int(mesh_resolution),
            ),
            saliency_config=SaliencyConfig(
                fft_stride=int(fft_stride),
                fft_energy_threshold=float(fft_energy_threshold),
                weight_blur_sigma=float(weight_blur_sigma),
                structure_sigma=float(structure_sigma),
                tool_angle_deg=float(tool_angle),
                tool_tolerance_mm=float(tool_tolerance),
                terrace_steps_high=int(steps_high),
                terrace_steps_low=int(steps_low),
                saliency_threshold_high=float(thresh_high),
                saliency_threshold_low=float(thresh_low),
                stripe_expansion_factor=float(stripe_expansion_factor),
                stripe_boost_strength=float(stripe_boost_strength),
            ),
            save_saliency_map=str(OUT / "weight_map.png"),
        )
        elapsed = time.time() - t0

        stl_path = OUT / "stl_fabrication" / "saliency_guided_terrace.stl"
        terrace_save_stl(mesh, stl_path)

        weight_map_img = np.array(
            __import__("PIL").Image.open(str(OUT / "weight_map.png"))
        )

        hf_preview = _arr_to_uint8(hf_processed)
        stl_preview = _render_heightmap_3d(
            hf_processed, physical_size, max_height,
            base_thickness=2.0,
            title="Saliency-Guided Terrace",
        )

        info = (
            f"**Machinability Pipeline done** ({elapsed:.2f}s)\n\n"
            f"- Input source: {hf_source}\n"
            f"- STL saved: `{stl_path.relative_to(ROOT)}`\n\n"
            f"**Weight Report**\n"
            f"- Mean weight: {sal_report.weight_mean:.3f}\n"
            f"- High-weight fraction: {sal_report.high_weight_fraction:.1%}\n"
            f"- Low-weight fraction: {sal_report.low_weight_fraction:.1%}\n"
            f"- Mean period: {sal_report.mean_period_px:.1f} px\n"
            f"- Mean coherence: {sal_report.mean_coherence:.3f}\n"
            f"- Mean height range: {sal_report.mean_height_range_mm:.3f} mm\n"
            f"- Non-periodic fraction: {sal_report.non_periodic_fraction:.1%}\n\n"
            f"**Machining Filter**\n"
            f"- Pixel size: {mach_report.pixel_size_mm:.3f} mm\n"
            f"- Passed: {mach_report.passed}\n\n"
            f"**Terrace Mesh**\n"
            f"- Terrace steps: {steps_low}–{steps_high} (adaptive)\n"
            f"- Grid resolution: {mesh_resolution}x{mesh_resolution}\n"
            f"- Faces: {ter_report.face_count:,}\n"
            f"- Watertight: {ter_report.watertight}"
        )
        if mach_report.issues:
            info += "\n\n**Issues:**\n" + "\n".join(f"- {i}" for i in mach_report.issues)

        return hf_preview, stl_preview, weight_map_img, info
    except Exception as e:
        raise gr.Error(f"Saliency pipeline failed: {e}")


# ===== 5. Mockup (OBJ preview) =============================================

def run_mockup(heightfield_file, physical_size: float, max_height: float):
    try:
        hf, _ = _load_best_heightfield(heightfield_file)

        from mockup import generate_mockup
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        obj_path = OUT / "mockup" / "preview.obj"
        t0 = time.time()
        generate_mockup(hf, obj_path, physical_size, max_height)
        elapsed = time.time() - t0

        # render to image for display
        import cv2
        small = cv2.resize(hf, (256, 256), interpolation=cv2.INTER_AREA)
        zv = small * max_height * 2.0  # z_scale=2.0

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(zv, cmap="terrain")
        plt.colorbar(im, ax=ax, label="Height (mm × 2)")
        ax.set_title("Mockup Preview (256×256)")
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        plot_img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)

        info = (
            f"**Mockup passed**\n\n"
            f"- Output resolution: 256x256\n"
            f"- Z scale factor: 2.0x\n"
            f"- Build time: {elapsed:.2f}s\n"
            f"- OBJ saved: `{obj_path.relative_to(ROOT)}`"
        )
        return plot_img, info
    except Exception as e:
        raise gr.Error(f"Mockup failed: {e}")


# ===== 6. Agent Transform ====================================================

def run_agent_transform(
    image: np.ndarray | None,
    user_intent: str,
    api_key: str,
    base_url: str,
    model: str,
    max_iterations: int,
    diffusion_steps: int,
    physical_size: float = 100.0,
    max_height: float = 20.0,
    tool_diameter: float = 6.0,
    terrace_steps: int = 12,
):
    """Run the agent pipeline: image -> diffusion -> preprocess -> agent plan -> validate -> terrace STL."""
    if image is None:
        raise gr.Error("Upload an image.")
    if not user_intent.strip():
        raise gr.Error("Enter a tactile intent description.")
    try:
        import cv2
        from agent_planner import run_agent_pipeline, AgentConfig
        from heightmap_analyzer import analysis_to_text, analysis_to_json
        from diffusion_pipeline import DiffusionConfig, generate_heightfield
        from pipeline import apply_agent_plan, validate_agent_modified_heightfield, normalize_for_terrace_quantize
        from terrace_geometry import (
            TerraceConfig,
            preprocess_for_terrace,
            heightfield_to_terrace_mesh,
            save_stl,
        )

        # Generate raw heightmap from image
        rgb = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        diff_config = DiffusionConfig(num_inference_steps=int(diffusion_steps))
        raw_hf = generate_heightfield(rgb, diff_config)

        t0 = time.time()

        # Image-guided blend: percentile stretch + mild denoise
        from scipy.ndimage import gaussian_filter as _gf
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # 1. Percentile soft normalization
        lo, hi = np.percentile(gray, 2), np.percentile(gray, 98)
        gray_norm = np.clip((gray - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        # 2. Mild denoise (only pixel noise, all wood grain preserved)
        gray_clean = _gf(gray_norm, sigma=1.5)
        # 3. Blend with diffusion output
        def _norm(x):
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.full_like(x, 0.5)
        blended_hf = np.clip(0.8 * gray_clean + 0.2 * _norm(raw_hf), 0.0, 1.0).astype(np.float32)

        # Terrace config (shared between preprocess and mesh generation)
        tc = TerraceConfig(
            physical_size_mm=float(physical_size),
            max_height_mm=float(max_height),
            tool_diameter_mm=float(tool_diameter),
            terrace_steps=int(terrace_steps),
        )

        # Preprocess FIRST — produces stable base heightfield
        base_hf = preprocess_for_terrace(
            blended_hf,
            tool_diameter_mm=tc.tool_diameter_mm,
            physical_size_mm=tc.physical_size_mm,
        )

        # Agent planning — agent directly executes tools
        agent_config = AgentConfig(
            model=model,
            max_iterations=int(max_iterations),
            api_key=api_key.strip(),
            base_url=base_url.strip(),
        )
        # Prepare image_gray for image_guided_ridge_restore tool
        image_gray = cv2.resize(gray, (base_hf.shape[1], base_hf.shape[0]), interpolation=cv2.INTER_AREA)
        result = run_agent_pipeline(base_hf, raw_hf, user_intent, agent_config, image_gray=image_gray)

        # Use agent's final heightmap if available, otherwise apply plan
        if result.final_hf is not None:
            modified_hf = result.final_hf
        else:
            modified_hf = apply_agent_plan(base_hf, raw_hf, result.plan)

        # Validate against base
        accepted, reason = validate_agent_modified_heightfield(base_hf, modified_hf)
        agent_hf = modified_hf if accepted else base_hf

        # Quantization pre-equalization: prevent agent operations from piling heights at one end
        import cv2 as _cv2
        agent_hf_u8 = (np.clip(agent_hf, 0, 1) * 255).astype(np.uint8)
        clahe = _cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
        agent_hf = clahe.apply(agent_hf_u8).astype(np.float32) / 255.0

        # Normalize for terrace quantization (expands range for more visible levels)
        terrace_hf = normalize_for_terrace_quantize(agent_hf, p_low=1.0, p_high=99.0, preserve_margin=0.02)

        # Pre-smoothing before terrace mesh: sigma scales inversely with n_steps
        # More steps → narrower per-step width → needs finer smoothing for continuity
        from scipy.ndimage import gaussian_filter as _gf
        pre_smooth_sigma = max((256 / tc.terrace_steps) * 0.3, 1.0)
        terrace_hf = _gf(terrace_hf, sigma=pre_smooth_sigma).astype(np.float32)
        terrace_hf = np.clip(terrace_hf, 0.02, 0.98)

        # Generate terrace mesh
        mesh, _report = heightfield_to_terrace_mesh(terrace_hf, tc)

        agent_dir = OUT / "agent_run"
        agent_dir.mkdir(parents=True, exist_ok=True)
        stl_path = agent_dir / "terrace.stl"
        save_stl(mesh, stl_path)

        np.save(str(agent_dir / "heightfield_blended.npy"), blended_hf)
        np.save(str(agent_dir / "heightfield_base.npy"), base_hf)
        np.save(str(agent_dir / "heightfield_modified.npy"), modified_hf)
        np.save(str(agent_dir / "heightfield_final.npy"), agent_hf)
        np.save(str(agent_dir / "heightfield_terrace_input.npy"), terrace_hf)

        # Save analysis
        from heightmap_analyzer import analyze_heightmap as _analyze
        result.analysis_after = _analyze(agent_hf)
        (agent_dir / "analysis_before.json").write_text(
            analysis_to_json(result.analysis_before)
        )
        (agent_dir / "analysis_after.json").write_text(
            analysis_to_json(result.analysis_after)
        )

        elapsed = time.time() - t0

        # Build info
        before_text = analysis_to_text(result.analysis_before)
        plan = result.plan

        log_lines = []
        for entry in result.conversation_log:
            for item in entry.get("content", []):
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        log_lines.append(item["text"][:200])
                    elif item.get("type") == "tool_use":
                        log_lines.append(f"**Tool**: `{item['name']}`({json.dumps(item.get('input', {}))[:100]})")

        stl_preview = _render_heightmap_3d(agent_hf, tc.physical_size_mm, tc.max_height_mm, title="Agent Terrace")

        info = (
            f"**Agent Transform done** ({elapsed:.1f}s)\n\n"
            f"- Intent: \"{user_intent}\"\n"
            f"- Model: {model}\n"
            f"- Iterations used: {result.iterations_used}\n"
            f"- Validation: {'ACCEPTED' if accepted else 'REJECTED (fallback to base)'}: {reason}\n"
            f"- STL saved: `{stl_path.relative_to(ROOT)}`\n\n"
            f"### Creative Plan\n"
            f"- Operations applied: {len(plan.operations)}\n"
            f"- Topology preserved: {plan.topology_preserved}\n"
            f"- Reasoning: {plan.reasoning}\n"
        )
        if plan.operations:
            info += "\n**Operations:**\n"
            for i, op in enumerate(plan.operations, 1):
                info += f"  {i}. {op.get('tool', 'unknown')}\n"
        info += (
            f"\n### Before Analysis\n{before_text}\n\n"
            f"### Evaluation\n{result.evaluation_notes}"
        )
        if log_lines:
            info += "\n\n### Agent Log\n" + "\n".join(f"- {l}" for l in log_lines[:20])

        return _arr_to_uint8(base_hf), _arr_to_uint8(agent_hf), stl_preview, info

    except Exception as e:
        raise gr.Error(f"Agent transform failed: {e}")


# ===== 7. Environment Check =================================================

def run_env_check():
    results = []

    # Python
    results.append(f"- Python: {sys.version.split()[0]}")

    # numpy
    results.append(f"- NumPy: {np.__version__}")

    # opencv
    try:
        import cv2
        results.append(f"- OpenCV: {cv2.__version__}")
    except ImportError:
        results.append("- OpenCV: **not installed**")

    # torch
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else "N/A"
        results.append(f"- PyTorch: {torch.__version__}")
        results.append(f"- CUDA available: {cuda}  |  GPU: {gpu_name}")
    except ImportError:
        results.append("- PyTorch: **not installed**")

    # diffusers
    try:
        import diffusers
        results.append(f"- Diffusers: {diffusers.__version__}")
    except ImportError:
        results.append("- Diffusers: **not installed**")

    # trimesh
    try:
        import trimesh
        results.append(f"- Trimesh: {trimesh.__version__}")
    except ImportError:
        results.append("- Trimesh: **not installed**")

    # PIL
    try:
        import PIL
        results.append(f"- Pillow: {PIL.__version__}")
    except ImportError:
        results.append("- Pillow: **not installed**")

    # matplotlib
    try:
        import matplotlib
        results.append(f"- Matplotlib: {matplotlib.__version__}")
    except ImportError:
        results.append("- Matplotlib: **not installed**")

    # scipy
    try:
        import scipy
        results.append(f"- SciPy: {scipy.__version__}")
    except ImportError:
        results.append("- SciPy: **not installed**")

    # gradio
    results.append(f"- Gradio: {gr.__version__}")

    return "**Environment Check Results**\n\n" + "\n".join(results)


# ===== test heightfield ====================================================

def _make_test_heightfield(size: int = 512) -> np.ndarray:
    """Generate a synthetic test heightfield (concentric + noise)."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv ** 2 + yv ** 2)
    hf = 0.5 + 0.3 * np.cos(r * 8 * np.pi) * np.exp(-r * 2)
    hf += np.random.default_rng(42).uniform(0, 0.05, hf.shape)
    return hf.astype(np.float32).clip(0, 1)


# ===== Build UI =============================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Tactile Geometry — Module Test", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Tactile Geometry Generation — Module Test Panel")
        gr.Markdown("Upload an image or load a .npy heightfield to test each pipeline module.")

        # ---- Tab 0: Environment Check ----
        with gr.Tab("0. Environment Check"):
            gr.Markdown("Click the button to check if all dependencies are installed.")
            btn_env = gr.Button("Check Environment", variant="primary")
            out_env = gr.Markdown()
            btn_env.click(run_env_check, outputs=out_env)

        # ---- Tab 1: Preprocessing ----
        with gr.Tab("1. Preprocessing"):
            gr.Markdown("Upload image -> Grayscale / Edge detection / High-frequency features")
            with gr.Row():
                inp_pre_img = gr.Image(label="Input Image", type="numpy")
            btn_pre = gr.Button("Run Preprocessing", variant="primary")
            with gr.Row():
                out_pre_gray = gr.Image(label="Grayscale")
                out_pre_edge = gr.Image(label="Edge Detection")
                out_pre_freq = gr.Image(label="High Frequency")
            out_pre_info = gr.Markdown()
            btn_pre.click(
                run_preprocessing, inputs=inp_pre_img,
                outputs=[out_pre_gray, out_pre_edge, out_pre_freq, out_pre_info],
            )

        # ---- Tab 2: Tactile Mapping ----
        with gr.Tab("2. Tactile Mapping"):
            gr.Markdown("Upload image -> Compute tactile descriptors (roughness / directionality / frequency)")
            inp_tac_img = gr.Image(label="Input Image", type="numpy")
            btn_tac = gr.Button("Run Tactile Mapping", variant="primary")
            out_tac = gr.Markdown()
            btn_tac.click(run_tactile_mapping, inputs=inp_tac_img, outputs=out_tac)

        # ---- Tab 2.5: Smart Crop ----
        with gr.Tab("2.5 Smart Crop"):
            gr.Markdown(
                "When texture details are too fine for the tool, auto or manual crop region selection.\n\n"
                "**Auto**: argmax of weighted (roughness + directionality + frequency) heatmap.\n"
                "**Manual**: use X/Y sliders to select position from heatmap."
            )
            inp_crop_img = gr.Image(label="Input Image", type="numpy")
            inp_crop_frac = gr.Slider(
                0.2, 1.0, value=0.5, step=0.05,
                label="Crop side / shorter image side",
            )
            with gr.Row():
                inp_crop_w_rough = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.1, label="Roughness weight",
                )
                inp_crop_w_dir = gr.Slider(
                    0.0, 2.0, value=0.5, step=0.1, label="Directionality weight",
                )
                inp_crop_w_freq = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.1, label="Frequency weight",
                )
            inp_crop_mode = gr.Radio(
                choices=["Auto", "Manual"], value="Auto", label="Crop positioning mode",
            )
            with gr.Row():
                inp_crop_mx = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Manual X center (normalized)",
                )
                inp_crop_my = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Manual Y center (normalized)",
                )
            btn_crop = gr.Button("Run", variant="primary")
            with gr.Row():
                out_crop_overlay = gr.Image(label="Feature heatmap + crop box")
                out_crop_image = gr.Image(label="Cropped result")
            out_crop_info = gr.Markdown()
            btn_crop.click(
                run_smart_crop,
                inputs=[inp_crop_img, inp_crop_frac,
                        inp_crop_w_rough, inp_crop_w_dir, inp_crop_w_freq,
                        inp_crop_mode, inp_crop_mx, inp_crop_my],
                outputs=[out_crop_overlay, out_crop_image, out_crop_info],
            )

        # ---- Tab 3: Diffusion ----
        with gr.Tab("3. Diffusion Pipeline"):
            gr.Markdown("Upload image -> Local trained model -> Generate heightfield\n\n"
                        "**Train model first**: `python src/training/train.py`")
            inp_diff_img = gr.Image(label="Input Image", type="numpy")
            inp_diff_steps = gr.Slider(10, 100, value=50, step=1, label="Sampling steps")
            btn_diff = gr.Button("Run Diffusion", variant="primary")
            out_diff_img = gr.Image(label="Raw heightfield")
            out_diff_info = gr.Markdown()
            btn_diff.click(
                run_diffusion,
                inputs=[inp_diff_img, inp_diff_steps],
                outputs=[out_diff_img, out_diff_info],
            )

        # ---- Tab 4: Saliency-Guided Geometry ----
        with gr.Tab("4. Saliency-Guided Terrace"):
            gr.Markdown(
                "Multi-scale FFT + structure tensor + height range → machinability weight.\n\n"
                "Weight = w_period x w_orientation x w_height_range. "
                "All three conditions must be satisfied for full preservation.\n\n"
                "- **Period**: multi-scale FFT (32/64/128px), energy-weighted average\n"
                "- **Orientation**: structure tensor coherence x alignment with toolpath\n"
                "- **Height range**: peak-valley difference; negligible features (< tolerance) are smoothed"
            )
            inp_sal_file = gr.File(label="Upload raw heightfield .npy", file_types=[".npy"])
            with gr.Row():
                inp_sal_size = gr.Number(label="Physical size (mm)", value=100.0)
                inp_sal_h = gr.Number(label="Max height (mm)", value=5.0)
                inp_sal_tr = gr.Number(label="Tool radius (mm)", value=3.0)
            with gr.Row():
                inp_sal_steps_hi = gr.Slider(
                    4, 24, value=12, step=1,
                    label="Terrace steps (high weight / machinable)",
                )
                inp_sal_steps_lo = gr.Slider(
                    2, 12, value=4, step=1,
                    label="Terrace steps (low weight / coarse)",
                )
            with gr.Row():
                inp_sal_thresh_hi = gr.Slider(
                    0.3, 0.9, value=0.65, step=0.05,
                    label="Weight threshold (high)",
                )
                inp_sal_thresh_lo = gr.Slider(
                    0.1, 0.6, value=0.30, step=0.05,
                    label="Weight threshold (low)",
                )
            gr.Markdown("**FFT analysis** (multi-scale: 32 / 64 / 128 px)")
            with gr.Row():
                inp_sal_fft_stride = gr.Slider(
                    4, 64, value=16, step=4,
                    label="FFT stride (px)",
                )
                inp_sal_fft_thresh = gr.Slider(
                    0.05, 0.5, value=0.20, step=0.05,
                    label="Energy threshold (non-periodic detection)",
                )
            gr.Markdown("**Structure tensor & height range**")
            with gr.Row():
                inp_sal_struct_sigma = gr.Slider(
                    4.0, 20.0, value=10.0, step=1.0,
                    label="Structure tensor sigma (px)",
                )
                inp_sal_tool_angle = gr.Slider(
                    0, 180, value=0, step=5,
                    label="Toolpath angle (deg, 0=horizontal)",
                )
                inp_sal_tolerance = gr.Slider(
                    0.01, 0.20, value=0.05, step=0.01,
                    label="Tool tolerance (mm)",
                )
            with gr.Row():
                inp_sal_blur = gr.Slider(
                    1.0, 20.0, value=8.0, step=1.0,
                    label="Weight blur sigma (px)",
                )
                inp_sal_mesh_res = gr.Slider(
                    64, 512, value=256, step=32,
                    label="Mesh resolution",
                )
            with gr.Row():
                inp_sal_stripe = gr.Slider(
                    0.0, 3.0, value=0.0, step=0.1,
                    label="Stripe expansion (× tool diameter, 0 = off)",
                    info="Expand dense parallel stripes to machinable spacing. "
                         "1.5 = target spacing 1.5× tool diameter.",
                )
                inp_sal_boost = gr.Slider(
                    1.0, 5.0, value=2.5, step=0.1,
                    label="Stripe boost strength",
                    info="Amplify stripe amplitude before terracing. "
                         "Auto-on when stripes detected (coherence > 0.06). "
                         "1.0 = off, 2.5 = default (2.5× stripe amplitude).",
                )
            btn_sal = gr.Button("Run Saliency Pipeline", variant="primary")
            with gr.Row():
                out_sal_hf = gr.Image(label="Machinable heightmap (2D)")
                out_sal_stl = gr.Image(label="STL 3D preview")
            out_sal_saliency = gr.Image(label="Machinability weight map")
            out_sal_info = gr.Markdown()
            btn_sal.click(
                run_saliency_geometry,
                inputs=[inp_sal_file,
                        inp_sal_size, inp_sal_h, inp_sal_tr,
                        inp_sal_steps_hi, inp_sal_steps_lo,
                        inp_sal_thresh_hi, inp_sal_thresh_lo,
                        inp_sal_mesh_res,
                        inp_sal_fft_stride, inp_sal_fft_thresh, inp_sal_blur,
                        inp_sal_struct_sigma, inp_sal_tool_angle, inp_sal_tolerance,
                        inp_sal_stripe, inp_sal_boost],
                outputs=[out_sal_hf, out_sal_stl, out_sal_saliency, out_sal_info],
            )

        # ---- Tab 5: Mockup ----
        with gr.Tab("5. Mockup (OBJ)"):
            gr.Markdown("Load .npy heightfield -> 256x256 low-res OBJ preview")
            inp_moc_file = gr.File(label="Upload .npy (optional)", file_types=[".npy"])
            with gr.Row():
                inp_moc_size = gr.Number(label="Physical size (mm)", value=100.0)
                inp_moc_h = gr.Number(label="Max height (mm)", value=10.0)
            btn_moc = gr.Button("Run Mockup", variant="primary")
            out_moc_img = gr.Image(label="Mockup Render")
            out_moc_info = gr.Markdown()
            btn_moc.click(
                run_mockup,
                inputs=[inp_moc_file, inp_moc_size, inp_moc_h],
                outputs=[out_moc_img, out_moc_info],
            )

        # ---- Tab 6: Agent Transform ----
        with gr.Tab("6. Agent Transform"):
            gr.Markdown(
                "LLM agent creatively transforms a heightmap based on your tactile intent.\n\n"
                "Pipeline: Image → Diffusion → Agent Transform → Terrace STL\n\n"
                "The agent uses 23 toolkit operations (frequency-aware, creative, patterns, ...) "
                "and iterates up to N times to match your description.\n\n"
                "**Requires**: API key for Anthropic or MiMo."
            )
            inp_ag_img = gr.Image(label="Input Image", type="numpy")
            inp_ag_intent = gr.Textbox(
                label="Tactile intent",
                value="rough organic surface with flowing ridges, preserve large-scale contours, convert fine texture to stepped stripes",
                lines=2,
            )
            with gr.Row():
                inp_ag_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="sk-ant-... or tp-...",
                    type="password",
                )
                inp_ag_model = gr.Dropdown(
                    choices=[
                        "claude-sonnet-4-6",
                        "claude-haiku-4-5-20251001",
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "mimo-v2-pro",
                        "mimo-v2-omni",
                        "mimo-v2-flash",
                    ],
                    value="claude-sonnet-4-6",
                    label="LLM Model",
                    allow_custom_value=True,
                )
            inp_ag_base_url = gr.Textbox(
                label="Base URL (leave empty for Anthropic, or set MiMo URL)",
                placeholder="https://token-plan-cn.xiaomimimo.com/anthropic",
            )
            with gr.Row():
                inp_ag_iters = gr.Slider(1, 8, value=5, step=1, label="Max iterations")
                inp_ag_diff_steps = gr.Slider(10, 100, value=50, step=1, label="Diffusion steps")
            with gr.Accordion("Terrace Config (optional)", open=False):
                gr.Markdown("Override terrace geometry parameters. Defaults work for most cases.")
                with gr.Row():
                    inp_ag_phys_size = gr.Number(label="Physical size (mm)", value=100.0)
                    inp_ag_max_h = gr.Number(label="Max height (mm)", value=20.0)
                with gr.Row():
                    inp_ag_tool_dia = gr.Number(label="Tool diameter (mm)", value=6.0)
                    inp_ag_steps = gr.Slider(2, 24, value=12, step=1, label="Terrace steps")
            btn_ag = gr.Button("Run Agent Transform", variant="primary")
            with gr.Row():
                out_ag_before = gr.Image(label="Before (raw)")
                out_ag_after = gr.Image(label="After (agent)")
            out_ag_stl = gr.Image(label="STL 3D preview")
            out_ag_info = gr.Markdown()
            btn_ag.click(
                run_agent_transform,
                inputs=[
                    inp_ag_img, inp_ag_intent,
                    inp_ag_api_key, inp_ag_base_url,
                    inp_ag_model, inp_ag_iters, inp_ag_diff_steps,
                    inp_ag_phys_size, inp_ag_max_h,
                    inp_ag_tool_dia, inp_ag_steps,
                ],
                outputs=[out_ag_before, out_ag_after, out_ag_stl, out_ag_info],
            )

    return app


# ===========================================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        inbrowser=True,
    )
