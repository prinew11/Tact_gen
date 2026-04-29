"""
Gradio UI entry point: test each pipeline module independently.
Run:  python src/app.py
"""
from __future__ import annotations

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
        raise gr.Error("请先上传一张图片")
    try:
        import cv2
        from preprocessing import load_image_gray, extract_edges, extract_frequency

        # Gradio gives RGB uint8 (H, W, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
        edges = extract_edges(gray)
        freq = extract_frequency(gray)

        info = (
            f"**Preprocessing 模块测试通过**\n\n"
            f"- 输出尺寸: {gray.shape}\n"
            f"- Gray 范围: [{gray.min():.3f}, {gray.max():.3f}]\n"
            f"- Edges 非零像素: {(edges > 0).sum():,}\n"
            f"- Frequency 均值: {freq.mean():.4f}"
        )
        return _arr_to_uint8(gray), _arr_to_uint8(edges), _arr_to_uint8(freq), info
    except Exception as e:
        raise gr.Error(f"Preprocessing 失败: {e}")


# ===== 2. Tactile Mapping ===================================================

def run_tactile_mapping(image: np.ndarray | None):
    if image is None:
        raise gr.Error("请先上传一张图片")
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
            f"**Tactile Mapping 模块测试通过**\n\n"
            f"| 指标 | 值 |\n"
            f"|---|---|\n"
            f"| Roughness (粗糙度) | {desc.roughness:.4f} |\n"
            f"| Directionality (方向性) | {desc.directionality:.4f} |\n"
            f"| Frequency (频率) | {desc.frequency:.4f} |"
        )
        return info
    except Exception as e:
        raise gr.Error(f"Tactile Mapping 失败: {e}")


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
        raise gr.Error("请先上传一张图片")
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
            f"**Smart Crop v3 完成** ({mode})\n\n"
            f"- 输入尺寸: {w}×{h}\n"
            f"- 裁剪尺寸: {side}×{side} ({crop_fraction*100:.0f}% of min side)\n"
            f"- 裁剪中心: ({cx}, {cy})  归一化: ({cx/w:.3f}, {cy/h:.3f})\n"
            f"- 范围: x[{x0}:{x1}], y[{y0}:{y1}]\n\n"
            f"**权重（归一化）**: rough={wr:.2f}, dir={wd:.2f}, freq={wf:.2f}\n\n"
            f"**裁剪区域内平均**\n"
            f"- 粗糙度: {crop_R:.3f}\n"
            f"- 方向一致性: {crop_D:.3f}\n"
            f"- 频率（|Laplacian|）: {crop_F:.3f}\n"
            f"- 总分: {float(score[cy, cx]):.3f}\n\n"
            f"已保存: `{out_path.relative_to(ROOT)}`"
        )
        return overlay, cropped, info
    except Exception as e:
        raise gr.Error(f"Smart Crop 失败: {e}")


# ===== 3. Diffusion Pipeline ===============================================

def run_diffusion(image: np.ndarray | None, steps: int):
    if image is None:
        raise gr.Error("请先上传一张图片")
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
            f"**Diffusion 模块测试通过（使用本地训练模型）**\n",
            f"- Checkpoint: `{config.trained_model_path}`",
            f"- 设备: {config.device}",
            f"- 采样步数: {config.num_inference_steps}",
            f"- 耗时: {elapsed:.1f}s",
            f"- 高度场范围: [{hf.min():.3f}, {hf.max():.3f}]",
            f"- Raw 已保存: `{hf_raw_path.relative_to(ROOT)}`",
        ])
        return _arr_to_uint8(hf), info
    except FileNotFoundError as e:
        raise gr.Error(f"Diffusion 失败: {e}")
    except Exception as e:
        raise gr.Error(f"Diffusion 失败: {e}")


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
    Z = np.flipud(hf) * max_height + base_thickness

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


def run_geometry(heightfield_file, physical_size: float, max_height: float,
                 tool_radius: float,
                 terrace_mode: bool, terrace_steps: int,
                 mesh_resolution: int):
    """One-shot pipeline: raw heightfield → machinable heightmap → STL.

    Returns (machinable_hf_2d_preview, stl_3d_preview, info_markdown).
    """
    try:
        from terrace_geometry import (
            MachiningFilterConfig, filter_heightfield_for_machining,
            save_heightfield as save_machinable,
            save_report_json,
            TerraceConfig, preprocess_for_terrace,
            heightfield_to_terrace_mesh,
            save_stl as terrace_save_stl,
        )

        hf, hf_source = _load_best_heightfield(heightfield_file)
        t0 = time.time()

        # --- Step 1: Machining filter → machinable heightmap ---
        mf_cfg = MachiningFilterConfig(
            physical_size_mm=float(physical_size),
            max_height_mm=float(max_height),
            tool_radius_mm=float(tool_radius),
            terrace_steps=int(terrace_steps) if not terrace_mode else 1,
            terrace_mode=bool(terrace_mode),
        )
        hf_machinable, mf_report = filter_heightfield_for_machining(hf, mf_cfg)
        mac_path = OUT / "heightfields" / "heightfield_machinable.npy"
        save_machinable(hf_machinable, mac_path)
        save_report_json(mf_report, OUT / "heightfields" / "machining_filter_report.json")

        # --- Step 2: Build STL ---
        if terrace_mode:
            hf_for_mesh = preprocess_for_terrace(
                hf_machinable,
                tool_diameter_mm=tool_radius * 2.0,
                physical_size_mm=physical_size,
                target_resolution=int(mesh_resolution),
            )
            t_cfg = TerraceConfig(
                physical_size_mm=physical_size,
                max_height_mm=max_height,
                terrace_steps=int(terrace_steps),
                tool_diameter_mm=tool_radius * 2.0,
                mesh_resolution=int(mesh_resolution),
            )
            mesh, t_report = heightfield_to_terrace_mesh(hf_for_mesh, t_cfg)
            stl_path = OUT / "stl_fabrication" / "tactile_terrace.stl"
            terrace_save_stl(mesh, stl_path)
            base_thickness = t_cfg.base_thickness_mm
            mesh_info = (
                f"- Mesh 类型: terrace\n"
                f"- 台阶数: {t_cfg.terrace_steps}\n"
                f"- 网格分辨率: {mesh_resolution}×{mesh_resolution}\n"
                f"- 顶点数: {t_report.vertex_count:,}\n"
                f"- 面数: {t_report.face_count:,}\n"
                f"- 水密性: {t_report.watertight}"
            )
        else:
            from geometry import GeometryConfig, heightfield_to_mesh, save_stl
            g_cfg = GeometryConfig(
                physical_size_mm=physical_size,
                max_height_mm=max_height,
            )
            mesh = heightfield_to_mesh(hf_machinable, g_cfg)
            stl_path = OUT / "stl_fabrication" / "tactile.stl"
            save_stl(mesh, stl_path)
            base_thickness = g_cfg.base_thickness_mm
            mesh_info = (
                f"- Mesh 类型: smooth\n"
                f"- 顶点数: {len(mesh.vertices):,}\n"
                f"- 面数: {len(mesh.faces):,}\n"
                f"- 水密性: {mesh.is_watertight}"
            )
        elapsed = time.time() - t0

        # --- Previews ---
        hf_preview = _arr_to_uint8(hf_machinable)
        stl_preview = _render_heightmap_3d(
            hf_machinable, physical_size, max_height,
            base_thickness=base_thickness,
            title=f"STL preview ({stl_path.name})",
        )

        info = (
            f"**Pipeline 完成** ({elapsed:.2f}s)\n\n"
            f"- 输入来源: {hf_source}\n"
            f"- Machinable 已保存: `{mac_path.relative_to(ROOT)}`\n"
            f"- STL 已保存: `{stl_path.relative_to(ROOT)}`\n\n"
            f"**Machining Filter**\n"
            f"- 台阶数: {mf_report.terrace_steps_applied}\n"
            f"- Pixel size: {mf_report.pixel_size_mm:.3f} mm\n"
            f"- Min feature: {mf_report.min_feature_target_mm:.1f} mm\n\n"
            f"**Mesh**\n{mesh_info}"
        )
        if mf_report.issues:
            info += "\n\n**Issues:**\n" + "\n".join(f"- {i}" for i in mf_report.issues)
        return hf_preview, stl_preview, info
    except Exception as e:
        raise gr.Error(f"Geometry 失败: {e}")


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
            f"**Mockup 模块测试通过**\n\n"
            f"- 输出分辨率: 256×256\n"
            f"- Z 放大倍数: 2.0×\n"
            f"- 构建耗时: {elapsed:.2f}s\n"
            f"- OBJ 已保存: `{obj_path.relative_to(ROOT)}`"
        )
        return plot_img, info
    except Exception as e:
        raise gr.Error(f"Mockup 失败: {e}")


# ===== 7. 环境检查 ==========================================================

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
        results.append("- OpenCV: **未安装**")

    # torch
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else "无"
        results.append(f"- PyTorch: {torch.__version__}")
        results.append(f"- CUDA 可用: {cuda}  |  GPU: {gpu_name}")
    except ImportError:
        results.append("- PyTorch: **未安装**")

    # diffusers
    try:
        import diffusers
        results.append(f"- Diffusers: {diffusers.__version__}")
    except ImportError:
        results.append("- Diffusers: **未安装**")

    # trimesh
    try:
        import trimesh
        results.append(f"- Trimesh: {trimesh.__version__}")
    except ImportError:
        results.append("- Trimesh: **未安装**")

    # PIL
    try:
        import PIL
        results.append(f"- Pillow: {PIL.__version__}")
    except ImportError:
        results.append("- Pillow: **未安装**")

    # matplotlib
    try:
        import matplotlib
        results.append(f"- Matplotlib: {matplotlib.__version__}")
    except ImportError:
        results.append("- Matplotlib: **未安装**")

    # scipy
    try:
        import scipy
        results.append(f"- SciPy: {scipy.__version__}")
    except ImportError:
        results.append("- SciPy: **未安装**")

    # gradio
    results.append(f"- Gradio: {gr.__version__}")

    return "**环境检查结果**\n\n" + "\n".join(results)


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
    with gr.Blocks(title="Tactile Geometry — 模块测试", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Tactile Geometry Generation — 模块测试面板")
        gr.Markdown("上传图片或加载 .npy 高度场，逐模块测试 pipeline 是否正常运行。")

        # ---- Tab 0: 环境检查 ----
        with gr.Tab("0. 环境检查"):
            gr.Markdown("点击按钮检查所有依赖库是否已安装。")
            btn_env = gr.Button("检查环境", variant="primary")
            out_env = gr.Markdown()
            btn_env.click(run_env_check, outputs=out_env)

        # ---- Tab 1: Preprocessing ----
        with gr.Tab("1. Preprocessing"):
            gr.Markdown("上传图片 → 灰度图 / 边缘检测 / 高频特征")
            with gr.Row():
                inp_pre_img = gr.Image(label="输入图片", type="numpy")
            btn_pre = gr.Button("运行 Preprocessing", variant="primary")
            with gr.Row():
                out_pre_gray = gr.Image(label="灰度图")
                out_pre_edge = gr.Image(label="边缘检测")
                out_pre_freq = gr.Image(label="高频特征")
            out_pre_info = gr.Markdown()
            btn_pre.click(
                run_preprocessing, inputs=inp_pre_img,
                outputs=[out_pre_gray, out_pre_edge, out_pre_freq, out_pre_info],
            )

        # ---- Tab 2: Tactile Mapping ----
        with gr.Tab("2. Tactile Mapping"):
            gr.Markdown("上传图片 → 计算触觉描述符（粗糙度/方向性/频率）")
            inp_tac_img = gr.Image(label="输入图片", type="numpy")
            btn_tac = gr.Button("运行 Tactile Mapping", variant="primary")
            out_tac = gr.Markdown()
            btn_tac.click(run_tactile_mapping, inputs=inp_tac_img, outputs=out_tac)

        # ---- Tab 2.5: Smart Crop ----
        with gr.Tab("2.5 Smart Crop"):
            gr.Markdown(
                "图片细节太密、刀加工不出来时，自动或手动定位裁剪区域。\n\n"
                "**Auto**：argmax of (粗糙度 + 方向性 + 频率) 加权热力图。\n"
                "**Manual**：用 X / Y 滑块对照热力图手动选位置。"
            )
            inp_crop_img = gr.Image(label="输入图片", type="numpy")
            inp_crop_frac = gr.Slider(
                0.2, 1.0, value=0.5, step=0.05,
                label="裁剪边长 / 原图较短边",
            )
            with gr.Row():
                inp_crop_w_rough = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.1, label="粗糙度权重",
                )
                inp_crop_w_dir = gr.Slider(
                    0.0, 2.0, value=0.5, step=0.1, label="方向性权重",
                )
                inp_crop_w_freq = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.1, label="频率权重",
                )
            inp_crop_mode = gr.Radio(
                choices=["Auto", "Manual"], value="Auto", label="裁剪定位模式",
            )
            with gr.Row():
                inp_crop_mx = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="手动 X 中心 (归一化)",
                )
                inp_crop_my = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="手动 Y 中心 (归一化)",
                )
            btn_crop = gr.Button("运行", variant="primary")
            with gr.Row():
                out_crop_overlay = gr.Image(label="特征热力图 + 裁剪框")
                out_crop_image = gr.Image(label="裁剪结果")
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
            gr.Markdown("上传图片 → 本地训练模型 → 生成高度场\n\n"
                        "**需要先训练模型**：`python src/training/train.py`")
            inp_diff_img = gr.Image(label="输入图片", type="numpy")
            inp_diff_steps = gr.Slider(10, 100, value=50, step=1, label="采样步数")
            btn_diff = gr.Button("运行 Diffusion", variant="primary")
            out_diff_img = gr.Image(label="Raw heightfield")
            out_diff_info = gr.Markdown()
            btn_diff.click(
                run_diffusion,
                inputs=[inp_diff_img, inp_diff_steps],
                outputs=[out_diff_img, out_diff_info],
            )

        # ---- Tab 4: Geometry (machining filter + STL) ----
        with gr.Tab("4. Geometry (Machinable + STL)"):
            gr.Markdown(
                "一次跑完：machining filter 生成可加工 heightmap，再构建 STL。"
                "两种输出都会预览 + 保存到磁盘。"
            )
            inp_geo_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            with gr.Row():
                inp_geo_size = gr.Number(label="Physical size (mm)", value=100.0)
                inp_geo_h = gr.Number(label="Max height (mm)", value=10.0)
                inp_geo_tr = gr.Number(label="Tool radius (mm)", value=3.0)
            with gr.Row():
                inp_geo_steps = gr.Slider(
                    0, 50, value=5, step=1,
                    label="Terrace 台阶数（0 = auto，smooth 模式下忽略）",
                )
                inp_geo_res = gr.Slider(
                    64, 512, value=256, step=32,
                    label="Terrace mesh 分辨率（仅 terrace mode）",
                )
            inp_geo_terrace = gr.Checkbox(
                label="Terrace mode（输出锐利台阶 STL）",
                value=False,
            )
            btn_geo = gr.Button("运行", variant="primary")
            with gr.Row():
                out_geo_hf = gr.Image(label="Machinable heightmap (2D)")
                out_geo_stl = gr.Image(label="STL 3D 预览")
            out_geo_info = gr.Markdown()
            btn_geo.click(
                run_geometry,
                inputs=[inp_geo_file, inp_geo_size, inp_geo_h,
                        inp_geo_tr,
                        inp_geo_terrace, inp_geo_steps, inp_geo_res],
                outputs=[out_geo_hf, out_geo_stl, out_geo_info],
            )

        # ---- Tab 5: Mockup ----
        with gr.Tab("5. Mockup (OBJ)"):
            gr.Markdown("加载 .npy 高度场 → 256*256 低精度 OBJ 预览)")
            inp_moc_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            with gr.Row():
                inp_moc_size = gr.Number(label="Physical size (mm)", value=100.0)
                inp_moc_h = gr.Number(label="Max height (mm)", value=10.0)
            btn_moc = gr.Button("运行 Mockup", variant="primary")
            out_moc_img = gr.Image(label="Mockup 渲染")
            out_moc_info = gr.Markdown()
            btn_moc.click(
                run_mockup,
                inputs=[inp_moc_file, inp_moc_size, inp_moc_h],
                outputs=[out_moc_img, out_moc_info],
            )

    return app


# ===========================================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        inbrowser=True,
    )
