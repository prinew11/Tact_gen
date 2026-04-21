"""
Gradio UI entry point: test each pipeline module independently.
Run:  python src/app.py
"""
from __future__ import annotations

import sys
import time
import traceback
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

def _status(ok: bool, msg: str) -> str:
    tag = "PASS" if ok else "FAIL"
    return f"**[{tag}]** {msg}"


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
        np.save(str(OUT / "heightfields" / "heightfield.npy"), hf)

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


# ===== 3.5 Fabrication Correction ==========================================

def run_fabrication_correction(heightfield_file, physical_size, max_height,
                               tool_radius, max_slope, material_hint):
    try:
        if heightfield_file is not None:
            hf = np.load(heightfield_file)
        else:
            default = OUT / "heightfields" / "heightfield.npy"
            if not default.exists():
                hf = _make_test_heightfield()
            else:
                hf = np.load(str(default))

        from fabrication_corrector import CorrectionConfig, correct_heightfield

        config = CorrectionConfig(
            physical_size_mm=float(physical_size),
            max_height_mm=float(max_height),
            tool_radius_mm=float(tool_radius),
            max_slope_deg=float(max_slope),
            resolution=hf.shape[0],
        )

        t0 = time.time()
        hf_corrected, report = correct_heightfield(hf, config)
        elapsed = time.time() - t0

        corr_path = OUT / "heightfields" / "heightfield_corrected.npy"
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(corr_path), hf_corrected)

        info = (
            f"**Fabrication Correction 完成** ({elapsed:.2f}s)\n\n"
            f"| 指标 | 修正前 | 修正后 |\n"
            f"|---|---|---|\n"
            f"| 最大坡度 | {report.original_max_slope_deg:.1f}° | {report.corrected_max_slope_deg:.1f}° |\n"
            f"| 坡度违规像素 | {report.slope_violations_before:,} | {report.slope_violations_after:,} |\n\n"
            f"- 形态学修改像素: {report.morph_pixels_changed:,}\n"
            f"- 传播扫描次数: {report.propagation_sweeps_used}\n"
            f"- 应用约束: {', '.join(report.constraints_applied)}\n"
            f"- 已保存: `{corr_path.relative_to(ROOT)}`"
        )

        filter_report_json = ""
        hf_display = hf_corrected

        if apply_machining_filter:
            import json as _json
            from machining_filter import (
                MachiningFilterConfig,
                filter_heightfield_for_machining,
                save_report_json as save_mf_report,
                save_heightfield as save_machinable,
            )
            mf_cfg = MachiningFilterConfig(
                physical_size_mm=float(physical_size),
                max_height_mm=float(max_height),
                tool_radius_mm=float(tool_radius),
                max_slope_deg=float(max_slope),
            )
            hf_machinable, mf_report = filter_heightfield_for_machining(hf_corrected, mf_cfg)
            mac_path = OUT / "heightfields" / "heightfield_machinable.npy"
            save_machinable(hf_machinable, mac_path)
            rpt_path = OUT / "heightfields" / "machining_filter_report.json"
            save_mf_report(mf_report, rpt_path)
            # Save geometry params so Tab 6 uses the exact same values
            cfg_path = OUT / "heightfields" / "heightfield_machinable_config.json"
            with open(cfg_path, "w") as _f:
                _json.dump({
                    "max_height_mm": float(max_height),
                    "physical_size_mm": float(physical_size),
                }, _f, indent=2)
            import dataclasses
            filter_report_json = _json.dumps(dataclasses.asdict(mf_report), indent=2)
            hf_display = hf_machinable
            info += (
                f"\n\n**Machining Filter 完成**\n"
                f"- 坡度: {mf_report.max_slope_deg_before:.1f}° → {mf_report.max_slope_deg_after:.1f}°\n"
                f"- 台阶数: {mf_report.terrace_steps_applied}\n"
                f"- Machinable 已保存: `{mac_path.relative_to(ROOT)}`"
            )

        return _arr_to_uint8(hf), _arr_to_uint8(hf_display), info, filter_report_json
    except Exception as e:
        raise gr.Error(f"Fabrication Correction 失败: {e}")


# ===== 4. Geometry (STL) ====================================================

def _load_best_heightfield(heightfield_file) -> tuple[np.ndarray, str]:
    """Load heightfield: uploaded > machinable > corrected > raw > synthetic."""
    if heightfield_file is not None:
        return np.load(heightfield_file), "uploaded"
    machinable = OUT / "heightfields" / "heightfield_machinable.npy"
    if machinable.exists():
        return np.load(str(machinable)), str(machinable.relative_to(ROOT))
    corrected = OUT / "heightfields" / "heightfield_corrected.npy"
    if corrected.exists():
        return np.load(str(corrected)), str(corrected.relative_to(ROOT))
    raw = OUT / "heightfields" / "heightfield.npy"
    if raw.exists():
        return np.load(str(raw)), str(raw.relative_to(ROOT))
    return _make_test_heightfield(), "synthetic test"


def run_geometry(heightfield_file, physical_size: float, max_height: float):
    try:
        hf, hf_source = _load_best_heightfield(heightfield_file)

        from geometry import GeometryConfig, heightfield_to_mesh, save_stl

        config = GeometryConfig(
            physical_size_mm=physical_size,
            max_height_mm=max_height,
        )
        t0 = time.time()
        mesh = heightfield_to_mesh(hf, config)
        elapsed = time.time() - t0

        stl_path = OUT / "stl_fabrication" / "tactile.stl"
        save_stl(mesh, stl_path)

        info = (
            f"**Geometry 模块测试通过**\n\n"
            f"- 输入来��: {hf_source}\n"
            f"- 输入高度场: {hf.shape}\n"
            f"- 顶点数: {len(mesh.vertices):,}\n"
            f"- 面数: {len(mesh.faces):,}\n"
            f"- 水密性: {mesh.is_watertight}\n"
            f"- 构建耗时: {elapsed:.2f}s\n"
            f"- STL 已保存: `{stl_path.relative_to(ROOT)}`"
        )
        return _arr_to_uint8(hf), info
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
        im = ax.imshow(zv, origin="lower", cmap="terrain")
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


# ===== 6. Fabrication Check =================================================

def run_fabrication(heightfield_file, hf_source_choice: str,
                    tool_radius: float, max_slope: float,
                    physical_size: float, max_height: float):
    try:
        import json as _json

        # Load heightfield based on source choice
        if hf_source_choice == "raw":
            raw_path = OUT / "heightfields" / "heightfield_raw.npy"
            hf = np.load(str(raw_path)) if raw_path.exists() else _make_test_heightfield()
            hf_source = "raw (heightfield_raw.npy)"
        elif hf_source_choice == "machinable":
            mac_path = OUT / "heightfields" / "heightfield_machinable.npy"
            hf = np.load(str(mac_path)) if mac_path.exists() else _make_test_heightfield()
            hf_source = "machinable (heightfield_machinable.npy)"
            # Override geometry params with the values stored when the filter ran
            cfg_path = OUT / "heightfields" / "heightfield_machinable_config.json"
            if cfg_path.exists():
                with open(cfg_path) as _f:
                    _mac_cfg = _json.load(_f)
                max_height = _mac_cfg["max_height_mm"]
                physical_size = _mac_cfg["physical_size_mm"]
        else:
            hf, hf_source = _load_best_heightfield(heightfield_file)

        # Load machining filter report if available
        filter_report_json = ""
        report_path = OUT / "heightfields" / "machining_filter_report.json"
        if report_path.exists():
            with open(report_path) as f:
                filter_report_json = _json.dumps(_json.load(f), indent=2)

        from geometry import GeometryConfig, heightfield_to_mesh
        from fabrication import FabricationConfig, check_mesh

        geo_cfg = GeometryConfig(
            physical_size_mm=physical_size,
            max_height_mm=max_height,
        )
        mesh = heightfield_to_mesh(hf, geo_cfg)
        config = FabricationConfig(
            tool_radius_mm=tool_radius,
            max_slope_deg=max_slope,
            physical_size_mm=physical_size,
            max_height_mm=max_height,
        )
        report = check_mesh(mesh, config)

        status = "PASS" if report.passes else "FAIL"
        issues_str = "\n".join(f"  - {i}" for i in report.issues) if report.issues else "  无"

        info = (
            f"**Fabrication 检查结果: {status}**\n\n"
            f"- 输入来源: {hf_source}\n\n"
            f"| 检查项 | 结果 |\n"
            f"|---|---|\n"
            f"| 水密性 | {report.watertight} |\n"
            f"| 面数 | {report.face_count:,} |\n"
            f"| 最大坡度 (仅顶面) | {report.max_slope_deg:.1f}° (限制 {max_slope}°) |\n"
            f"| 最小特征 | {report.min_feature_mm:.3f} mm "
            f"(刀具直径 {tool_radius * 2:.1f} mm) |\n"
            f"| GRBL 兼容 | {report.grbl_compatible} |\n\n"
            f"**问题列表:**\n{issues_str}"
        )
        return info, filter_report_json
    except Exception as e:
        raise gr.Error(f"Fabrication Check 失败: {e}")


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

        # ---- Tab 3.5: Fabrication Correction ----
        with gr.Tab("3.5 Fabrication Correction"):
            gr.Markdown("对高度场进行加工约束修正：坡度限制、最小特征尺寸、形态学滤波\n\n"
                        "基于 Memory Store (RAG) 检索加工知识并自适应修正。")
            inp_corr_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            with gr.Row():
                inp_corr_size = gr.Number(label="Physical size (mm)", value=50.0)
                inp_corr_h = gr.Number(label="Max height (mm)", value=5.0)
                inp_corr_tr = gr.Number(label="Tool radius (mm)", value=3.0)
            inp_corr_slope = gr.Slider(20, 60, value=45, step=1, label="最大坡度 (°)")
            inp_corr_mat = gr.Textbox(label="材料提示 (可选)", placeholder="e.g. wood, bark, concrete")
            btn_corr = gr.Button("运行 Fabrication Correction", variant="primary")
            with gr.Row():
                out_corr_before = gr.Image(label="修正前高度场")
                out_corr_after = gr.Image(label="修正后高度场 / Machinable")
            out_corr_info = gr.Markdown()
            out_corr_filter_json = gr.Code(
                label="Machining Filter Report (JSON)", language="json", visible=False
            )
            inp_corr_filter.change(
                fn=lambda x: gr.update(visible=x),
                inputs=inp_corr_filter,
                outputs=out_corr_filter_json,
            )
            btn_corr.click(
                run_fabrication_correction,
                inputs=[inp_corr_file, inp_corr_size, inp_corr_h, inp_corr_tr,
                        inp_corr_slope, inp_corr_mat],
                outputs=[out_corr_before, out_corr_after, out_corr_info],
            )

        # ---- Tab 4: Geometry ----
        with gr.Tab("4. Geometry (STL)"):
            gr.Markdown("加载 .npy 高度场 → 生成防水 STL\n\n"
                        "不上传文件则使用 `outputs/heightfields/heightfield.npy` 或自动生成测试数据。")
            inp_geo_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            with gr.Row():
                inp_geo_size = gr.Number(label="Physical size (mm)", value=50.0)
                inp_geo_h = gr.Number(label="Max height (mm)", value=5.0)
            btn_geo = gr.Button("运行 Geometry", variant="primary")
            out_geo_img = gr.Image(label="高度场预览")
            out_geo_info = gr.Markdown()
            btn_geo.click(
                run_geometry,
                inputs=[inp_geo_file, inp_geo_size, inp_geo_h],
                outputs=[out_geo_img, out_geo_info],
            )

        # ---- Tab 5: Mockup ----
        with gr.Tab("5. Mockup (OBJ)"):
            gr.Markdown("加载 .npy 高度场 → 256×256 低精度 OBJ 预览（Z×2）")
            inp_moc_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            with gr.Row():
                inp_moc_size = gr.Number(label="Physical size (mm)", value=50.0)
                inp_moc_h = gr.Number(label="Max height (mm)", value=5.0)
            btn_moc = gr.Button("运行 Mockup", variant="primary")
            out_moc_img = gr.Image(label="Mockup 渲染")
            out_moc_info = gr.Markdown()
            btn_moc.click(
                run_mockup,
                inputs=[inp_moc_file, inp_moc_size, inp_moc_h],
                outputs=[out_moc_img, out_moc_info],
            )

        # ---- Tab 6: Fabrication Check ----
        with gr.Tab("6. Fabrication Check"):
            gr.Markdown(
                "加载高度场 → 构建 mesh → 检查水密性/面数/坡度(仅顶面)/最小特征/GRBL兼容\n\n"
                "优先使用 machinable 高度场 (`heightfield_machinable.npy`)。\n"
                "**刀具直径 6 mm** — 小于 6 mm 的凹槽/沟道将被报告为问题。"
            )
            inp_fab_file = gr.File(label="上传 .npy (可选)", file_types=[".npy"])
            inp_fab_source = gr.Dropdown(
                choices=["auto", "raw", "machinable"],
                value="auto",
                label="Heightfield source (auto = best available)",
            )
            with gr.Row():
                inp_fab_tr = gr.Number(label="Tool radius (mm)", value=3.0)
                inp_fab_slope = gr.Number(label="Max slope (°)", value=45.0)
            with gr.Row():
                inp_fab_size = gr.Number(label="Physical size (mm)", value=50.0)
                inp_fab_h = gr.Number(label="Max height (mm)", value=5.0)
            btn_fab = gr.Button("运行 Fabrication Check", variant="primary")
            out_fab_info = gr.Markdown()
            out_fab_filter_json = gr.Code(
                label="Machining Filter Report (JSON)", language="json"
            )
            btn_fab.click(
                run_fabrication,
                inputs=[inp_fab_file, inp_fab_source, inp_fab_tr, inp_fab_slope,
                        inp_fab_size, inp_fab_h],
                outputs=[out_fab_info, out_fab_filter_json],
            )

    return app


# ===========================================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        inbrowser=True,
    )
