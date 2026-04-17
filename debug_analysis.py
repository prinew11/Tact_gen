"""
Diagnostic analysis: compare diffuse → pipeline output vs ground-truth heightmap.
"""
import sys
sys.path.insert(0, 'D:/homework/lund/CS_project/Tact_gen/src')

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DIFFUSE_PATH = "D:/homework/lund/CS_project/dataset/bark 99/bark 99_diffuse.png"
HEIGHT_PATH  = "D:/homework/lund/CS_project/dataset/bark 99/bark 99_height.png"
OUT_DIR = Path("D:/homework/lund/CS_project/Tact_gen/outputs/debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load images ──────────────────────────────────────────────────────────
diffuse_bgr = cv2.imread(DIFFUSE_PATH)
diffuse_rgb = cv2.cvtColor(diffuse_bgr, cv2.COLOR_BGR2RGB)
diffuse_gray = cv2.cvtColor(diffuse_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
height_gt = cv2.imread(HEIGHT_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

print(f"Diffuse shape : {diffuse_bgr.shape}")
print(f"Height GT shape: {height_gt.shape}")
print(f"Height GT range: [{height_gt.min():.3f}, {height_gt.max():.3f}]  "
      f"mean={height_gt.mean():.3f}  std={height_gt.std():.3f}")

# Resize to 512×512 for pipeline
SZ = (512, 512)
diffuse_gray_512 = cv2.resize(diffuse_gray, SZ, interpolation=cv2.INTER_AREA)
height_gt_512    = cv2.resize(height_gt,    SZ, interpolation=cv2.INTER_AREA)

# ── 2. Run preprocessing + tactile mapping ─────────────────────────────────
from preprocessing import extract_edges, extract_frequency, extract_orientation
from tactile_mapping import map_features

orientation = extract_orientation(diffuse_gray_512)
features = {
    "gray":      diffuse_gray_512,
    "edges":     extract_edges(diffuse_gray_512),
    "frequency": extract_frequency(diffuse_gray_512),
    **orientation,
}
descriptor = map_features(features)
print(f"\nTactileDescriptor:")
print(f"  roughness       = {descriptor.roughness:.4f}")
print(f"  directionality  = {descriptor.directionality:.4f}")
print(f"  frequency       = {descriptor.frequency:.4f}")

# ── 3. Baseline: compare diffuse-gray vs height-GT ─────────────────────────
corr_matrix = np.corrcoef(diffuse_gray_512.ravel(), height_gt_512.ravel())
r_diffuse_height = corr_matrix[0, 1]

mse_naive = float(np.mean((diffuse_gray_512 - height_gt_512) ** 2))
print(f"\nBaseline (diffuse_gray vs GT height):")
print(f"  Pearson r = {r_diffuse_height:.4f}")
print(f"  MSE       = {mse_naive:.6f}")
print(f"  RMSE      = {np.sqrt(mse_naive):.4f}")

# ── 4. Compute improved conditioning signal ─────────────────────────────────
#   Better approach: combine luminance with inverted frequency
#   (bright areas in diffuse often correlate with high-relief in bark)
freq_map    = features["frequency"]
orient_str  = features["orientation_strength"]

# Candidate 1: raw diffuse gray
cand1 = diffuse_gray_512

# Candidate 2: luminance-normalised (histogram equalisation)
cand2 = cv2.equalizeHist((diffuse_gray_512 * 255).astype(np.uint8)).astype(np.float32) / 255.0

# Candidate 3: frequency-weighted — boost structure edges
cand3 = np.clip(diffuse_gray_512 * 0.6 + freq_map * 0.4, 0, 1)

# Candidate 4: orientation-strength blended
cand4 = np.clip(diffuse_gray_512 * 0.7 + orient_str * 0.3, 0, 1)

candidates = {
    "diffuse_gray":      cand1,
    "hist_equalized":    cand2,
    "freq_weighted":     cand3,
    "orient_weighted":   cand4,
}

print(f"\nCorrelation vs GT height (each candidate conditioning):")
for name, cand in candidates.items():
    r = np.corrcoef(cand.ravel(), height_gt_512.ravel())[0, 1]
    mse = float(np.mean((cand - height_gt_512) ** 2))
    print(f"  {name:<22}  r={r:+.4f}  RMSE={np.sqrt(mse):.4f}")

# ── 5. Visualise ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Bark 99 — Conditioning signal analysis", fontsize=14)

titles = ["Diffuse (RGB)", "Diffuse Gray", "GT Height", "Hist-Equalized",
          "Freq-Weighted", "Orient-Weighted", "Edges", "Orientation Strength"]
imgs   = [diffuse_rgb, diffuse_gray_512, height_gt_512, cand2,
          cand3,       cand4,            features["edges"], orient_str]
cmaps  = ["none", "gray", "gray", "gray", "gray", "gray", "gray", "hot"]

for ax, title, img, cmap in zip(axes.ravel(), titles, imgs, cmaps):
    if cmap == "none":
        ax.imshow(cv2.resize(img, SZ))
    else:
        ax.imshow(cv2.resize(img, SZ), cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
save_path = OUT_DIR / "analysis.png"
plt.savefig(str(save_path), dpi=150)
plt.close()
print(f"\nAnalysis plot saved: {save_path}")

# ── 6. Print root-cause summary ───────────────────────────────────────────
print("\n" + "="*60)
print("ROOT CAUSE ANALYSIS")
print("="*60)
print(f"""
1. DOMAIN MISMATCH (主要原因)
   SD v1-5 是在自然图像上训练的生成模型，从未见过
   heightmap/displacement-map 的映射关系。它不理解
   "diffuse → height" 这个物理关系，只能生成视觉上
   合理但与真实高度场无关的灰度图。

2. CONDITIONING SIGNAL 弱 (次要原因)
   img2img 把 diffuse 当作"风格参考"而非"结构输入"。
   diffuse 含有颜色/光照信息，height 只关心几何起伏。
   Pearson r = {r_diffuse_height:+.4f}（diffuse 灰度与 GT 高度）
   → 两者本身相关性{'较强' if abs(r_diffuse_height) > 0.5 else '较弱'}。

3. STRENGTH 参数问题
   当前 strength=0.75 → 75% 噪声注入。
   对于需要保留结构的任务，建议 0.3–0.5。

4. 改进路径（已在代码中实现分析）:
   - 最佳条件信号: {'hist_equalized' if abs(np.corrcoef(cand2.ravel(), height_gt_512.ravel())[0,1]) > abs(r_diffuse_height) else 'diffuse_gray'}
   - 长期方案: 用 (diffuse, height) 配对数据 fine-tune
     ControlNet 或 InstructPix2Pix。
""")
