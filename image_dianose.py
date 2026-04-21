# scripts/diagnose_data.py
"""
跑完这个脚本你会知道：
1. CGAxis heightmap 的坡度分布（决定需要多少平滑）
2. 高度值域（决定 z-scale 怎么设）
3. 最小特征尺寸（决定对不对得上 6mm 刀具）
"""
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = Path("D:/homework/lund/CS_project/dataset_resize")

paths = sorted(DATA_ROOT.rglob("*_height.png"))[:50]  # 先看 50 张

max_slopes, height_ranges, min_feature_sizes = [], [], []

for p in paths:
    arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
    H, W = arr.shape

    # 坡度（像素空间，无物理单位）
    gy, gx = np.gradient(arr)
    slope = np.sqrt(gx**2 + gy**2)
    max_slopes.append(slope.max())

    # 高度范围
    height_ranges.append(arr.max() - arr.min())

    # 最小特征：找局部极值之间的平均间距（粗略估计）
    from scipy.ndimage import label, maximum_filter
    local_max = (arr == maximum_filter(arr, size=5))
    labeled, n = label(local_max)
    if n > 1:
        # 用特征密度估算最小间距（px）
        min_feature_sizes.append(np.sqrt(H * W / n))

print("=" * 50)
print(f"分析了 {len(paths)} 张 heightmap")
print()
print("坡度统计（像素坐标系，无物理单位）：")
print(f"  中位数: {np.median(max_slopes):.4f}")
print(f"  95th 百分位: {np.percentile(max_slopes, 95):.4f}")
print(f"  最大值: {max(max_slopes):.4f}")
print(f"  → tan(45°)=1.0，超过此值需要平滑")
print()
print("高度范围（归一化后）：")
print(f"  均值: {np.mean(height_ranges):.3f}")
print(f"  → 接近 1.0 表示用满了全部 8-bit 动态范围")
print()
if min_feature_sizes:
    print("最小特征间距（像素）：")
    print(f"  中位数: {np.median(min_feature_sizes):.1f} px")
    print(f"  → 你的 6mm 刀具对应多少像素？取决于加工尺寸")
    print(f"     若加工件 100mm×100mm，256px 图 = 0.39mm/px")
    print(f"     6mm 刀具 = 15px 最小特征，低于此值无法加工")

# 画坡度分布直方图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(max_slopes, bins=30, color='steelblue', edgecolor='white')
axes[0].axvline(1.0, color='red', linestyle='--', label='tan(45°) = 1.0')
axes[0].axvline(np.tan(np.deg2rad(30)), color='orange',
                linestyle='--', label='tan(30°) = 0.577')
axes[0].set_xlabel("Max slope per image")
axes[0].set_title("Slope distribution in training data")
axes[0].legend()

axes[1].hist(height_ranges, bins=30, color='teal', edgecolor='white')
axes[1].set_xlabel("Height range (normalized)")
axes[1].set_title("Height dynamic range")

plt.tight_layout()
Path("outputs").mkdir(exist_ok=True)
plt.savefig("outputs/data_diagnosis.png", dpi=120)
print("\n图表已保存: outputs/data_diagnosis.png")