"""
Heightfield diagnostic visualizer.
Shows raw hf, detrended versions, stripe analysis, and simulates what
the pipeline does to the stripes.

Run: python visualize_heightfield.py [path/to/file.npy]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion

# ---------------------------------------------------------------------------
NPY_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else \
           Path(__file__).parent / "outputs" / "heightfields" / "heightfield_raw.npy"

if not NPY_PATH.exists():
    sys.exit(f"File not found: {NPY_PATH}")

hf_raw = np.load(str(NPY_PATH)).astype(np.float32)
if hf_raw.ndim != 2:
    sys.exit(f"Expected 2-D array, got shape {hf_raw.shape}")

# ---------------------------------------------------------------------------
# Pipeline parameters (must match run_saliency_pipeline defaults)
PHYSICAL_SIZE_MM = 150.0
TOOL_DIAMETER_MM = 6.0
TERRACE_STEPS    = 12
# ---------------------------------------------------------------------------

H, W = hf_raw.shape
pixel_size_mm = PHYSICAL_SIZE_MM / max(H - 1, 1)
tool_radius_px = (TOOL_DIAMETER_MM / 2) / pixel_size_mm

print(f"Loaded : {NPY_PATH}")
print(f"Shape  : {hf_raw.shape}  pixel_size={pixel_size_mm:.3f} mm/px")
print(f"Raw    : min={hf_raw.min():.4f}  max={hf_raw.max():.4f}  "
      f"range={hf_raw.max()-hf_raw.min():.4f}")
print(f"Tool   : diameter={TOOL_DIAMETER_MM}mm  radius_px={tool_radius_px:.1f}px")


# ── helpers ──────────────────────────────────────────────────────────────────

def _clahe(x, clip=2.0, grid=8):
    u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return c.apply(u8).astype(np.float32) / 255.0


def _structure_tensor(x, sigma=8):
    """Returns (angle_map[gradient dir], coherence [0,1])."""
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = gaussian_filter(gx * gx, sigma)
    Jyy = gaussian_filter(gy * gy, sigma)
    Jxy = gaussian_filter(gx * gy, sigma)
    angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    disc = np.sqrt(np.maximum((Jxx - Jyy) ** 2 * 0.25 + Jxy ** 2, 0.0))
    l1 = 0.5 * (Jxx + Jyy) + disc
    l2 = 0.5 * (Jxx + Jyy) - disc
    coh = (l1 - l2) / (l1 + l2 + 1e-8)
    return angle, np.clip(coh, 0, 1).astype(np.float32)


def _morph_open(x, radius_px):
    """Isotropic morphological opening (what the pipeline does by default)."""
    r = max(int(np.ceil(radius_px)), 2)
    ker = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
    inv = (1.0 - x).astype(np.float32)
    opened = grey_dilation(grey_erosion(inv, footprint=ker), footprint=ker)
    return np.clip(1.0 - opened, 0, 1).astype(np.float32)


def _quantize(x, n):
    q = np.round(np.clip(x, 0, 1) * (n - 1)) / (n - 1)
    return q.astype(np.float32)


# ── Pipeline simulation ───────────────────────────────────────────────────────

# Step 1: percentile normalization
p2, p98 = np.percentile(hf_raw, 2), np.percentile(hf_raw, 98)
hf_p = np.clip((hf_raw - p2) / (p98 - p2 + 1e-8), 0, 1).astype(np.float32)

# Step 2: detrend (sigma=150px, same as run_saliency_pipeline)
bg = gaussian_filter(hf_p, sigma=150)
hf_dt = hf_p - bg
hf_dt = (hf_dt - hf_dt.min()) / (hf_dt.max() - hf_dt.min() + 1e-8)
hf_dt = hf_dt.astype(np.float32)

# Step 3: CLAHE blend (same as pipeline: 0.4 CLAHE + 0.6 original)
hf_clahe = _clahe(hf_dt, clip=1.5, grid=4)
hf_after_clahe = 0.4 * hf_clahe + 0.6 * hf_dt

# Step 4: morphological opening (isotropic, pipeline default with no protection)
hf_after_morph = _morph_open(hf_after_clahe, tool_radius_px)

# Step 5: quantize (simulate terrace output)
hf_terrace_raw  = _quantize(hf_after_clahe, TERRACE_STEPS)
hf_terrace_morph = _quantize(hf_after_morph, TERRACE_STEPS)

# Structure tensor on detrended (to show stripe coherence / direction)
angle_map, coh = _structure_tensor(hf_dt, sigma=max(min(H, W) // 32, 3))

print(f"\nDetrended: min={hf_dt.min():.4f}  max={hf_dt.max():.4f}  "
      f"range={hf_dt.max()-hf_dt.min():.4f}  std={hf_dt.std():.4f}")
print(f"Stripe coherence (detrended): mean={coh.mean():.3f}  max={coh.max():.3f}")
step_h = (hf_dt.max() - hf_dt.min()) / TERRACE_STEPS
print(f"Terrace step height (detrended range / {TERRACE_STEPS} steps) = {step_h:.4f}")
# estimate stripe amplitude from local std in cross-stripe direction
mid_row = hf_dt[H // 2, :]
amplitudes = np.abs(mid_row - gaussian_filter(mid_row, sigma=20))
stripe_amp = float(amplitudes.mean()) * 2  # rough peak-to-peak
print(f"Approx stripe amplitude (row {H//2}): {stripe_amp:.4f}  "
      f"≈ {stripe_amp / step_h:.1f} terrace steps")

# ── Figure ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 11), facecolor="#12121e")
fig.suptitle(
    f"Heightfield Pipeline Diagnostic  —  {NPY_PATH.name}\n"
    f"pixel={pixel_size_mm:.3f}mm  tool_diam={TOOL_DIAMETER_MM}mm  "
    f"tool_radius_px={tool_radius_px:.1f}px  steps={TERRACE_STEPS}",
    color="white", fontsize=12, y=0.99,
)

gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.30,
              left=0.04, right=0.98, top=0.94, bottom=0.05)
ax_kw = dict(facecolor="#0a0a18")

CMAP_HF   = "gray"
CMAP_HEAT = "inferno"
CMAP_COH  = "plasma"
CMAP_TERR = "terrain"

# Row 0: pipeline stages (heightfield appearance)
panels_top = [
    (hf_p,           CMAP_HF,   "1 · Normalized raw"),
    (hf_dt,          CMAP_HEAT, "2 · Detrended (σ=150px)"),
    (hf_after_clahe, CMAP_HF,   "3 · +CLAHE blend (0.4/0.6)"),
    (hf_after_morph, CMAP_HF,   f"4 · +Morph open (r={tool_radius_px:.0f}px)\n"
                                 f"← THIS destroys stripes"),
]
for col, (data, cmap, title) in enumerate(panels_top):
    ax = fig.add_subplot(gs[0, col], **ax_kw)
    im = ax.imshow(data, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# Row 1: terrace simulation + coherence + histogram
# [0] Terrace WITHOUT morph
ax = fig.add_subplot(gs[1, 0], **ax_kw)
im = ax.imshow(hf_terrace_raw, cmap=CMAP_TERR, origin="upper")
ax.set_title(f"Terrace ({TERRACE_STEPS} steps) — NO morph\n← stripes visible", color="lime", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# [1] Terrace WITH morph (current output)
ax = fig.add_subplot(gs[1, 1], **ax_kw)
im = ax.imshow(hf_terrace_morph, cmap=CMAP_TERR, origin="upper")
ax.set_title(f"Terrace ({TERRACE_STEPS} steps) — WITH morph\n← stripes gone (current output)", color="tomato", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# [2] Coherence map
ax = fig.add_subplot(gs[1, 2], **ax_kw)
im = ax.imshow(coh, cmap=CMAP_COH, vmin=0, vmax=1, origin="upper")
ax.set_title(f"Stripe coherence (mean={coh.mean():.3f})\nBright = clear stripe direction", color="white", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# [3] Histogram comparison
ax = fig.add_subplot(gs[1, 3], **ax_kw)
ax.hist(hf_raw.ravel(), bins=80, color="#4fc3f7", alpha=0.6, label="raw", density=True)
ax.hist(hf_dt.ravel(),  bins=80, color="#ff8a65", alpha=0.6, label="detrended", density=True)
ax.set_title("Height distributions\nraw vs detrended", color="white", fontsize=9)
ax.set_xlabel("Height value", color="white", fontsize=8)
ax.legend(fontsize=8, labelcolor="white", facecolor="#111")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

# Row 2: Cross-sections to show stripe amplitude
mid_row = H // 2
mid_col = W // 2
xs = np.arange(W)
ys = np.arange(H)

# [0] Raw cross-sections
ax = fig.add_subplot(gs[2, 0], **ax_kw)
ax.plot(xs, hf_p[mid_row, :], color="#4fc3f7", lw=0.8, label=f"row {mid_row} (raw norm.)")
ax.plot(xs, hf_dt[mid_row, :], color="#ff8a65", lw=0.8, label=f"row {mid_row} (detrended)")
ax.set_title("Horizontal cross-section\nraw vs detrended", color="white", fontsize=9)
ax.set_xlabel("Pixel (X)", color="white", fontsize=8)
ax.legend(fontsize=7, labelcolor="white", facecolor="#111")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

# [1] Right-half cross-section (stripe region) zoomed
ax = fig.add_subplot(gs[2, 1], **ax_kw)
x_start = W // 2
strip = hf_dt[mid_row, x_start:]
bg_strip = gaussian_filter(strip, sigma=30)
residual = strip - bg_strip
ax.plot(residual, color="#a5d6a7", lw=0.9, label="detrended stripe residual")
ax.axhline(0, color="white", lw=0.5, ls="--")
amp = float(np.abs(residual).mean()) * 2
step = (hf_dt.max() - hf_dt.min()) / TERRACE_STEPS
ax.set_title(f"Right-half stripe residual (row {mid_row})\n"
             f"amp≈{amp:.4f}  step_h≈{step:.4f}  ratio≈{amp/step:.1f}×", color="white", fontsize=9)
ax.set_xlabel(f"Pixel (X from {x_start})", color="white", fontsize=8)
ax.legend(fontsize=7, labelcolor="white", facecolor="#111")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

# [2] Before/after morph cross-section
ax = fig.add_subplot(gs[2, 2], **ax_kw)
ax.plot(xs, hf_after_clahe[mid_row, :], color="#80cbc4", lw=0.8, alpha=0.9, label="after CLAHE")
ax.plot(xs, hf_after_morph[mid_row, :], color="#ef9a9a", lw=0.8, alpha=0.9, label="after morph")
ax.set_title(f"Morph effect on row {mid_row}\nCLAHE vs after opening (r={tool_radius_px:.0f}px)", color="white", fontsize=9)
ax.set_xlabel("Pixel (X)", color="white", fontsize=8)
ax.legend(fontsize=7, labelcolor="white", facecolor="#111")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

# [3] Stripe period estimate (FFT of right-half detrended row)
ax = fig.add_subplot(gs[2, 3], **ax_kw)
row_right = hf_dt[mid_row, W // 2:]
row_right = row_right - row_right.mean()
fft_mag = np.abs(np.fft.rfft(row_right))
freqs = np.fft.rfftfreq(len(row_right))
periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), 0)
# Only search for stripe peak in 5–80px period range (exclude background trend)
valid_mask = (freqs >= 1.0 / 80) & (freqs <= 0.5)
valid_indices = np.where(valid_mask)[0]
if len(valid_indices) > 0:
    peak_idx = int(valid_indices[np.argmax(fft_mag[valid_indices])])
else:
    peak_idx = int(np.argmax(fft_mag[1:]) + 1)
ax.plot(periods[1:], fft_mag[1:], color="#ce93d8", lw=0.9)
peak_period_px = float(periods[peak_idx])
peak_period_mm = peak_period_px * pixel_size_mm
ax.axvline(peak_period_px, color="yellow", lw=1, ls="--", label=f"dominant {peak_period_px:.0f}px={peak_period_mm:.1f}mm")
ax.axvline(tool_radius_px * 2, color="tomato", lw=1, ls="--", label=f"tool diam {TOOL_DIAMETER_MM}mm")
ax.set_xlim(0, min(100, len(row_right) // 2))
ax.set_title(f"Stripe period FFT (right half, row {mid_row})\nPeak≈{peak_period_px:.0f}px={peak_period_mm:.1f}mm", color="white", fontsize=9)
ax.set_xlabel("Period (pixels)", color="white", fontsize=8)
ax.legend(fontsize=7, labelcolor="white", facecolor="#111")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("#444")

for ax in fig.axes:
    ax.tick_params(colors="white", labelsize=7)

print(f"\nDominant stripe period: {peak_period_px:.0f}px = {peak_period_mm:.1f}mm")
print(f"Tool diameter: {TOOL_DIAMETER_MM}mm = {tool_radius_px*2:.0f}px")
print(f"Stripe period vs tool: {peak_period_mm/TOOL_DIAMETER_MM:.2f}×  "
      f"({'MACHINABLE' if peak_period_mm >= TOOL_DIAMETER_MM else 'TOO NARROW for tool'})")

plt.show()
