# Tactile Geometry Generation Project

## Project Goal
Transform visual wood texture input into fabrication-ready 
stepped tactile heightmaps, conditioned on user tactile intent 
(soft / rough / hard / organic / ...).
Same input image produces different stepped heightmap variants
depending on user intent — AI as creator, not translator.

## Tech Stack
- Python 3.11 (conda "tact")
- PyTorch 2.x, Diffusers, scikit-image, scikit-learn
- CLIP (OpenCLIP) for semantic intent encoding
- NumPy, OpenCV, SciPy, trimesh
- Local GPU (CUDA)
- UMAP for feature space visualization

## Pipeline Stages

### Stage 1: Visual → Continuous Heightmap
- Input: wood texture image (bark / wood / floor)
- Model: existing img2img model (fixed, not retrained)
- Output: continuous grayscale heightmap (512×512, float32)
- Status: already working — not modified in this project

### Stage 2: Feature Space Analysis (one-time, pre-training)
- Compute GLCM features for all 500 heightmaps
  (Contrast, Homogeneity, Energy, Entropy, Correlation)
  computed at 4 directions (0°, 45°, 90°, 135°), averaged
- Compute geometric features per heightmap
  (height std, gradient magnitude mean, histogram shape)
- Dimensionality reduction: PCA / UMAP → 2D visualization
- Goal: confirm bark/wood/floor form navigable structure
  in feature space before any model training

### Stage 3: Intent–Feature Mapping
- Encode tactile intent words via CLIP
  ("soft", "rough", "hard", "grainy", "flowing", ...)
- Map CLIP vectors → target GLCM feature vectors
  using DTD dataset (5640 images, 47 perceptual attributes)
  as statistical bridge — no manual labeling required
- Output: for any user text input, a target feature vector
  that defines what the output heightmap should "feel like"

### Stage 4: Conditional VAE (CVAE) — core generative module
- Input: continuous heightmap + intent condition vector
- Encoder: heightmap → latent distribution (μ, σ)
- Decoder: latent vector + condition → stepped heightmap
- Loss = reconstruction loss
       + KL divergence
       + GLCM target loss (drives creative deviation)
- Quantization layer: differentiable, produces discrete
  stepped levels directly — not post-processed
- Training data: 500 heightmaps, no manual annotation needed
- User intent navigates the latent space → same image,
  different outputs depending on intent

### Stage 5: Machining Constraint Repair
- Tool: 6mm flat end mill
- Real constraint: internal concave corner radius ≥ 3mm
                   isolated ridge min width ≥ 6mm
                   (narrow steps ARE machinable if surrounding
                    space allows tool entry)
- Operation: morphological opening (erosion + dilation,
             3mm circular kernel) on each contour layer
- Island detection: merge features narrower than 6mm
  into adjacent layer
- Input: stepped heightmap
- Output: machinable stepped heightmap (same format)

### Stage 6: Geometry Conversion
- geometry.py → STL from stepped heightmap
  (512×512, watertight, Z-up, flat bottom)
  Each step = vertical wall + flat platform
  STL face count < 500,000
- mockup.py → preview OBJ (256×256, z_scale=2.0)
  for visual confirmation before CAM

### Stage 7: Fabrication (Manual)
- Import STL into Fusion 360 Manufacture workspace
- CAM strategy: 2D Contour per step layer
  (NOT 3D Parallel — stepped geometry = contour per level)
- Tool: 6mm flat end mill
- Post Processor: grbl.cps (GRBL 1.1 compatible)
- Output: one .gcode file per step level, or merged
- G-code sender: UGS or Candle
- GRBL firmware: 1.1+, no tool change support

## Key Constraints
- Tool: 6mm flat end mill (NOT ball end mill)
- Stepped heightmap: discrete levels, not continuous surface
- Internal concave radius ≥ 3mm (hard geometric constraint)
- Isolated feature minimum width ≥ 6mm
- Max wall height per step: determined by material and feeds
- Heightmap resolution: 512×512
- STL face count: < 500,000 for Fusion CAM stability

## Module Structure
src/
├── preprocessing.py        # 保留：视觉预处理
├── diffusion_pipeline.py   # 保留：原图→连续heightmap（不动）
├── tactile_mapping.py      # 改造：原来做什么？→ 整合为
│                           #   GLCM特征计算 + intent映射
├── feature_analysis.py     # 新增：UMAP可视化，one-time分析用
├── intent_mapping.py       # 新增：CLIP → 目标GLCM特征向量
├── terrace_geometry.py     # 改造：加入意图条件化量化逻辑
├── geometry.py             # 保留：台阶heightmap → STL
├── mockup.py               # 保留：预览OBJ
└── app.py                  # 保留：整合入口


tests/ (pytest)

## Code Conventions
- All modules independent Python files under src/
- Dataclasses for configs and descriptors
- Type hints everywhere
- Tests in tests/ using pytest
