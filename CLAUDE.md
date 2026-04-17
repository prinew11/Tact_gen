# Tactile Geometry Generation Project

## Project Goal
Transform visual input (images) into fabrication-ready tactile heightfield geometry
using an img2img diffusion pipeline.

## Tech Stack
- Python 3.11 (in conda "tact"), PyTorch 2.x, Diffusers, NumPy, OpenCV, trimshe, scipy
- Local GPU (CUDA)
- Stable Diffusion img2img as core generative module

## Pipeline Stages
1. Visual preprocessing (grayscale, edge, frequency features)
2. Tactile feature mapping (roughness, directionality, frequency descriptors)
3. img2img diffusion generation → single-channel heightfield
4. Geometry conversion: heightfield → triangular mesh → STL
5. Fabrication check (slope, tool radius, watertightness)
6. Mockup Geometry → visualize model(.OBJ/.STL)
7. Fusion 360 CAM → 3D Parallel Toolpath → GRBL Post Processor → .gcode

## Mockup Module
- Purpose: Low-res preview mesh for visual confirmation before CAM
- Input: heightfield numpy array (512x512)
- Output: .obj file at 256x256 resolution with z-exaggeration (2x)
- Tool: trimesh + matplotlib for quick render
- NOT the fabrication mesh — that stays at full 512x512 resolution

## Fabrication Output Pipeline

### Stage 1: Python Output
- geometry.py → full-res STL (512x512, watertight, Z-up, flat bottom)
- mockup.py → preview OBJ (256x256, z_scale=2.0 for visual clarity)
- fabrication.py → Fusion compatibility check before export

### Stage 2: Fusion 360 CAM (Manual)
- Import STL into Fusion 360 Manufacture workspace
- Toolpath strategy: 3D Parallel (ball end mill, stepover 0.2-0.5mm)
- Post Processor: grbl.cps (GRBL 1.1 compatible)
- Output: separate .gcode files per tool

### Stage 3: GRBL Execution (Manual)
- G-code sender: Universal G-code Sender (UGS) or Candle
- GRBL firmware: 1.1+
- No tool change support — one tool per .gcode file


## Key Constraints
- Output must be machinable: max slope < 45°, min feature > tool_radius*2
- Heightfield resolution: 512x512 default
- Output format: STL or G-code compatible mesh
- STL face count: < 500,000 for Fusion CAM stability
- Tool radius: ball end mill diameter determines min feature size
- GRBL min feedrate: 45 mm/min (hard lower limit)
- Max slope for 3-axis machining: < 45° (no undercuts)

## Code Conventions
- All modules are independent Python files under src/
- Use dataclasses for descriptors and configs
- Write type hints everywhere
- Tests go in tests/ using pytest