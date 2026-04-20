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

## Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
