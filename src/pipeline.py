"""
Unified pipeline: Image -> Raw Heightmap -> Agent Transform -> Machinable STL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from diffusion_pipeline import generate_heightfield, DiffusionConfig
from agent_planner import run_agent_pipeline, AgentConfig, AgentResult
from heightmap_analyzer import analysis_to_json
from terrace_geometry import (
    preprocess_for_terrace,
    heightfield_to_terrace_mesh,
    TerraceConfig,
    save_stl,
)


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    image_path: str
    user_intent: str
    output_dir: str = "outputs/agent_run"
    diffusion_steps: int = 50
    diffusion_seed: int = 20
    agent_model: str = "claude-sonnet-4-20250514"
    agent_max_iterations: int = 5
    physical_size_mm: float = 50.0
    max_height_mm: float = 5.0
    tool_diameter_mm: float = 6.0
    terrace_steps: int = 0


@dataclass
class PipelineResult:
    raw_heightmap: np.ndarray
    agent_heightmap: np.ndarray
    machinable_heightmap: np.ndarray
    stl_path: Path
    agent_result: AgentResult


def run_full_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the complete pipeline from image to STL."""
    out_dir = ROOT / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(config.image_path).convert("RGB"))
    diff_config = DiffusionConfig(
        num_inference_steps=config.diffusion_steps,
        seed=config.diffusion_seed,
    )
    raw_hf = generate_heightfield(img, diff_config)
    np.save(out_dir / "heightfield_raw.npy", raw_hf)

    agent_config = AgentConfig(
        model=config.agent_model,
        max_iterations=config.agent_max_iterations,
    )
    agent_result = run_agent_pipeline(raw_hf, config.user_intent, agent_config)
    np.save(out_dir / "heightfield_agent.npy", agent_result.heightmap)

    machinable_hf = preprocess_for_terrace(
        agent_result.heightmap,
        tool_diameter_mm=config.tool_diameter_mm,
        physical_size_mm=config.physical_size_mm,
    )
    np.save(out_dir / "heightfield_machinable.npy", machinable_hf)

    tc = TerraceConfig(
        physical_size_mm=config.physical_size_mm,
        max_height_mm=config.max_height_mm,
        tool_diameter_mm=config.tool_diameter_mm,
        terrace_steps=config.terrace_steps if config.terrace_steps > 0 else 5,
    )
    mesh, report = heightfield_to_terrace_mesh(machinable_hf, tc)
    stl_path = out_dir / "terrace.stl"
    save_stl(mesh, stl_path)

    (out_dir / "analysis_before.json").write_text(
        analysis_to_json(agent_result.analysis_before)
    )
    (out_dir / "analysis_after.json").write_text(
        analysis_to_json(agent_result.analysis_after)
    )

    return PipelineResult(
        raw_heightmap=raw_hf,
        agent_heightmap=agent_result.heightmap,
        machinable_heightmap=machinable_hf,
        stl_path=stl_path,
        agent_result=agent_result,
    )


def run_agent_only(
    heightmap_path: str,
    user_intent: str,
    output_dir: str = "outputs/agent_run",
) -> PipelineResult:
    """Run only the agent transformation on an existing heightmap."""
    out_dir = ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_hf = np.load(heightmap_path).astype(np.float32)
    agent_result = run_agent_pipeline(raw_hf, user_intent)
    np.save(out_dir / "heightfield_agent.npy", agent_result.heightmap)

    machinable_hf = preprocess_for_terrace(agent_result.heightmap)
    np.save(out_dir / "heightfield_machinable.npy", machinable_hf)

    tc = TerraceConfig()
    mesh, report = heightfield_to_terrace_mesh(machinable_hf, tc)
    stl_path = out_dir / "terrace.stl"
    save_stl(mesh, stl_path)

    return PipelineResult(
        raw_heightmap=raw_hf,
        agent_heightmap=agent_result.heightmap,
        machinable_heightmap=machinable_hf,
        stl_path=stl_path,
        agent_result=agent_result,
    )
