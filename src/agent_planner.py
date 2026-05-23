"""
LLM agent that plans heightmap transformations using the toolkit.

Architecture:
  1. Analyze raw heightmap (via heightmap_analyzer)
  2. Send analysis + user intent to LLM
  3. LLM returns structured tool calls
  4. Execute calls, analyze result
  5. LLM evaluates result, decides: accept or refine
  6. Max 3 iterations to keep cost/time bounded
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from heightmap_analyzer import (
    HeightmapAnalysis,
    analyze_heightmap,
    analysis_to_text,
    compare_analyses,
)
from agent_tools import TOOL_SCHEMAS, execute_tool, quick_machinability_check


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""
    max_iterations: int = 5
    temperature: float = 0.3


@dataclass
class AgentResult:
    heightmap: np.ndarray
    analysis_before: HeightmapAnalysis
    analysis_after: HeightmapAnalysis
    iterations_used: int
    evaluation_notes: str
    conversation_log: list[dict] = field(default_factory=list)


SYSTEM_PROMPT = """You are a heightmap transformation agent for CNC-fabricable tactile surfaces.

You receive a heightmap analysis and a user intent description.
Your job: call toolkit operations to transform the heightmap to match the user's intent.

You are a CREATOR, not an editor. Make bold, dramatic transformations.

CRITICAL RULE — NEVER FLATTEN THE HEIGHTMAP:
- The input heightmap has rich features (roughness, ridges, valleys). You MUST preserve or enhance them.
- NEVER reduce the heightmap to a smooth/flat surface. The output must have MORE contrast than the input.
- At most ONE smoothing operation per iteration (e.g. one call to freq_preserve_lowpass or bandpass_filter).
- Do NOT chain multiple smoothing/blurring operations — they compound and destroy features.
- If you see height_std drop below 0.08, STOP smoothing and start boosting contrast/ridges.
- The final heightmap MUST have height_std > 0.10 (measured by evaluate_result).

DEFAULT STRATEGY — Boost + Texture:
1. Analyze the heightmap with analyze_heightmap.
2. Boost contrast: boost_contrast(gamma=2.0-3.0) to sharpen features.
3. Enhance ridges: enhance_ridges(strength=1.0-2.0) to amplify linear features.
4. Add texture: blend_pattern or texture_overlay with amplitude=0.3-0.5.
5. Directional flow: directional_warp or anisotropic_emphasis for grain direction.
6. Evaluate: check that roughness and height_std are HIGHER than before.

If the intent asks for "stepped" or "stripes":
- Use freq_stepped_convert(n_levels=8-12) — NOT fewer levels, which flattens.
- Apply it ONCE on the original heightmap, not after multiple smoothing ops.

If the intent asks for "smooth" or "soft":
- Use gentle operations only: boost_contrast(gamma=1.3), enhance_ridges(strength=0.3).
- Do NOT flatten — "smooth" means reduce roughness slightly, not eliminate features.

GUIDELINES:
- Start with analyze_heightmap to understand the current state.
- Plan 3-5 operations per iteration. Quality over quantity.
- Always end with evaluate_result to check progress.
- Use accept_heightmap when the result matches the intent.
- Amplitude 0.3-0.8 is normal for additive operations. Don't be timid.
- The final heightmap must be valid for terrace_geometry: float32 [0,1], 512x512.
- Max 5 refinement iterations.
- Downstream terrace_geometry will fix machinability issues — focus on creativity.

AVAILABLE TOOLS (grouped by purpose):
BOOST: boost_contrast, enhance_ridges, freq_band_boost
TEXTURE: blend_pattern, texture_overlay, add_perlin_noise
DIRECTION: directional_warp, anisotropic_emphasis, bend_contours
REGIONAL: height_selective_transform, height_redistribute, height_zone_remap
CREATIVE: replace_region, surface_warp, procedural_generate, symmetry_apply
COMPOSITION: blend_two, generate_mask + mask_apply
FREQUENCY: freq_preserve_lowpass, freq_stepped_convert, bandpass_filter

MACHINING CONSTRAINTS:
- Tool: 6mm flat end mill
- Features narrower than 6mm will be suppressed by downstream processing
- Maximum physical height: 5mm
- Keep heightmap values in [0, 1]"""


def run_agent_pipeline(
    raw_heightmap: np.ndarray,
    user_intent: str,
    config: AgentConfig | None = None,
) -> AgentResult:
    """Main entry point: run the full agent pipeline."""
    if config is None:
        config = AgentConfig()

    analysis_before = analyze_heightmap(raw_heightmap)
    current_hf = raw_heightmap.copy()
    log: list[dict] = []
    stored_masks: dict = {}

    messages = [
        {"role": "user", "content": _build_initial_message(user_intent, analysis_before)}
    ]

    for iteration in range(config.max_iterations):
        response = _call_llm(messages, config, log)
        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            break

        tool_results = []
        is_final = False
        for tc in tool_calls:
            hf_new, desc, done = execute_tool(
                tc["name"], tc.get("input", {}),
                current_hf, raw_heightmap, analysis_before,
                stored_masks=stored_masks,
            )
            current_hf = hf_new
            tool_results.append({"tool_use_id": tc["id"], "content": desc})
            if done:
                is_final = True

        messages.append({"role": "assistant", "content": response["content"]})
        messages.append({"role": "user", "content": tool_results})

        if is_final:
            break

        check = quick_machinability_check(current_hf)
        if not check["likely_machinable"]:
            messages.append({
                "role": "user",
                "content": f"Machinability warning: max_slope={check['max_slope_deg']:.1f}deg, "
                           f"height_range={check['height_range']:.3f}. "
                           f"Consider reducing operation strengths.",
            })

        # Anti-flattening guard
        current_std = float(current_hf.std())
        original_std = float(raw_heightmap.std())
        if current_std < 0.06:
            messages.append({
                "role": "user",
                "content": f"FLATTENING DETECTED: height_std={current_std:.4f} is critically low. "
                           f"The heightmap is nearly flat. STOP all smoothing operations immediately. "
                           f"Use boost_contrast(gamma>2.0) or enhance_ridges(strength>1.0) to restore features.",
            })
        elif current_std < original_std * 0.4:
            messages.append({
                "role": "user",
                "content": f"FEATURE LOSS WARNING: height_std dropped from {original_std:.4f} to {current_std:.4f} "
                           f"({current_std/original_std*100:.0f}% remaining). "
                           f"Do NOT apply more smoothing. Use boost_contrast or blend_pattern to add detail back.",
            })

    analysis_after = analyze_heightmap(current_hf)
    comparison = compare_analyses(analysis_before, analysis_after)

    return AgentResult(
        heightmap=current_hf,
        analysis_before=analysis_before,
        analysis_after=analysis_after,
        iterations_used=min(iteration + 1, config.max_iterations),
        evaluation_notes=comparison,
        conversation_log=log,
    )


def _build_initial_message(user_intent: str, analysis: HeightmapAnalysis) -> str:
    return f"""## User Intent
"{user_intent}"

## Current Heightmap Analysis
{analysis_to_text(analysis)}

## Baseline Metrics (MUST preserve or improve)
- height_std: {analysis.height_std:.4f} — do NOT let this drop below {analysis.height_std * 0.5:.4f}
- roughness: {analysis.roughness:.3f} — do NOT let this drop below {analysis.roughness * 0.5:.3f}
- height_range: {analysis.height_range:.3f} — do NOT let this drop below {analysis.height_range * 0.5:.3f}

Please plan a sequence of toolkit operations to transform this heightmap
to match the user's intent. Start with analyze_heightmap, then apply
3-5 operations, and end with evaluate_result.

REMEMBER: The heightmap already has rich features. Your job is to SHAPE them, not ERASE them."""


def _call_llm(messages: list[dict], config: AgentConfig, log: list[dict]) -> dict:
    """Call the LLM API via Anthropic protocol (works for Claude and MiMo)."""
    import anthropic

    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client_kwargs = {"api_key": api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    client = anthropic.Anthropic(**client_kwargs)

    response = client.messages.create(
        model=config.model,
        max_tokens=8192,
        temperature=config.temperature,
        system=SYSTEM_PROMPT,
        tools=_format_tools(),
        messages=messages,
    )
    content = []
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    log.append({"role": "assistant", "content": content})
    return {"content": content}


def _format_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOL_SCHEMAS]


def _extract_tool_calls(response: dict) -> list[dict]:
    calls = []
    for item in response.get("content", []):
        if isinstance(item, dict) and item.get("type") == "tool_use":
            calls.append({"id": item["id"], "name": item["name"], "input": item.get("input", {})})
    return calls
