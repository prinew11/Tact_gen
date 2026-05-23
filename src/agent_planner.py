"""
LLM agent that plans creative tactile heightmap transformations.

Architecture (Creative Tactile Designer):
  1. Analyze preprocessed base heightfield (via heightmap_analyzer)
  2. Send analysis + user intent + constraints to LLM
  3. LLM uses creative tools to directly transform the heightmap
  4. Agent executes tools sequentially, building an operation log
  5. Final heightmap is the result of the tool chain
  6. Max 4 iterations for creative exploration
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from heightmap_analyzer import (
    HeightmapAnalysis,
    analyze_heightmap,
    analysis_to_text,
)
from agent_tools import TOOL_SCHEMAS, execute_tool, extract_plan_parameters


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    base_url: str = ""
    max_iterations: int = 4
    temperature: float = 0.3


@dataclass
class AgentEditPlan:
    """Creative transformation plan — records what the agent did."""
    operations: list[dict] = field(default_factory=list)
    topology_preserved: bool = True
    reasoning: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    plan: AgentEditPlan
    analysis_before: HeightmapAnalysis
    analysis_after: HeightmapAnalysis | None  # None until plan is applied
    final_hf: np.ndarray | None  # Heightmap after agent tool execution
    iterations_used: int
    evaluation_notes: str
    conversation_log: list[dict] = field(default_factory=list)


SYSTEM_PROMPT = """You are a creative tactile surface designer for CNC-fabricable stepped heightmaps.

You receive a PREPROCESSED heightmap analysis, a user intent description,
and machining constraints. Your job: creatively transform the heightmap
into a visually compelling tactile surface that satisfies the user's intent.

You are a DESIGNER, not a cautious enhancer. You have freedom to:
- Redistribute height topology to create distinct tactile regions
- Generate directional textures (wood grain, ridges, flow patterns)
- Blend procedural bases (voronoi, hex, waves) into the surface
- Carve regional features (valleys, ridges, channels)
- Smooth topology for organic feel

TOPOLOGY RULES (hard constraints — violations break CNC):
1. NEVER create isolated islands narrower than tool_diameter (fragments break)
2. NEVER create internal concave corners sharper than tool_radius (tool can't reach)
3. Prefer wide connected regions over scattered thin features
4. After major reshaping, call smooth_topology to repair topology violations

STRATEGY:
1. Call analyze_heightmap to understand the current surface.
2. Based on user intent, call creative tools in sequence:
   - redistribute_topology for height rebalancing
   - generate_directional_texture for grain/ridge patterns
   - blend_procedural_base for organic structure
   - regional_carve for focused features
   - smooth_topology to ensure machinability
3. Call evaluate_result to check metrics against intent.
4. Call accept_result when satisfied with the transformation.

You may call tools multiple times and in any order. Build up the
surface progressively. Each tool operates on the CURRENT heightmap
state, so order matters.

Max 4 refinement iterations."""


def run_agent_pipeline(
    base_hf: np.ndarray,
    raw_hf: np.ndarray,
    user_intent: str,
    config: AgentConfig | None = None,
    image_gray: np.ndarray | None = None,
    constraints: dict | None = None,
) -> AgentResult:
    """Run the creative agent planning pipeline.

    The LLM analyzes the preprocessed base heightfield and directly
    executes creative tools to transform it. The final heightmap is
    returned in AgentResult.final_hf.
    """
    if config is None:
        config = AgentConfig()

    if constraints is None:
        constraints = {}

    analysis_before = analyze_heightmap(base_hf)
    current_hf = base_hf.copy()
    log: list[dict] = []
    stored_masks: dict = {}
    all_tool_calls: list[dict] = []

    # Store image_gray for image_guided_ridge_restore tool
    if image_gray is not None:
        stored_masks["_image_gray"] = image_gray

    messages = [
        {"role": "user", "content": _build_initial_message(user_intent, analysis_before, constraints)}
    ]

    for iteration in range(config.max_iterations):
        response = _call_llm(messages, config, log)
        tool_calls = _extract_tool_calls(response)

        if not tool_calls:
            break

        all_tool_calls.extend(tool_calls)
        tool_results = []
        is_final = False
        for tc in tool_calls:
            hf_new, desc, done = execute_tool(
                tc["name"], tc.get("input", {}),
                current_hf, base_hf, analysis_before,
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

    # Extract plan info from operation log
    plan_params = extract_plan_parameters(all_tool_calls, stored_masks)

    plan = AgentEditPlan(
        operations=plan_params.get("operations", []),
        topology_preserved=plan_params.get("topology_preserved", True),
        reasoning=plan_params.get("reasoning", ""),
        notes=plan_params.get("notes", []),
    )

    return AgentResult(
        plan=plan,
        analysis_before=analysis_before,
        analysis_after=None,
        final_hf=current_hf,
        iterations_used=min(iteration + 1, config.max_iterations),
        evaluation_notes=f"Creative plan: {len(plan.operations)} operations applied",
        conversation_log=log,
    )


def _build_initial_message(user_intent: str, analysis: HeightmapAnalysis, constraints: dict) -> str:
    constraint_text = ""
    if constraints:
        lines = []
        if "tool_diameter_mm" in constraints:
            lines.append(f"- Tool diameter: {constraints['tool_diameter_mm']}mm (min feature width)")
        if "tool_radius_mm" in constraints:
            lines.append(f"- Tool radius: {constraints['tool_radius_mm']}mm (min concave radius)")
        if "physical_size_mm" in constraints:
            lines.append(f"- Physical size: {constraints['physical_size_mm']}mm")
        if "max_height_mm" in constraints:
            lines.append(f"- Max height: {constraints['max_height_mm']}mm")
        if "terrace_steps" in constraints:
            lines.append(f"- Terrace steps: {constraints['terrace_steps']}")
        if lines:
            constraint_text = "\n\n## Machining Constraints\n" + "\n".join(lines)

    return f"""## User Intent
"{user_intent}"

## Preprocessed Heightmap Analysis (base surface)
{analysis_to_text(analysis)}
{constraint_text}

## Task
Transform this heightmap into a creative tactile surface matching the user's intent.
You have full creative freedom — redistribute topology, add textures, carve features.

Available tools:
- analyze_heightmap: read current metrics
- redistribute_topology: remap height histogram (uniform/gaussian/bimodal)
- generate_directional_texture: add grain-aligned texture patterns
- blend_procedural_base: blend procedural pattern (voronoi/hex/waves/brick)
- regional_carve: carve focused features (valley/ridge/channel/dome) in a region
- smooth_topology: smooth surface to repair topology violations
- generate_mask: create spatial masks for regional control
- mask_apply: apply modifications only within a mask region
- evaluate_result: check metrics against intent keywords
- accept_result: accept current heightmap as final

IMPORTANT: After major reshaping, call smooth_topology to ensure
the surface has no isolated islands or sharp concave corners.

Start with analyze_heightmap, then build up your transformation."""


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
