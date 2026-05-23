"""
LLM agent that plans bounded heightmap enhancements.

Architecture (Bounded Agent Transform):
  1. Analyze preprocessed base heightfield (via heightmap_analyzer)
  2. Send analysis + user intent to LLM
  3. LLM uses analysis-only tools and proposes an edit plan
  4. Plan parameters are extracted and clamped to safe ranges
  5. Deterministic code applies the plan later (not the LLM)
  6. Max 3 iterations to keep cost/time bounded
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
    max_iterations: int = 3
    temperature: float = 0.3


@dataclass
class AgentEditPlan:
    """Bounded edit parameters returned by the agent. NOT a heightfield."""
    ridge_boost: float = 0.0       # [0, 0.35]
    contrast_boost: float = 0.0    # [0, 0.25]
    texture_amount: float = 0.0    # [0, 0.10]
    smoothing_sigma: float = 0.0   # [0, 1.5]
    target_regions: str = "global" # "global", "ridges", "valleys", "high", "low"
    preserve_large_structures: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    plan: AgentEditPlan
    analysis_before: HeightmapAnalysis
    analysis_after: HeightmapAnalysis | None  # None until plan is applied
    iterations_used: int
    evaluation_notes: str
    conversation_log: list[dict] = field(default_factory=list)


SYSTEM_PROMPT = """You are a heightmap enhancement planner for CNC-fabricable tactile surfaces.

You receive a PREPROCESSED heightmap analysis and a user intent description.
Your job: analyze the surface and propose BOUNDED ENHANCEMENT PARAMETERS.

You are an ENHANCER, not a creator. You modify an existing stable surface within safe bounds.

CRITICAL RULES:
1. The heightmap has already been preprocessed for terrace machining. Do NOT flatten it.
2. All modifications are bounded: ridge boost [0, 0.35], contrast [0, 0.25], texture [0, 0.10].
3. You CANNOT replace, warp, or redistribute the entire surface.
4. The terrace mesh generator is the final authority on geometry.

ALLOWED TOOLS (analysis and planning only):
- analyze_heightmap: read current metrics
- generate_mask: create spatial masks for regional control
- mask_apply: preview mask effect (requires generate_mask first)
- evaluate_result: check metrics against intent
- propose_edit_plan: submit your final bounded enhancement parameters

You may NOT directly mutate the geometry with boost_contrast, enhance_ridges, etc.
Those operations are applied later by deterministic code based on your plan.

STRATEGY:
1. Call analyze_heightmap to understand the preprocessed surface.
2. Optionally generate_mask for spatial targeting.
3. Call evaluate_result with intent keywords.
4. Call propose_edit_plan with explicit bounded parameters:
   - ridge_boost: 0.0-0.35 (enhance_ridges strength)
   - contrast_boost: 0.0-0.25 (boost_contrast amount)
   - texture_amount: 0.0-0.10 (texture overlay intensity)
   - smoothing_sigma: 0.0-1.5 (smoothing sigma)
   - target_regions: "global", "ridges", "valleys", "high", "low"
   - summary: brief description of the plan

Max 3 refinement iterations."""


def run_agent_pipeline(
    base_hf: np.ndarray,
    raw_hf: np.ndarray,
    user_intent: str,
    config: AgentConfig | None = None,
) -> AgentResult:
    """Run the bounded agent planning pipeline.

    The LLM analyzes the preprocessed base heightfield and proposes
    bounded enhancement parameters. No geometry-mutating tools are
    executed during planning — only analysis, mask generation, and
    plan proposal.
    """
    if config is None:
        config = AgentConfig()

    analysis_before = analyze_heightmap(base_hf)
    current_hf = base_hf.copy()
    log: list[dict] = []
    stored_masks: dict = {}
    all_tool_calls: list[dict] = []

    messages = [
        {"role": "user", "content": _build_initial_message(user_intent, analysis_before)}
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
            # Planning-only: discard geometry changes from tools,
            # keep only analysis feedback and mask/plan state
            if tc["name"] in ("analyze_heightmap", "evaluate_result",
                              "generate_mask", "mask_apply", "propose_edit_plan"):
                current_hf = hf_new
            tool_results.append({"tool_use_id": tc["id"], "content": desc})
            if done:
                is_final = True

        messages.append({"role": "assistant", "content": response["content"]})
        messages.append({"role": "user", "content": tool_results})

        if is_final:
            break

    # Extract bounded parameters from the plan proposal or tool call history
    plan_params = extract_plan_parameters(all_tool_calls, stored_masks)

    plan = AgentEditPlan(
        ridge_boost=plan_params.get("ridge_boost", 0.0),
        contrast_boost=plan_params.get("contrast_boost", 0.0),
        texture_amount=plan_params.get("texture_amount", 0.0),
        smoothing_sigma=plan_params.get("smoothing_sigma", 0.0),
        target_regions=plan_params.get("target_regions", "global"),
        notes=plan_params.get("notes", []),
    )

    return AgentResult(
        plan=plan,
        analysis_before=analysis_before,
        analysis_after=None,
        iterations_used=min(iteration + 1, config.max_iterations),
        evaluation_notes=f"Plan extracted: {len(plan.notes)} parameter notes",
        conversation_log=log,
    )


def _build_initial_message(user_intent: str, analysis: HeightmapAnalysis) -> str:
    return f"""## User Intent
"{user_intent}"

## Preprocessed Heightmap Analysis (base surface)
{analysis_to_text(analysis)}

## Task
Analyze this preprocessed heightmap and propose bounded enhancement parameters.
This surface is already machinable. Your job is to suggest improvements
within safe parameter ranges.

Available tools:
- analyze_heightmap: read current metrics
- generate_mask: create spatial masks for regional control
- evaluate_result: check metrics against intent keywords
- propose_edit_plan: submit bounded enhancement parameters

Do NOT call geometry-mutating tools directly. Use propose_edit_plan to
specify what enhancements to apply. The parameters will be clamped:
- ridge_boost: 0.0-0.35
- contrast_boost: 0.0-0.25
- texture_amount: 0.0-0.10
- smoothing_sigma: 0.0-1.5

Start with analyze_heightmap, evaluate, then propose_edit_plan."""


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
