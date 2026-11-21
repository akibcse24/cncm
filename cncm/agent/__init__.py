"""CNCM Dual-Mode Agentic AI package (tools, JSON grammar masking, runtime)."""

from __future__ import annotations

from cncm.agent.grammar import (
    evaluate_tool_call_semantic_accuracy,
    extract_json_object,
    extract_prompt_math_spans,
    parse_plan_tokens,
)
from cncm.agent.tools import (
    ToolCallResult,
    ToolRegistry,
    ToolSpec,
    create_default_registry,
    make_calculator_tool,
    make_clock_utc_tool,
    make_kv_memory_tool,
    make_text_stats_tool,
    safe_eval_math,
)

__all__ = [
    "CNCMAgent",
    "AgentStep",
    "AgentTrace",
    "JSONSchemaConstrainedMasker",
    "KeywordPlanMasker",
    "ToolCallResult",
    "ToolRegistry",
    "ToolSpec",
    "create_default_registry",
    "evaluate_tool_call_semantic_accuracy",
    "extract_json_object",
    "extract_prompt_math_spans",
    "make_calculator_tool",
    "make_clock_utc_tool",
    "make_kv_memory_tool",
    "make_text_stats_tool",
    "parse_plan_tokens",
    "safe_eval_math",
]


def __getattr__(name: str):
    if name in ("JSONSchemaConstrainedMasker", "KeywordPlanMasker"):
        from cncm.agent import grammar
        return getattr(grammar, name)
    if name in ("CNCMAgent", "AgentStep", "AgentTrace"):
        from cncm.agent import runtime
        return getattr(runtime, name)
    raise AttributeError(f"module 'cncm.agent' has no attribute {name!r}")
