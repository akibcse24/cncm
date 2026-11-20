"""CNCM Dual-Mode Agentic AI package (tools, JSON grammar masking, runtime)."""

from cncm.agent.grammar import (
    JSONSchemaConstrainedMasker,
    KeywordPlanMasker,
    extract_json_object,
    parse_plan_tokens,
)
from cncm.agent.runtime import AgentStep, AgentTrace, CNCMAgent
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
    "ToolSpec",
    "ToolCallResult",
    "ToolRegistry",
    "safe_eval_math",
    "make_calculator_tool",
    "make_kv_memory_tool",
    "make_text_stats_tool",
    "make_clock_utc_tool",
    "create_default_registry",
    "JSONSchemaConstrainedMasker",
    "KeywordPlanMasker",
    "extract_json_object",
    "parse_plan_tokens",
    "AgentStep",
    "AgentTrace",
    "CNCMAgent",
]
