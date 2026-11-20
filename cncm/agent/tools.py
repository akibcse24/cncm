"""Local zero-dependency tool specifications and registry for CNCM Agentic AI.

Provides:
- ``ToolSpec``: declarative tool definition with JSON-schema parameters.
- ``ToolCallResult``: structured execution result serialized back into
  ``<tool_response>...</tool_response>``.
- ``ToolRegistry``: registration, compact prompt schema formatting, parameter
  validation, and sandboxed safe execution.
- 4 built-in edge-friendly zero-dependency tools:
  1. ``calculator`` (safe AST arithmetic evaluator)
  2. ``kv_memory`` (session-scoped key-value working memory)
  3. ``text_stats`` (deterministic character/word/line metrics)
  4. ``clock_utc`` (UTC wall-clock timestamp)
"""

from __future__ import annotations

import ast
import datetime
import json
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ToolSpec:
    """Declarative specification for a single callable agent tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]

    def to_schema_dict(self) -> Dict[str, Any]:
        """Return OpenAI / Qwen / MiniMind-compatible tool schema dictionary."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_schema(self) -> Dict[str, Any]:
        """Alias for ``to_schema_dict`` compatible with ``CNCMTextTokenizer.apply_chat_template``."""
        return self.to_schema_dict()

    def required_params(self) -> List[str]:
        return list(self.parameters.get("required", []))

    def properties(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.parameters.get("properties", {}))


@dataclass
class ToolCallResult:
    """Structured outcome from executing a registered tool."""

    name: str
    arguments: Dict[str, Any]
    ok: bool
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
        }
        if self.ok:
            payload["result"] = self.output
        else:
            payload["error"] = self.error or "Unknown tool error"
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Safe AST Calculator
# ---------------------------------------------------------------------------

_SAFE_BIN_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_SAFE_CONSTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_MAX_AST_NODES = 128
_MAX_EXPONENT = 1000.0


def safe_eval_math(expression: str) -> float:
    """Safely evaluate an arithmetic expression via Python's AST module.

    Rejects attribute access, imports, subscripts, comprehensions, lambdas,
    and unbounded exponentiation.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")
    expr = expression.strip()
    if len(expr) > 256:
        raise ValueError("Expression exceeds maximum length (256 chars).")

    tree = ast.parse(expr, mode="eval")
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_AST_NODES:
        raise ValueError(f"Expression too complex ({node_count} AST nodes).")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTS:
                return _SAFE_CONSTS[node.id]
            raise ValueError(f"Unknown identifier: {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return float(op_fn(_eval(node.operand)))
        if isinstance(node, ast.BinOp):
            op_fn = _SAFE_BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent {right} exceeds safety limit {_MAX_EXPONENT}.")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0.0:
                raise ZeroDivisionError("Division by zero.")
            return float(op_fn(left, right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct calls to whitelisted math functions are allowed.")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in calculator.")
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Disallowed function: {node.func.id!r}")
            args = [_eval(arg) for arg in node.args]
            return float(fn(*args))
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    result = _eval(tree)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric result: {result}")
    # Return clean int when exact integer value fits comfortably
    if result.is_integer() and abs(result) < 1e15:
        return int(result)
    return round(result, 10)


# ---------------------------------------------------------------------------
# Built-in Tool Builders
# ---------------------------------------------------------------------------

def make_calculator_tool() -> ToolSpec:
    def _calculator(expression: str) -> Dict[str, Any]:
        value = safe_eval_math(expression)
        return {"expression": expression.strip(), "value": value}

    return ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression safely using AST math.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "format": "math_expression",
                    "description": "Arithmetic expression, e.g. '12 * 34 + 5'.",
                }
            },
            "required": ["expression"],
        },
        fn=_calculator,
    )


def make_kv_memory_tool(store: Optional[Dict[str, str]] = None) -> ToolSpec:
    memory: Dict[str, str] = store if store is not None else {}

    def _kv_memory(action: str, key: str = "", value: str = "") -> Dict[str, Any]:
        act = str(action).strip().lower()
        if act == "set":
            if not key:
                raise ValueError("kv_memory 'set' requires a non-empty 'key'.")
            memory[key] = str(value)
            return {"action": "set", "key": key, "value": memory[key], "size": len(memory)}
        if act == "get":
            return {"action": "get", "key": key, "found": key in memory, "value": memory.get(key)}
        if act == "delete":
            existed = key in memory
            memory.pop(key, None)
            return {"action": "delete", "key": key, "deleted": existed, "size": len(memory)}
        if act == "list":
            return {"action": "list", "keys": sorted(memory.keys()), "size": len(memory)}
        raise ValueError(f"Unknown kv_memory action {action!r}; expected get|set|delete|list.")

    return ToolSpec(
        name="kv_memory",
        description="Read, write, delete, or list working-memory key/value pairs.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "delete", "list"],
                    "description": "Memory operation to perform.",
                },
                "key": {
                    "type": "string",
                    "description": "Memory key name.",
                },
                "value": {
                    "type": "string",
                    "description": "Value to store when action is 'set'.",
                },
            },
            "required": ["action", "key"],
        },
        fn=_kv_memory,
    )


def make_text_stats_tool() -> ToolSpec:
    def _text_stats(text: str) -> Dict[str, Any]:
        s = str(text)
        words = s.split()
        lines = s.splitlines() if s else []
        unique_words = len({w.lower() for w in words})
        avg_len = round(sum(len(w) for w in words) / len(words), 3) if words else 0.0
        return {
            "chars": len(s),
            "words": len(words),
            "lines": len(lines),
            "unique_words": unique_words,
            "avg_word_len": avg_len,
        }

    return ToolSpec(
        name="text_stats",
        description="Compute character, word, line, and unique-word counts for a text string.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text to inspect.",
                }
            },
            "required": ["text"],
        },
        fn=_text_stats,
    )


def make_clock_utc_tool() -> ToolSpec:
    def _clock_utc(format: str = "iso") -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        fmt = str(format).strip().lower()
        if fmt == "date":
            formatted = now.strftime("%Y-%m-%d")
        elif fmt == "unix":
            formatted = str(int(now.timestamp()))
        else:
            formatted = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "utc": formatted,
            "unix": int(now.timestamp()),
            "weekday": now.strftime("%A"),
        }

    return ToolSpec(
        name="clock_utc",
        description="Return the current UTC date/time timestamp.",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["iso", "date", "unix"],
                    "description": "Output timestamp format.",
                }
            },
            "required": ["format"],
        },
        fn=_clock_utc,
    )


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry managing agent tools, prompt schemas, and safe execution."""

    def __init__(self, register_builtins: bool = False) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self.memory_store: Dict[str, str] = {}
        if register_builtins:
            self.register_builtins()

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("ToolSpec.name must be a non-empty string.")
        self._tools[spec.name] = spec

    def register_fn(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        fn: Callable[..., Any],
    ) -> ToolSpec:
        spec = ToolSpec(name=name, description=description, parameters=parameters, fn=fn)
        self.register(spec)
        return spec

    def register_builtins(self) -> "ToolRegistry":
        """Register the 4 built-in zero-dependency local tools."""
        self.register(make_calculator_tool())
        self.register(make_kv_memory_tool(self.memory_store))
        self.register(make_text_stats_tool())
        self.register(make_clock_utc_tool())
        return self

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def to_schema_list(self) -> List[Dict[str, Any]]:
        return [spec.to_schema_dict() for spec in self._tools.values()]

    def format_prompt_schema(self, tool_names: Optional[Iterable[str]] = None) -> str:
        """Render a compact tool schema block optimized for small edge contexts."""
        selected = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names is not None
            else list(self._tools.values())
        )
        lines = ["Available tools (invoke inside <tool_call>{...}</tool_call>):"]
        for spec in selected:
            props = spec.properties()
            req = set(spec.required_params())
            arg_parts = []
            for k, schema in props.items():
                typ = schema.get("type", "any")
                if "enum" in schema:
                    typ = "|".join(str(v) for v in schema["enum"])
                star = "" if k in req else "?"
                arg_parts.append(f"{k}{star}: {typ}")
            sig = ", ".join(arg_parts)
            lines.append(f"- {spec.name}({sig}): {spec.description}")
        return "\n".join(lines)

    def _validate_arguments(self, spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            raise TypeError(f"Tool arguments must be a JSON object, got {type(arguments).__name__}.")
        props = spec.properties()
        required = spec.required_params()
        missing = [k for k in required if k not in arguments]
        if missing:
            raise ValueError(f"Missing required argument(s) for '{spec.name}': {missing}")

        cleaned: Dict[str, Any] = {}
        for k, v in arguments.items():
            if props and k not in props:
                # Ignore unknown extra keys gracefully unless schema explicitly forbids them
                if spec.parameters.get("additionalProperties") is False:
                    raise ValueError(f"Unexpected argument '{k}' for tool '{spec.name}'.")
                continue
            prop_schema = props.get(k, {})
            if "enum" in prop_schema and v not in prop_schema["enum"]:
                raise ValueError(
                    f"Argument '{k}'={v!r} not in allowed enum {prop_schema['enum']}."
                )
            cleaned[k] = v
        return cleaned

    def execute(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> ToolCallResult:
        """Execute a registered tool safely, catching all runtime/validation errors."""
        args = {} if arguments is None else arguments
        spec = self._tools.get(name)
        if spec is None:
            return ToolCallResult(
                name=str(name),
                arguments=args if isinstance(args, dict) else {},
                ok=False,
                error=f"Unknown tool {name!r}. Registered: {self.list_names()}",
            )
        try:
            validated = self._validate_arguments(spec, args)
            output = spec.fn(**validated)
            return ToolCallResult(
                name=spec.name,
                arguments=validated,
                ok=True,
                output=output,
            )
        except Exception as exc:  # noqa: BLE001 — tool errors must never crash agent
            return ToolCallResult(
                name=spec.name,
                arguments=args if isinstance(args, dict) else {},
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def execute_json(self, raw_json: str) -> ToolCallResult:
        """Parse a ``{"name": ..., "arguments": {...}}`` JSON payload and execute it."""
        try:
            payload = json.loads(raw_json)
        except Exception as exc:  # noqa: BLE001
            return ToolCallResult(
                name="<parse_error>",
                arguments={},
                ok=False,
                error=f"Invalid JSON: {exc}",
            )
        if not isinstance(payload, dict):
            return ToolCallResult(
                name="<schema_error>",
                arguments={},
                ok=False,
                error="Tool call payload must be a JSON object.",
            )
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            return ToolCallResult(
                name="<schema_error>",
                arguments=arguments if isinstance(arguments, dict) else {},
                ok=False,
                error="Tool call JSON missing string field 'name'.",
            )
        if not isinstance(arguments, dict):
            return ToolCallResult(
                name=name,
                arguments={},
                ok=False,
                error="Tool call JSON field 'arguments' must be an object.",
            )
        return self.execute(name, arguments)


def create_default_registry() -> ToolRegistry:
    """Create a ``ToolRegistry`` pre-populated with all 4 built-in local tools."""
    return ToolRegistry(register_builtins=True)


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
]
