"""Safe AST-gated and timeout-enforced Python code sandbox for PG-FIM & GRPO RL.

Provides ``run_code_sandbox(code, test_code="", timeout_sec=2.0)`` returning:
  ``{"parses": bool, "compiles": bool, "tests_passed": bool, "non_trivial": bool, "error": Optional[str], "elapsed_ms": float}``

Security & isolation layers:
1. Static AST audit rejects dangerous imports (`os`, `sys`, `subprocess`, `socket`,
   `urllib`, `http`, `shutil`, `pathlib`, `ctypes`, etc.), dangerous builtins
   (`eval`, `exec`, `open`, `__import__`, `compile`, `globals`, `locals`, `getattr`),
   and dunder attribute escapes (`__globals__`, `__subclasses__`, `__builtins__`, etc.).
2. AST loop/function tick instrumentation (`_SandboxTickTransformer`) injects a
   wall-clock deadline and step-budget check into every loop and function body,
   aborting infinite loops (`while True: pass`) or runaway recursion in milliseconds
   without slow Windows process-spawn overhead.
3. Restricted `__builtins__` dictionary exposing only safe pure-computation builtins
   and whitelisted standard library modules (`math`, `collections`, `itertools`, etc.).
"""

from __future__ import annotations

import ast
import builtins
import collections
import copy
import functools
import heapq
import itertools
import json
import math
import operator
import re
import statistics
import string
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

from cncm.code.symbols import _is_non_trivial_middle

_ALLOWED_IMPORTS: Dict[str, Any] = {
    "math": math,
    "collections": collections,
    "itertools": itertools,
    "functools": functools,
    "heapq": heapq,
    "re": re,
    "string": string,
    "operator": operator,
    "statistics": statistics,
    "copy": copy,
    "json": json,
}

_ALLOWED_TYPING_NAMES: Set[str] = {
    "Any",
    "Callable",
    "Dict",
    "Iterable",
    "Iterator",
    "List",
    "Mapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Union",
}

_DISALLOWED_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "exit",
    "quit",
    "help",
    "memoryview",
}

_DISALLOWED_ATTRS: Set[str] = {
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
    "__dict__",
    "__class__",
    "__loader__",
    "__spec__",
    "system",
    "popen",
    "spawn",
    "fork",
    "kill",
    "unlink",
    "remove",
    "rmdir",
    "rmtree",
}

_SAFE_BUILTIN_NAMES: Tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "ZeroDivisionError",
    "AssertionError",
    "RuntimeError",
    "StopIteration",
    "ArithmeticError",
    "OverflowError",
)


class SandboxSecurityError(RuntimeError):
    """Raised when candidate code violates sandbox AST or import policies."""


class SandboxTimeoutError(TimeoutError):
    """Raised when candidate code exceeds the sandbox wall-clock or step budget."""


def _validate_ast_safety(tree: ast.AST) -> Optional[str]:
    """Inspect ``tree`` for forbidden imports, builtin calls, or dunder attribute escapes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod != "typing" and root_mod not in _ALLOWED_IMPORTS:
                    return f"SandboxSecurityError: Disallowed import '{alias.name}'"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod != "typing" and mod not in _ALLOWED_IMPORTS:
                return f"SandboxSecurityError: Disallowed import from '{node.module}'"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _DISALLOWED_CALLS:
                return f"SandboxSecurityError: Disallowed call '{node.func.id}()'"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _DISALLOWED_CALLS:
                return f"SandboxSecurityError: Disallowed method call '.{node.func.attr}()'"
        elif isinstance(node, ast.Attribute):
            if node.attr in _DISALLOWED_ATTRS or (
                node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ("__init__", "__len__", "__getitem__", "__setitem__", "__iter__", "__next__", "__repr__", "__str__", "__eq__", "__lt__", "__le__", "__gt__", "__ge__", "__add__", "__sub__", "__mul__", "__call__")
            ):
                return f"SandboxSecurityError: Disallowed attribute access '.{node.attr}'"
        elif isinstance(node, ast.Name):
            if node.id in ("__builtins__", "__loader__", "__spec__", "_sandbox_tick"):
                return f"SandboxSecurityError: Disallowed identifier '{node.id}'"
    return None


class _SandboxTickTransformer(ast.NodeTransformer):
    """Inject `_sandbox_tick()` at the start of every loop and function body."""

    @staticmethod
    def _make_tick_stmt(ref_node: ast.AST) -> ast.stmt:
        call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="_sandbox_tick", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        )
        return ast.copy_location(call, ref_node)

    def _inject_body(self, node: ast.AST, body: list[ast.stmt]) -> list[ast.stmt]:
        tick = self._make_tick_stmt(node)
        return [tick] + [self.visit(stmt) for stmt in body]

    def visit_While(self, node: ast.While) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self._inject_body(node, node.body)
        node.orelse = [self.visit(s) for s in node.orelse]
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._inject_body(node, node.body)
        node.orelse = [self.visit(s) for s in node.orelse]
        return node

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._inject_body(node, node.body)
        node.orelse = [self.visit(s) for s in node.orelse]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._inject_body(node, node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = self._inject_body(node, node.body)
        return node


def _safe_import(
    name: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None,
    fromlist: Sequence[str] = (),
    level: int = 0,
) -> Any:
    del globals_dict, locals_dict, fromlist
    if level != 0:
        raise SandboxSecurityError("Relative imports are disabled in sandbox.")
    root = name.split(".")[0]
    if root == "typing":
        import typing
        return typing
    if root in _ALLOWED_IMPORTS:
        return _ALLOWED_IMPORTS[root]
    raise SandboxSecurityError(f"Disallowed import '{name}' in sandbox.")


def _build_safe_globals(tick_fn: Callable[[], None]) -> Dict[str, Any]:
    safe_b: Dict[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    safe_b["__import__"] = _safe_import
    # Silence print inside sandbox so rollouts don't spam stdout
    safe_b["print"] = lambda *args, **kwargs: None

    env: Dict[str, Any] = {
        "__builtins__": safe_b,
        "__name__": "__sandbox__",
        "_sandbox_tick": tick_fn,
    }
    import typing
    for tname in _ALLOWED_TYPING_NAMES:
        if hasattr(typing, tname):
            env[tname] = getattr(typing, tname)
    return env


def run_code_sandbox(
    code: str,
    test_code: str = "",
    timeout_sec: float = 2.0,
    min_tokens: int = 2,
    middle: Optional[str] = None,
    max_ticks: int = 200_000,
) -> Dict[str, Any]:
    """Parse, compile, security-check, and execute ``code + test_code`` in a timeout-guarded sandbox.

    Parameters
    ----------
    code:
        Full candidate code string (e.g. ``prefix + middle + suffix``).
    test_code:
        Optional assertion / unit-test script executed in the same namespace after ``code``.
    timeout_sec:
        Hard wall-clock execution timeout in seconds (default ``2.0``).
    min_tokens:
        Minimum lexical token count required for ``non_trivial=True``.
    middle:
        Optional middle infill span to evaluate for ``non_trivial`` (defaults to ``code``).
    max_ticks:
        Maximum loop/function iterations permitted before raising ``SandboxTimeoutError``.
    """
    t0 = time.perf_counter()
    target_span = middle if middle is not None else code
    non_trivial = _is_non_trivial_middle(target_span, min_tokens=min_tokens)

    if not isinstance(code, str) or not code.strip():
        return {
            "parses": False,
            "compiles": False,
            "tests_passed": False,
            "non_trivial": False,
            "error": "SyntaxError: Empty code",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 1. Parse candidate code (and combined code + test_code)
    try:
        code_tree = ast.parse(code)
    except Exception as exc:  # noqa: BLE001
        return {
            "parses": False,
            "compiles": False,
            "tests_passed": False,
            "non_trivial": non_trivial,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    full_script = f"{code}\n\n{test_code}" if test_code and test_code.strip() else code
    try:
        full_tree = ast.parse(full_script)
    except Exception as exc:  # noqa: BLE001
        return {
            "parses": True,
            "compiles": False,
            "tests_passed": False,
            "non_trivial": non_trivial,
            "error": f"TestParseError ({type(exc).__name__}): {exc}",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 2. Static AST security validation & bytecode compile check
    sec_err = _validate_ast_safety(code_tree) or _validate_ast_safety(full_tree)
    if sec_err is not None:
        return {
            "parses": True,
            "compiles": False,
            "tests_passed": False,
            "non_trivial": non_trivial,
            "error": sec_err,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    try:
        instrumented = _SandboxTickTransformer().visit(full_tree)
        ast.fix_missing_locations(instrumented)
        compiled = compile(instrumented, "<cncm_sandbox>", "exec")
    except Exception as exc:  # noqa: BLE001
        return {
            "parses": True,
            "compiles": False,
            "tests_passed": False,
            "non_trivial": non_trivial,
            "error": f"CompileError ({type(exc).__name__}): {exc}",
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 3. Execute in restricted namespace with wall-clock deadline & tick cap
    deadline = time.perf_counter() + max(0.01, float(timeout_sec))
    ticks = 0

    def _sandbox_tick() -> None:
        nonlocal ticks
        ticks += 1
        if ticks > max_ticks:
            raise SandboxTimeoutError(f"Exceeded max sandbox loop/call ticks ({max_ticks})")
        if (ticks & 63) == 0 and time.perf_counter() > deadline:
            raise SandboxTimeoutError(f"Exceeded sandbox timeout ({timeout_sec:.2f}s)")

    env = _build_safe_globals(_sandbox_tick)
    try:
        exec(compiled, env, env)  # noqa: S102 — AST-audited + restricted builtins + tick guard
        passed = True
        err_msg = None
    except Exception as exc:  # noqa: BLE001
        passed = False
        err_msg = f"{type(exc).__name__}: {exc}"

    return {
        "parses": True,
        "compiles": True,
        "tests_passed": passed,
        "non_trivial": non_trivial,
        "error": err_msg,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


__all__ = [
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "run_code_sandbox",
]
