"""CNCM Code Analysis, Symbol Table, Span Extraction, and Safe Execution Sandbox."""

from cncm.code.sandbox import SandboxSecurityError, SandboxTimeoutError, run_code_sandbox
from cncm.code.symbols import (
    CodeSymbolTable,
    SymbolInfo,
    check_syntax_and_keywords,
    extract_ast_or_line_spans,
    extract_oracle_keywords,
    extract_symbols,
)

__all__ = [
    "SymbolInfo",
    "CodeSymbolTable",
    "extract_symbols",
    "extract_ast_or_line_spans",
    "extract_oracle_keywords",
    "check_syntax_and_keywords",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "run_code_sandbox",
]
