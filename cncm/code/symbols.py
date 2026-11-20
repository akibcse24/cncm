"""Zero-mandatory-C-binary code symbol table, AST span splitter, and syntax/keyword verifier.

Supports PG-FIM (Plan-Guided Fill-in-the-Middle) across edge and cloud targets:
- Python AST + lexical analysis (`ast` + `tokenize` + `builtins` + `keyword`) with
  automatic line/regex recovery on incomplete prefix/suffix snippets.
- Optional `tree_sitter` hook when installed, plus a zero-dependency balanced-bracket
  and indentation fallback for non-Python languages on Android / iGPU laptops.
- `extract_ast_or_line_spans`: implements the 40% AST-node / 30% whole-line / 30% random
  character span sampling strategy from PG-FIM Section 6.
- `extract_oracle_keywords`: extracts up to `max_kw` (0–3) priority identifiers, control
  keywords, and literals from the middle span (or salient terms for prose).
- `check_syntax_and_keywords`: verifies whether `prefix + middle + suffix` parses,
  whether all planned keywords appear in `middle`, whether `middle` is non-trivial,
  and flags undefined identifiers.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import keyword
import random
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGN_RE = re.compile(r"(?:^|[;\n])\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:[+\-*/%&|^]?=)(?!=)")
_FOR_VAR_RE = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s+in\b")
_IMPORT_RE = re.compile(r"\b(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)")
_FUNC_HEADER_RE = re.compile(r"\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)")

_PYTHON_BUILTINS: FrozenSet[str] = frozenset(dir(builtins))
_PYTHON_KEYWORDS: FrozenSet[str] = frozenset(keyword.kwlist)
_PYTHON_BUILTINS_AND_KEYWORDS: FrozenSet[str] = _PYTHON_BUILTINS | _PYTHON_KEYWORDS

_CONTROL_PRIORITY: Tuple[str, ...] = (
    "return",
    "for",
    "while",
    "if",
    "elif",
    "else",
    "yield",
    "raise",
    "break",
    "continue",
    "assert",
    "try",
    "except",
    "with",
)

_GENERIC_CODE_KEYWORDS: FrozenSet[str] = _PYTHON_KEYWORDS | frozenset(
    {
        "const",
        "let",
        "var",
        "function",
        "struct",
        "enum",
        "impl",
        "fn",
        "pub",
        "mut",
        "match",
        "switch",
        "case",
        "default",
        "void",
        "int",
        "float",
        "double",
        "bool",
        "char",
        "string",
        "new",
        "this",
        "super",
        "null",
        "nullptr",
        "true",
        "false",
    }
)

_PROSE_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
        "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "should", "now", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "of",
        "that", "this", "these", "those", "it", "its", "as", "we", "you",
        "they", "he", "she", "his", "her", "their", "our",
    }
)


@dataclass
class SymbolInfo:
    """Extracted identifier and keyword symbol table for a code context (Prefix + Suffix)."""

    defined_identifiers: Set[str] = field(default_factory=set)
    used_identifiers: Set[str] = field(default_factory=set)
    all_identifiers: Set[str] = field(default_factory=set)
    builtins_and_keywords: Set[str] = field(default_factory=set)
    language: str = "python"

    def is_known(self, identifier: str) -> bool:
        return (
            identifier in self.all_identifiers
            or identifier in self.builtins_and_keywords
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defined_identifiers": sorted(self.defined_identifiers),
            "used_identifiers": sorted(self.used_identifiers),
            "all_identifiers": sorted(self.all_identifiers),
            "num_builtins_and_keywords": len(self.builtins_and_keywords),
            "language": self.language,
        }


def _collect_ast_symbols(tree: ast.AST, defined: Set[str], used: Set[str]) -> None:
    """Walk a parsed Python AST and populate `defined` and `used` identifier sets."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in (
                    list(node.args.posonlyargs)
                    + list(node.args.args)
                    + list(node.args.kwonlyargs)
                ):
                    defined.add(arg.arg)
                if node.args.vararg is not None:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Param)):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(str(node.name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name.split(".")[0]
                if bound and bound != "*":
                    defined.add(bound)


def _collect_lexical_symbols(text: str, defined: Set[str], used: Set[str], kw_set: Set[str]) -> None:
    """Fallback regex + tokenizer symbol extraction for partial/unparsable code snippets."""
    if not text:
        return
    for match in _DEF_RE.finditer(text):
        defined.add(match.group(1))
    for match in _ASSIGN_RE.finditer(text):
        ident = match.group(1)
        if ident not in kw_set:
            defined.add(ident)
    for match in _FOR_VAR_RE.finditer(text):
        for part in match.group(1).split(","):
            v = part.strip()
            if v and v not in kw_set:
                defined.add(v)
    for match in _IMPORT_RE.finditer(text):
        ident = match.group(1)
        if ident not in kw_set:
            defined.add(ident)
    for match in _FUNC_HEADER_RE.finditer(text):
        params_raw = match.group(1)
        for param_chunk in params_raw.split(","):
            head = param_chunk.split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
            if head and head.isidentifier() and head not in kw_set:
                defined.add(head)

    for match in _IDENT_RE.finditer(text):
        ident = match.group(0)
        if ident not in kw_set:
            used.add(ident)


class CodeSymbolTable:
    """Zero-mandatory-C-binary symbol extractor and syntax checker for PG-FIM."""

    def __init__(self, default_language: str = "python") -> None:
        self.default_language = default_language.lower()
        self.has_tree_sitter = importlib.util.find_spec("tree_sitter") is not None

    def extract_symbols(
        self,
        prefix: str,
        suffix: str = "",
        language: Optional[str] = None,
    ) -> SymbolInfo:
        lang = (language or self.default_language).strip().lower()
        defined: Set[str] = set()
        used: Set[str] = set()

        if lang == "python":
            builtins_kw = set(_PYTHON_BUILTINS_AND_KEYWORDS)
            kw_only = set(_PYTHON_KEYWORDS)
            # Attempt full or stitched AST parse first
            for candidate in (f"{prefix}\npass\n{suffix}", prefix, suffix):
                if not candidate.strip():
                    continue
                try:
                    tree = ast.parse(candidate)
                    _collect_ast_symbols(tree, defined, used)
                except SyntaxError:
                    pass
            # Always supplement with lexical pass so mid-line cuts never drop symbols
            _collect_lexical_symbols(prefix, defined, used, kw_only)
            _collect_lexical_symbols(suffix, defined, used, kw_only)
        else:
            builtins_kw = set(_GENERIC_CODE_KEYWORDS)
            _collect_lexical_symbols(prefix, defined, used, builtins_kw)
            _collect_lexical_symbols(suffix, defined, used, builtins_kw)

        defined -= set(_PYTHON_KEYWORDS)
        all_ids = (defined | used) - set(_PYTHON_KEYWORDS)
        return SymbolInfo(
            defined_identifiers=defined,
            used_identifiers=used,
            all_identifiers=all_ids,
            builtins_and_keywords=builtins_kw,
            language=lang,
        )


_DEFAULT_TABLE = CodeSymbolTable()


def extract_symbols(
    prefix: str,
    suffix: str = "",
    language: str = "python",
) -> SymbolInfo:
    """Module-level convenience wrapper around ``CodeSymbolTable.extract_symbols``."""
    return _DEFAULT_TABLE.extract_symbols(prefix=prefix, suffix=suffix, language=language)


# ---------------------------------------------------------------------------
# AST / Line / Random Character Span Splitter (PG-FIM Section 6)
# ---------------------------------------------------------------------------

def _line_offsets(code: str) -> List[int]:
    """Return character start offsets for each 1-based line in ``code``."""
    offsets = [0]
    for idx, ch in enumerate(code):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def _ast_candidate_spans(code: str, mode_hint: str = "ast") -> List[Tuple[int, int]]:
    """Collect character `(start, end)` spans of interior AST statements/bodies."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    lines = code.splitlines(keepends=True)
    if not lines:
        return []
    offsets = _line_offsets(code)
    n_lines = len(lines)

    body_spans: List[Tuple[int, int]] = []
    stmt_spans: List[Tuple[int, int]] = []

    target_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.If,
        ast.For,
        ast.While,
        ast.With,
        ast.Try,
        ast.Return,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Expr,
    )

    for node in ast.walk(tree):
        if not isinstance(node, target_types):
            continue
        # For function/control blocks, extracting the interior body is ideal for FIM
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.For, ast.While)) and getattr(
            node, "body", None
        ):
            first_stmt = node.body[0]
            last_stmt = node.body[-1]
            s_line = getattr(first_stmt, "lineno", None)
            e_line = getattr(last_stmt, "end_lineno", getattr(last_stmt, "lineno", None))
            if s_line is not None and e_line is not None and 1 <= s_line <= e_line <= n_lines:
                start_char = offsets[s_line - 1]
                end_char = offsets[e_line - 1] + len(lines[e_line - 1].rstrip("\r\n"))
                if 0 < start_char < end_char < len(code):
                    body_spans.append((start_char, end_char))

        s_line = getattr(node, "lineno", None)
        e_line = getattr(node, "end_lineno", s_line)
        s_col = getattr(node, "col_offset", 0)
        e_col = getattr(node, "end_col_offset", None)
        if s_line is not None and e_line is not None and 1 <= s_line <= e_line <= n_lines:
            start_char = offsets[s_line - 1] + s_col
            if e_col is not None:
                end_char = offsets[e_line - 1] + e_col
            else:
                end_char = offsets[e_line - 1] + len(lines[e_line - 1].rstrip("\r\n"))
            if 0 < start_char < end_char < len(code) and (end_char - start_char) < len(code) * 0.85:
                stmt_spans.append((start_char, end_char))

    if mode_hint == "function_body" and body_spans:
        return body_spans
    return body_spans + stmt_spans


def _line_candidate_spans(code: str, max_span_lines: int = 3) -> List[Tuple[int, int]]:
    lines = code.splitlines(keepends=True)
    if len(lines) < 2:
        return []
    offsets = _line_offsets(code)
    candidates: List[Tuple[int, int]] = []
    # Prefer interior lines (skip line 0 header when possible)
    start_indices = list(range(1, len(lines))) if len(lines) > 2 else list(range(len(lines)))
    for i in start_indices:
        if not lines[i].strip():
            continue
        for span_len in range(1, min(max_span_lines, len(lines) - i) + 1):
            j = i + span_len - 1
            s_char = offsets[i]
            e_char = offsets[j] + len(lines[j].rstrip("\r\n"))
            if 0 <= s_char < e_char <= len(code) and (e_char - s_char) < len(code):
                candidates.append((s_char, e_char))
    return candidates


def extract_ast_or_line_spans(
    code: str,
    mode: str = "mix",
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, str]:
    """Split ``code`` into ``(prefix, middle, suffix)`` with ``prefix + middle + suffix == code``.

    Implements the Section 6 span strategy:
      - ``mode="mix"``: 40% AST-node spans, 30% whole lines, 30% random character spans.
      - ``mode="ast"`` / ``"function_body"``: AST statement or function-body spans.
      - ``mode="line"`` / ``"single_line"`` / ``"multi_line"``: 1 or 1..3 whole lines.
      - ``mode="char"`` / ``"random"``: random character span.
    """
    if not code:
        return "", "", ""
    r = rng if rng is not None else random.Random(42)
    m = mode.strip().lower()

    if m == "mix":
        roll = r.random()
        if roll < 0.40:
            chosen_mode = "ast"
        elif roll < 0.70:
            chosen_mode = "line"
        else:
            chosen_mode = "char"
    else:
        chosen_mode = m

    if chosen_mode in ("ast", "function_body"):
        spans = _ast_candidate_spans(code, mode_hint=chosen_mode)
        if spans:
            s, e = r.choice(spans)
            return code[:s], code[s:e], code[e:]
        chosen_mode = "line"

    if chosen_mode in ("line", "single_line", "multi_line"):
        max_l = 1 if chosen_mode == "single_line" else 3
        spans = _line_candidate_spans(code, max_span_lines=max_l)
        if spans:
            if chosen_mode == "multi_line":
                multi = [sp for sp in spans if "\n" in code[sp[0] : sp[1]]]
                if multi:
                    spans = multi
            s, e = r.choice(spans)
            return code[:s], code[s:e], code[e:]

    # Fallback / character-span mode
    n = len(code)
    if n <= 2:
        mid = max(1, n // 2)
        return code[:mid], code[mid:], ""
    min_len = max(1, min(8, n // 4))
    max_len = max(min_len, int(n * 0.5))
    span_len = r.randint(min_len, max_len)
    start = r.randint(1 if n > span_len + 1 else 0, max(0, n - span_len - (1 if n > span_len + 1 else 0)))
    end = min(n, start + span_len)
    # Ensure non-whitespace middle when possible
    if not code[start:end].strip():
        non_ws = [i for i, ch in enumerate(code) if not ch.isspace()]
        if non_ws:
            start = non_ws[len(non_ws) // 2]
            end = min(n, start + max(1, min_len))
    return code[:start], code[start:end], code[end:]


# ---------------------------------------------------------------------------
# Oracle Keyword Extraction (PG-FIM Section 4.2 & Section 6)
# ---------------------------------------------------------------------------

def extract_oracle_keywords(
    middle: str,
    prefix: str = "",
    suffix: str = "",
    max_kw: int = 3,
    is_code: bool = True,
) -> List[str]:
    """Extract up to ``max_kw`` (0–3) priority keywords anchoring ``middle``.

    For code:
      1. Identifiers defined or used in ``middle`` (prioritizing those that also
         appear in ``prefix`` or ``suffix`` to pin down exact symbol names).
      2. High-signal control-flow keywords in ``middle`` (``return``, ``for``, ``if``, ...).
      3. Key numeric literals in ``middle``.
    For prose:
      Salient entities, numbers, and high-specificity content words.
    """
    if max_kw <= 0 or not middle or not middle.strip():
        return []

    keywords: List[str] = []
    seen: Set[str] = set()

    def _add(tok: str) -> None:
        clean = tok.strip().replace("|", "")
        if clean and len(clean) <= 24 and clean not in seen:
            seen.add(clean)
            keywords.append(clean)

    if is_code:
        ctx_symbols = extract_symbols(prefix, suffix, language="python")
        mid_defined: Set[str] = set()
        mid_used: Set[str] = set()
        _collect_lexical_symbols(middle, mid_defined, mid_used, set(_PYTHON_KEYWORDS))

        # Preserve order of appearance in `middle`
        ordered_idents = [
            m.group(0)
            for m in _IDENT_RE.finditer(middle)
            if m.group(0) not in _PYTHON_KEYWORDS
        ]

        # 1a. Identifiers defined in middle or shared with prefix/suffix symbol table (non-builtins first)
        for ident in ordered_idents:
            if ident in _PYTHON_BUILTINS:
                continue
            if ident in mid_defined or ident in ctx_symbols.all_identifiers:
                _add(ident)
                if len(keywords) >= max_kw:
                    return keywords

        # 1b. Other non-builtin identifiers in middle
        for ident in ordered_idents:
            if ident not in _PYTHON_BUILTINS:
                _add(ident)
                if len(keywords) >= max_kw:
                    return keywords

        # 2. Control-flow keywords present in middle
        mid_tokens = set(_IDENT_RE.findall(middle))
        for ctrl in _CONTROL_PRIORITY:
            if ctrl in mid_tokens:
                _add(ctrl)
                if len(keywords) >= max_kw:
                    return keywords

        # 3. Builtin calls or numeric literals in middle
        for ident in ordered_idents:
            _add(ident)
            if len(keywords) >= max_kw:
                return keywords
        for num in _NUMBER_RE.findall(middle):
            _add(num)
            if len(keywords) >= max_kw:
                return keywords
    else:
        # Prose: capitalized entities, numbers, and salient non-stopwords
        words = re.findall(r"\b[A-Za-z0-9_]+\b", middle)
        ctx_lower = set(re.findall(r"\b[a-z0-9_]+\b", (prefix + " " + suffix).lower()))
        # Capitalized words / numbers first
        for w in words:
            if (w[0].isupper() and w.lower() not in _PROSE_STOPWORDS) or w.isdigit():
                _add(w)
                if len(keywords) >= max_kw:
                    return keywords
        # Rare/long content words not in stopwords
        scored = sorted(
            [w for w in words if w.lower() not in _PROSE_STOPWORDS and len(w) >= 4],
            key=lambda w: (w.lower() not in ctx_lower, len(w)),
            reverse=True,
        )
        for w in scored:
            _add(w)
            if len(keywords) >= max_kw:
                return keywords

    return keywords[:max_kw]


# ---------------------------------------------------------------------------
# Syntax, Non-Triviality & Keyword Faithfulness Checker (Phase A Verify)
# ---------------------------------------------------------------------------

def _is_non_trivial_middle(middle: str, min_tokens: int = 2) -> bool:
    stripped = middle.strip()
    if not stripped or stripped in ("pass", "...", "None"):
        return False
    # Ignore comment-only middles
    non_comment_lines = [
        ln.strip() for ln in stripped.splitlines() if ln.strip() and not ln.strip().startswith("#")
    ]
    if not non_comment_lines:
        return False
    joined = " ".join(non_comment_lines)
    if joined in ("pass", "..."):
        return False
    lex_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[+\-*/%=<>!&|^~]+|[()\[\]{}:.,]", joined)
    return len(lex_tokens) >= min_tokens


def _check_balanced_brackets(text: str) -> Tuple[bool, Optional[str]]:
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    stack: List[str] = []
    in_quote: Optional[str] = None
    escape = False
    for ch in text:
        if in_quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
        elif ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False, f"Unbalanced closing bracket '{ch}'"
            stack.pop()
    if in_quote is not None:
        return False, f"Unclosed string literal ({in_quote})"
    if stack:
        return False, f"Unclosed opening bracket '{stack[-1]}'"
    return True, None


def _find_undefined_in_middle(full_tree: ast.AST, middle: str) -> List[str]:
    """Identify load identifiers used in `middle` that are never defined in `full_tree` or builtins."""
    defined: Set[str] = set()
    used: Set[str] = set()
    _collect_ast_symbols(full_tree, defined, used)

    # Also include comprehension targets & generator vars
    for node in ast.walk(full_tree):
        if isinstance(node, ast.comprehension):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for sub in ast.walk(node.optional_vars):
                if isinstance(sub, ast.Name):
                    defined.add(sub.id)

    mid_idents = [
        m.group(0)
        for m in _IDENT_RE.finditer(middle)
        if m.group(0) not in _PYTHON_BUILTINS_AND_KEYWORDS
    ]
    # Only flag identifiers that are loaded in full_tree and never defined anywhere in full_tree
    undef: List[str] = []
    seen: Set[str] = set()
    for ident in mid_idents:
        if ident in used and ident not in defined and ident not in seen:
            seen.add(ident)
            undef.append(ident)
    return undef


def check_syntax_and_keywords(
    prefix: str,
    middle: str,
    suffix: str,
    keywords: Sequence[str] = (),
    language: str = "python",
    min_tokens: int = 2,
) -> Dict[str, Any]:
    """Verify syntax of ``prefix + middle + suffix``, keyword presence, and non-triviality."""
    full_code = f"{prefix}{middle}{suffix}"
    lang = language.strip().lower()
    parses = False
    syntax_error: Optional[str] = None
    undefined_identifiers: List[str] = []

    if lang == "python":
        try:
            # Verify token stream + AST parse
            list(tokenize.generate_tokens(io.StringIO(full_code).readline))
            tree = ast.parse(full_code)
            parses = True
            undefined_identifiers = _find_undefined_in_middle(tree, middle)
        except (SyntaxError, tokenize.TokenError, IndentationError) as exc:
            parses = False
            syntax_error = str(exc)
    else:
        parses, syntax_error = _check_balanced_brackets(full_code)

    clean_kws = [k.strip() for k in keywords if k and k.strip()]
    missing = [k for k in clean_kws if k not in middle]
    keywords_present = len(missing) == 0
    non_trivial = _is_non_trivial_middle(middle, min_tokens=min_tokens)

    return {
        "parses": parses,
        "keywords_present": keywords_present,
        "missing_keywords": missing,
        "syntax_error": syntax_error,
        "undefined_identifiers": undefined_identifiers,
        "non_trivial": non_trivial,
    }


__all__ = [
    "SymbolInfo",
    "CodeSymbolTable",
    "extract_symbols",
    "extract_ast_or_line_spans",
    "extract_oracle_keywords",
    "check_syntax_and_keywords",
]
