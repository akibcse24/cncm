"""Deterministic JSON-Schema logit masker for CNCM Exact-AR Mode.

Guarantees that even a tiny (0.5M–15M parameter) model running on low-end
mobile hardware NEVER emits malformed JSON inside ``<tool_call>...</tool_call>``
blocks. Works with any byte-level, character-level, or subword BPE tokenizer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch import Tensor

from cncm.agent.tools import ToolRegistry, ToolSpec

_INVALID = 0
_PREFIX = 1
_COMPLETE = 2

_MATH_OPS = frozenset("+-*/")
_DIGITS = frozenset("0123456789")
_NONZERO_DIGITS = frozenset("123456789")
_PROMPT_MATH_RE = re.compile(r"\b\d+(?:\s*[+\-*/]\s*\d+)+\b")


def extract_prompt_math_spans(prompt_text: str) -> List[str]:
    """Extract normalized arithmetic expressions (e.g. '17*6+9') from a user prompt."""
    if not prompt_text:
        return []
    seen: Set[str] = set()
    spans: List[str] = []
    for match in _PROMPT_MATH_RE.findall(prompt_text):
        norm = re.sub(r"\s+", "", match)
        if norm and norm not in seen:
            seen.add(norm)
            spans.append(norm)
    return spans


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first balanced JSON object from ``text``."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:  # noqa: BLE001
                    return None
    return None


class _ToolCallPrefixRecognizer:
    """Recursive-descent prefix recognizer for schema-bound tool-call JSON.

    Target grammar::

        {"name":"<registered_tool>","arguments":{<schema_key>:<typed_val>,...}}

    Optional single space after ``:`` or ``,`` is allowed when ``allow_spaces=True``.
    """

    def __init__(
        self,
        tools: Dict[str, ToolSpec],
        max_string_len: int = 24,
        max_math_len: int = 8,
        max_number_len: int = 8,
        allow_spaces: bool = False,
        force_close_optional: bool = False,
        allowed_math_expressions: Optional[Sequence[str]] = None,
    ) -> None:
        self.tools = tools
        self.tool_names = list(tools.keys())
        self.max_string_len = max(1, max_string_len)
        self.max_math_len = max(1, max_math_len)
        self.max_number_len = max(1, max_number_len)
        self.allow_spaces = allow_spaces
        self.force_close_optional = force_close_optional
        self.allowed_math_expressions: Optional[List[str]] = (
            [str(e) for e in allowed_math_expressions if str(e)]
            if allowed_math_expressions
            else None
        )

    def classify(self, text: str) -> int:
        """Return ``_INVALID``, ``_PREFIX``, or ``_COMPLETE`` for ``text``."""
        n = len(text)
        pos = 0

        def match_lit(lit: str) -> int:
            nonlocal pos
            rem = n - pos
            if rem == 0:
                return _PREFIX
            take = min(rem, len(lit))
            if text[pos : pos + take] != lit[:take]:
                return _INVALID
            pos += take
            return _COMPLETE if take == len(lit) else _PREFIX

        def match_sep(ch: str) -> int:
            nonlocal pos
            if pos == n:
                return _PREFIX
            if text[pos] != ch:
                return _INVALID
            pos += 1
            if self.allow_spaces and pos < n and text[pos] == " ":
                pos += 1
            return _COMPLETE

        def match_choice_in_quotes(choices: Sequence[str]) -> Tuple[int, Optional[str]]:
            nonlocal pos
            st = match_lit('"')
            if st != _COMPLETE:
                return st, None
            start = pos
            while pos < n and text[pos] != '"':
                pos += 1
            partial = text[start:pos]
            if pos == n:
                if any(c.startswith(partial) for c in choices):
                    return _PREFIX, None
                return _INVALID, None
            # Found closing quote
            if partial not in choices:
                return _INVALID, None
            pos += 1  # consume '"'
            return _COMPLETE, partial

        def match_math_string() -> int:
            nonlocal pos
            st = match_lit('"')
            if st != _COMPLETE:
                return st
            # States inside math string:
            #   need_num: expect first digit (1-9 if prev op was '/', else 0-9)
            #   in_num:   reading digits of current operand (1..4 digits); may close '"' or emit op
            #   after_op: just saw operator +-*/, must read first digit of next operand
            state = "need_num"
            expr_len = 0
            operand_len = 0
            op_count = 0
            last_op = ""
            max_ops = 2
            max_operand_digits = 4

            while pos < n:
                ch = text[pos]
                if ch == '"':
                    if state != "in_num":
                        return _INVALID
                    pos += 1
                    return _COMPLETE

                if expr_len >= self.max_math_len:
                    return _INVALID

                if state == "need_num":
                    allowed = _NONZERO_DIGITS if last_op == "/" else _DIGITS
                    if ch not in allowed:
                        return _INVALID
                    # Avoid leading zero followed by more digits by setting operand_len appropriately
                    state = "in_num"
                    operand_len = 1 if ch != "0" else max_operand_digits
                    expr_len += 1
                    pos += 1
                elif state == "in_num":
                    if ch in _DIGITS and operand_len < max_operand_digits:
                        operand_len += 1
                        expr_len += 1
                        pos += 1
                    elif (
                        ch in _MATH_OPS
                        and op_count < max_ops
                        and expr_len + 2 <= self.max_math_len
                    ):
                        state = "need_num"
                        last_op = ch
                        op_count += 1
                        operand_len = 0
                        expr_len += 1
                        pos += 1
                    else:
                        return _INVALID
                else:
                    return _INVALID
            return _PREFIX

        def match_general_string() -> int:
            nonlocal pos
            st = match_lit('"')
            if st != _COMPLETE:
                return st
            content_len = 0
            has_non_space = False
            prev_space = False
            while pos < n:
                ch = text[pos]
                if ch == '"':
                    if not has_non_space:
                        return _INVALID
                    pos += 1
                    return _COMPLETE
                if content_len >= self.max_string_len:
                    return _INVALID
                code = ord(ch)
                if code < 0x20 or code > 0x7E or ch == "\\":
                    return _INVALID
                if ch == " ":
                    if content_len == 0 or prev_space:
                        return _INVALID
                    prev_space = True
                else:
                    has_non_space = True
                    prev_space = False
                content_len += 1
                pos += 1
            return _PREFIX

        def match_integer() -> int:
            nonlocal pos
            if pos == n:
                return _PREFIX
            digits = 0
            leading_zero = False
            if text[pos] == "-":
                pos += 1
                if pos == n:
                    return _PREFIX
            while pos < n:
                ch = text[pos]
                if ch in _DIGITS:
                    if digits == 0 and ch == "0":
                        leading_zero = True
                    elif leading_zero:
                        return _INVALID
                    digits += 1
                    if digits > self.max_number_len:
                        return _INVALID
                    pos += 1
                elif ch in (",", "}"):
                    return _COMPLETE if digits > 0 else _INVALID
                else:
                    return _INVALID
            return _PREFIX if digits == 0 else _COMPLETE

        def match_number() -> int:
            nonlocal pos
            if pos == n:
                return _PREFIX
            digits = 0
            seen_dot = False
            after_dot_digits = 0
            if text[pos] == "-":
                pos += 1
                if pos == n:
                    return _PREFIX
            while pos < n:
                ch = text[pos]
                if ch in _DIGITS:
                    digits += 1
                    if seen_dot:
                        after_dot_digits += 1
                    if digits > self.max_number_len:
                        return _INVALID
                    pos += 1
                elif ch == "." and not seen_dot and digits > 0:
                    seen_dot = True
                    pos += 1
                elif ch in (",", "}"):
                    if digits == 0 or (seen_dot and after_dot_digits == 0):
                        return _INVALID
                    return _COMPLETE
                else:
                    return _INVALID
            if digits == 0 or (seen_dot and after_dot_digits == 0):
                return _PREFIX
            return _COMPLETE

        def match_boolean() -> int:
            nonlocal pos
            if pos == n:
                return _PREFIX
            for lit in ("true", "false"):
                rem = n - pos
                take = min(rem, len(lit))
                if text[pos : pos + take] == lit[:take]:
                    pos += take
                    return _COMPLETE if take == len(lit) else _PREFIX
            return _INVALID

        def match_value(prop_schema: Dict[str, Any]) -> int:
            if "enum" in prop_schema:
                enum_vals = prop_schema["enum"]
                if all(isinstance(v, str) for v in enum_vals):
                    st, _ = match_choice_in_quotes(enum_vals)
                    return st
                lits = sorted([json.dumps(v) for v in enum_vals], key=len, reverse=True)
                rem = n - pos
                if rem == 0:
                    return _PREFIX
                for lit in lits:
                    take = min(rem, len(lit))
                    if text[pos : pos + take] == lit[:take]:
                        if take < len(lit):
                            pos += take
                            return _PREFIX
                        # Ensure complete match ends at end-of-string or delimiter
                        if pos + len(lit) == n or text[pos + len(lit)] in (",", "}"):
                            pos += len(lit)
                            return _COMPLETE
                return _INVALID

            typ = prop_schema.get("type", "string")
            fmt = prop_schema.get("format", "")
            if typ == "string":
                if fmt == "math_expression":
                    if self.allowed_math_expressions:
                        st, _ = match_choice_in_quotes(self.allowed_math_expressions)
                        return st
                    return match_math_string()
                return match_general_string()
            if typ == "integer":
                return match_integer()
            if typ == "number":
                return match_number()
            if typ == "boolean":
                return match_boolean()
            return match_general_string()

        # 1. {"name":"<tool>"
        st = match_lit('{"name"')
        if st != _COMPLETE:
            return st
        st = match_sep(":")
        if st != _COMPLETE:
            return st
        st, tool_name = match_choice_in_quotes(self.tool_names)
        if st != _COMPLETE:
            return st
        assert tool_name is not None
        spec = self.tools[tool_name]

        # 2. ,"arguments":{
        st = match_sep(",")
        if st != _COMPLETE:
            return st
        st = match_lit('"arguments"')
        if st != _COMPLETE:
            return st
        st = match_sep(":")
        if st != _COMPLETE:
            return st
        st = match_lit("{")
        if st != _COMPLETE:
            return st

        # 3. Arguments object keys/values
        props = spec.properties()
        required = spec.required_params()
        ordered_keys = list(required) + [k for k in props.keys() if k not in required]
        seen_keys: Set[str] = set()

        while True:
            if pos == n:
                return _PREFIX
            rem_req = [k for k in required if k not in seen_keys]
            rem_opt = [k for k in ordered_keys if k not in seen_keys and k not in required]

            if not seen_keys:
                if not rem_req and text[pos] == "}":
                    pos += 1
                    break
                if not rem_req and (self.force_close_optional or not rem_opt):
                    if text[pos] != "}":
                        return _INVALID
                    pos += 1
                    break
            else:
                if not rem_req and text[pos] == "}":
                    pos += 1
                    break
                if not rem_req and (self.force_close_optional or not rem_opt):
                    if text[pos] != "}":
                        return _INVALID
                    pos += 1
                    break
                st = match_sep(",")
                if st != _COMPLETE:
                    return st
                if pos == n:
                    return _PREFIX

            # Require mandatory keys first so token budget is never wasted before required args
            candidate_keys = rem_req if rem_req else rem_opt
            if not candidate_keys:
                return _INVALID

            st, key_name = match_choice_in_quotes(candidate_keys)
            if st != _COMPLETE:
                return st
            assert key_name is not None
            seen_keys.add(key_name)

            st = match_sep(":")
            if st != _COMPLETE:
                return st

            st = match_value(props.get(key_name, {"type": "string"}))
            if st != _COMPLETE:
                return st
            # For non-delimited JSON values (int/number/bool), hitting end-of-string
            # while matching digits still requires closing '}' later.
            if pos == n:
                return _PREFIX

        # 4. Closing root '}'
        st = match_lit("}")
        if st != _COMPLETE:
            return st
        if pos < n:
            return _INVALID
        return _COMPLETE


class JSONSchemaConstrainedMasker:
    """Token-level logit mask enforcing schema-valid JSON tool calls in Exact-AR mode.

    Callable signature matches ``CNCMModel.generate_exact(..., logit_mask_fn=masker)``::

        masked_logits = masker(generated_ids, logits)

    Parameters
    ----------
    tokenizer:
        ``CNCMTextTokenizer`` (or any tokenizer exposing ``decode`` and ``vocab_size``).
    registry:
        ``ToolRegistry`` containing allowed tools and their JSON schemas.
    allowed_tools:
        Optional subset of tool names to allow for the current call.
    stop_token_id:
        Token emitted once the JSON object is complete (defaults to
        ``tokenizer.tool_call_end_id`` or ``tokenizer.eos_token_id``).
    max_string_len:
        Maximum characters permitted inside free-form string values.
    max_math_len:
        Maximum characters permitted inside ``format="math_expression"`` strings.
    max_tokens:
        Optional token budget hint so values close cleanly before ``max_new_tokens``.
    prefix_len:
        Optional prefix length if ``generate_exact`` passes full ``prefix + generated``
        IDs into ``logit_mask_fn`` without a ``tool_call_start_id`` delimiter.
    """

    def __init__(
        self,
        tokenizer: Any,
        registry: ToolRegistry,
        allowed_tools: Optional[Iterable[str]] = None,
        stop_token_id: Optional[int] = None,
        max_string_len: int = 20,
        max_math_len: int = 8,
        max_number_len: int = 8,
        max_tokens: Optional[int] = None,
        allow_spaces: bool = True,
        prefix_len: Optional[int] = None,
        prompt_text: Optional[str] = None,
        allowed_math_expressions: Optional[Sequence[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_string_len = max_string_len
        self.max_math_len = max_math_len
        self.max_number_len = max_number_len
        self.max_tokens = max_tokens
        self.allow_spaces = allow_spaces
        self.prefix_len = prefix_len
        if allowed_math_expressions is not None:
            self.allowed_math_expressions: Optional[List[str]] = [
                str(e) for e in allowed_math_expressions if str(e)
            ] or None
        elif prompt_text is not None:
            spans = extract_prompt_math_spans(prompt_text)
            self.allowed_math_expressions = spans if spans else None
        else:
            self.allowed_math_expressions = None

        if allowed_tools is not None:
            names = [n for n in allowed_tools if n in registry]
        else:
            names = registry.list_names()
        if not names:
            raise ValueError("JSONSchemaConstrainedMasker requires at least one registered tool.")
        self.tools: Dict[str, ToolSpec] = {n: registry.get(n) for n in names if registry.get(n) is not None}

        self.vocab_size: int = int(getattr(tokenizer, "vocab_size", 4096))
        self.tool_call_start_id: Optional[int] = getattr(tokenizer, "tool_call_start_id", None)
        if stop_token_id is not None:
            self.stop_token_id = int(stop_token_id)
        else:
            tc_end = getattr(tokenizer, "tool_call_end_id", None)
            eos = getattr(tokenizer, "eos_token_id", 2)
            self.stop_token_id = int(tc_end if tc_end is not None else eos)

        self._special_ids: Set[int] = set()
        for tid in getattr(tokenizer, "special_token_ids", ()):
            if isinstance(tid, int) and 0 <= tid < self.vocab_size:
                self._special_ids.add(tid)
        for attr in (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "system_start_id",
            "user_start_id",
            "assistant_start_id",
            "turn_end_id",
            "think_start_id",
            "think_end_id",
            "tool_call_start_id",
            "tool_call_end_id",
            "tool_resp_start_id",
            "tool_resp_end_id",
            "tools_start_id",
            "tools_end_id",
        ):
            val = getattr(tokenizer, attr, None)
            if isinstance(val, int) and 0 <= val < self.vocab_size:
                self._special_ids.add(val)

        # Pre-decode vocabulary tokens once and filter to safe printable JSON candidates
        self._vocab_strs: List[str] = [""] * self.vocab_size
        self._candidate_ids: List[int] = []
        self._build_vocab_table()

        # Cache: (buf_text, force_close, eff_str_len, eff_math_len) -> list[int]
        self._transition_cache: Dict[Tuple[str, bool, int, int], List[int]] = {}

    def _build_vocab_table(self) -> None:
        for tid in range(self.vocab_size):
            if tid in self._special_ids:
                continue
            try:
                piece = self.tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:  # noqa: BLE001
                continue
            if not piece or "\ufffd" in piece:
                continue
            # Only allow printable ASCII without control chars or backslashes
            if all(0x20 <= ord(ch) <= 0x7E and ch != "\\" for ch in piece):
                self._vocab_strs[tid] = piece
                self._candidate_ids.append(tid)

    def reset(
        self,
        prefix_len: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Reset per-call tracking metadata before a new ``generate_exact`` call."""
        if prefix_len is not None:
            self.prefix_len = prefix_len
        if max_tokens is not None:
            self.max_tokens = max_tokens

    def _extract_tool_tokens(self, seq_ids: List[int]) -> List[int]:
        """Extract only the token IDs generated inside the active ``<tool_call>`` span."""
        if not seq_ids:
            return []
        if self.prefix_len is not None and len(seq_ids) >= self.prefix_len:
            start = self.prefix_len
            if self.tool_call_start_id is not None and self.tool_call_start_id in seq_ids:
                last_idx = len(seq_ids) - 1 - seq_ids[::-1].index(self.tool_call_start_id)
                start = max(start, last_idx + 1)
            return seq_ids[start:]
        # When prefix_len is not set, slice after the last <tool_call> token if present
        if self.tool_call_start_id is not None and self.tool_call_start_id in seq_ids:
            last_idx = len(seq_ids) - 1 - seq_ids[::-1].index(self.tool_call_start_id)
            return seq_ids[last_idx + 1 :]
        return seq_ids

    def allowed_token_ids(self, generated_ids: Sequence[int]) -> List[int]:
        """Return the exact list of grammar-valid next token IDs for a single sequence."""
        tool_ids = self._extract_tool_tokens(list(generated_ids))
        if tool_ids:
            buf = self.tokenizer.decode(tool_ids, skip_special_tokens=False)
        else:
            buf = ""

        # Budget-aware tightening so the JSON object always closes cleanly
        eff_str_len = self.max_string_len
        eff_math_len = self.max_math_len
        force_close = False
        if self.max_tokens is not None:
            rem = self.max_tokens - len(tool_ids)
            if rem <= 12:
                force_close = True
            if rem <= 8:
                eff_str_len = min(eff_str_len, 2)
                eff_math_len = min(eff_math_len, 2)
            elif rem <= 16:
                eff_str_len = min(eff_str_len, 6)
                eff_math_len = min(eff_math_len, 4)

        cache_key = (buf, force_close, eff_str_len, eff_math_len)
        cached = self._transition_cache.get(cache_key)
        if cached is not None:
            return cached

        recognizer = _ToolCallPrefixRecognizer(
            tools=self.tools,
            max_string_len=eff_str_len,
            max_math_len=eff_math_len,
            max_number_len=self.max_number_len,
            allow_spaces=self.allow_spaces,
            force_close_optional=force_close,
            allowed_math_expressions=self.allowed_math_expressions,
        )

        cur_status = recognizer.classify(buf)
        if cur_status == _COMPLETE:
            allowed = [self.stop_token_id]
            self._transition_cache[cache_key] = allowed
            return allowed

        allowed: List[int] = []
        for tid in self._candidate_ids:
            cand = buf + self._vocab_strs[tid]
            st = recognizer.classify(cand)
            if st != _INVALID:
                allowed.append(tid)

        # Safety fallback: never return an empty mask (would cause NaN in softmax)
        if not allowed:
            allowed = [self.stop_token_id]

        self._transition_cache[cache_key] = allowed
        return allowed

    def __call__(
        self,
        generated_ids: Union[Tensor, Sequence[int], Sequence[Sequence[int]]],
        logits: Tensor,
    ) -> Tensor:
        """Apply grammar mask to next-token ``logits``.

        Supports ``logits`` of shape ``[V]``, ``[B, V]``, or ``[B, 1, V]`` and
        ``generated_ids`` as a 1D/2D Tensor or Python list.
        """
        orig_shape = logits.shape
        if logits.ndim == 1:
            logits_2d = logits.unsqueeze(0)
        elif logits.ndim == 3 and logits.shape[1] == 1:
            logits_2d = logits.squeeze(1)
        elif logits.ndim == 2:
            logits_2d = logits
        else:
            raise ValueError(f"Expected logits of shape [V], [B, V], or [B, 1, V]; got {orig_shape}")

        batch_size, vocab_dim = logits_2d.shape

        if isinstance(generated_ids, Tensor):
            if generated_ids.ndim == 1:
                seqs = [generated_ids.detach().cpu().tolist()] * batch_size
            elif generated_ids.ndim == 2:
                seqs = generated_ids.detach().cpu().tolist()
            else:
                raise ValueError(f"Expected 1D or 2D generated_ids tensor; got {generated_ids.shape}")
        else:
            if len(generated_ids) > 0 and isinstance(generated_ids[0], (list, tuple)):
                seqs = [list(s) for s in generated_ids]  # type: ignore[arg-type]
            else:
                seqs = [list(generated_ids)] * batch_size  # type: ignore[arg-type]

        mask = torch.full(
            (batch_size, vocab_dim),
            float("-inf"),
            dtype=logits_2d.dtype,
            device=logits_2d.device,
        )
        for b in range(batch_size):
            seq_b = seqs[b] if b < len(seqs) else seqs[0]
            valid_ids = [tid for tid in self.allowed_token_ids(seq_b) if 0 <= tid < vocab_dim]
            if not valid_ids:
                valid_ids = [min(self.stop_token_id, vocab_dim - 1)]
            idx_tensor = torch.tensor(valid_ids, dtype=torch.long, device=logits_2d.device)
            mask[b, idx_tensor] = 0.0

        masked = logits_2d + mask
        return masked.view(orig_shape)


# ---------------------------------------------------------------------------
# PG-FIM Plan Grammar Masker & Plan Parser (<mlen_k> + <kw> kw1 | kw2 </kw>)
# ---------------------------------------------------------------------------

def _is_valid_kw_inner(
    cand: str,
    max_kw: int = 3,
    max_kw_chars: int = 24,
) -> Tuple[bool, bool]:
    """Validate partial inner text inside ``<kw> ... </kw>``.

    Returns ``(is_valid_prefix, can_close_with_kw_end)``.
    """
    if not cand:
        return True, False
    if any(ch in cand for ch in ("\n", "\r", "\t")):
        return False, False

    segments = cand.split("|")
    if len(segments) > max_kw:
        return False, False

    # All completed segments before the last '|' must be non-empty valid keywords/identifiers
    for seg in segments[:-1]:
        s = seg.strip()
        if not s or len(s) > max_kw_chars:
            return False, False
        if len(seg) - len(s) > 2:
            return False, False
        if any(ch.isspace() for ch in s):
            return False, False
        if not all(ch.isalnum() or ch in "_.-+" for ch in s):
            return False, False

    last_seg = segments[-1]
    if last_seg.startswith("  "):
        return False, False
    s_last = last_seg.strip()
    if len(s_last) > max_kw_chars:
        return False, False

    if not s_last:
        # Empty active segment (right after '<kw>' or '|'): allow at most single leading space
        return (len(last_seg) <= 1), False

    if last_seg.endswith("  ") or len(last_seg) - len(s_last) > 2:
        return False, False
    if any(ch.isspace() for ch in s_last):
        return False, False
    if not all(ch.isalnum() or ch in "_.-+" for ch in s_last):
        return False, False

    return True, True


def parse_plan_tokens(
    tokenizer: Any,
    plan_token_ids: Union[Tensor, Sequence[int]],
) -> Dict[str, Any]:
    """Parse PG-FIM plan tokens ``<mlen_k> (<kw/> | <kw> kw1 | kw2 </kw>)``.

    Returns
    -------
    dict with keys:
      ``mlen_id`` (Optional[int]), ``num_chunks`` (int), ``keywords`` (List[str]),
      ``is_empty_plan`` (bool), ``raw_plan_text`` (str).
    """
    if isinstance(plan_token_ids, Tensor):
        flat = (
            plan_token_ids[0].detach().cpu().tolist()
            if plan_token_ids.ndim >= 2
            else plan_token_ids.detach().cpu().tolist()
        )
        ids = [int(x) for x in flat]
    else:
        ids = [int(x) for x in plan_token_ids]

    fim_mid = getattr(tokenizer, "fim_middle_id", None)
    if isinstance(fim_mid, int) and fim_mid in ids:
        last_mid = len(ids) - 1 - ids[::-1].index(fim_mid)
        ids = ids[last_mid + 1 :]

    mlen_tuple = tuple(getattr(tokenizer, "mlen_token_ids", ()))
    mlen_id: Optional[int] = None
    num_chunks = 1
    for tid in ids:
        if tid in mlen_tuple:
            mlen_id = tid
            if hasattr(tokenizer, "chunks_for_mlen_id"):
                num_chunks = int(tokenizer.chunks_for_mlen_id(tid))
            break

    kw_start = getattr(tokenizer, "kw_start_id", None)
    kw_end = getattr(tokenizer, "kw_end_id", None)
    kw_empty = getattr(tokenizer, "kw_empty_id", None)
    kw_max = 3

    if isinstance(kw_empty, int) and kw_empty in ids:
        keywords: List[str] = []
        is_empty_plan = True
    else:
        inner_ids: List[int] = []
        started = not (isinstance(kw_start, int) and kw_start in ids)
        for tid in ids:
            if tid in mlen_tuple:
                continue
            if isinstance(kw_start, int) and tid == kw_start:
                started = True
                continue
            if isinstance(kw_end, int) and tid == kw_end:
                break
            if started:
                inner_ids.append(tid)
        raw_inner = tokenizer.decode(inner_ids, skip_special_tokens=True) if inner_ids else ""
        keywords = [k.strip() for k in raw_inner.split("|") if k.strip()][:kw_max]
        is_empty_plan = len(keywords) == 0

    return {
        "mlen_id": mlen_id,
        "num_chunks": num_chunks,
        "keywords": keywords,
        "is_empty_plan": is_empty_plan,
        "raw_plan_text": tokenizer.decode(ids, skip_special_tokens=False) if ids else "",
    }


class KeywordPlanMasker:
    """Grammar + soft symbol-table logit masker for PG-FIM Exact-AR Plan stage.

    Enforces the PG-FIM plan syntax immediately after ``<fim_middle>``:
      1. Step 0: hard-mask to ``mlen_token_ids`` (``<mlen_1>``..``<mlen_4+>``).
      2. Step 1: hard-mask to ``<kw>`` (``kw_start_id``) or ``<kw/>`` (``kw_empty_id``).
      3. Inside ``<kw> ... </kw>``:
         - Hard format rule: at most ``kw_max=3`` keywords separated by ``|``,
           closed by ``</kw>`` (``kw_end_id``) within ``max_plan_tokens``.
         - Soft identifier rule (Section 4.3): applies ``+symbol_bonus`` to tokens
           matching/prefixing identifiers in ``SymbolInfo`` (`P + S` symbol table
           or language syntax keywords/builtins), and ``-unknown_id_penalty`` (soft penalty,
           never ``-inf``) to out-of-table identifier tokens.
    """

    def __init__(
        self,
        tokenizer: Any,
        symbol_info: Optional[Any] = None,
        prefix_code: str = "",
        suffix_code: str = "",
        language: str = "python",
        kw_max: int = 3,
        max_plan_tokens: int = 16,
        max_kw_chars: int = 24,
        symbol_bonus: float = 2.5,
        unknown_id_penalty: float = 1.5,
        allow_empty_kw: bool = True,
        prefix_len: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.kw_max = max(1, int(kw_max))
        self.max_plan_tokens = max(4, int(max_plan_tokens))
        self.max_kw_chars = max(2, int(max_kw_chars))
        self.symbol_bonus = float(symbol_bonus)
        self.unknown_id_penalty = float(unknown_id_penalty)
        self.allow_empty_kw = bool(allow_empty_kw)
        self.prefix_len = prefix_len

        if symbol_info is None and (prefix_code or suffix_code):
            from cncm.code.symbols import extract_symbols

            symbol_info = extract_symbols(prefix_code, suffix_code, language=language)
        self.symbol_info = symbol_info

        self._known_symbols: Set[str] = set()
        if symbol_info is not None:
            self._known_symbols |= set(getattr(symbol_info, "all_identifiers", set()))
            self._known_symbols |= set(getattr(symbol_info, "builtins_and_keywords", set()))
        else:
            from cncm.code.symbols import _PYTHON_BUILTINS_AND_KEYWORDS

            self._known_symbols |= set(_PYTHON_BUILTINS_AND_KEYWORDS)

        self.vocab_size: int = int(getattr(tokenizer, "vocab_size", 4096))
        self.fim_middle_id: Optional[int] = getattr(tokenizer, "fim_middle_id", None)
        self.kw_start_id: int = int(getattr(tokenizer, "kw_start_id", self.vocab_size - 7))
        self.kw_end_id: int = int(getattr(tokenizer, "kw_end_id", self.vocab_size - 6))
        self.kw_empty_id: int = int(getattr(tokenizer, "kw_empty_id", self.vocab_size - 5))
        self.mlen_ids: Tuple[int, ...] = tuple(
            int(x)
            for x in getattr(
                tokenizer,
                "mlen_token_ids",
                (
                    self.vocab_size - 4,
                    self.vocab_size - 3,
                    self.vocab_size - 2,
                    self.vocab_size - 1,
                ),
            )
        )

        self._special_ids: Set[int] = set(int(t) for t in getattr(tokenizer, "special_token_ids", ()))
        self._vocab_strs: List[str] = [""] * self.vocab_size
        self._kw_candidate_ids: List[int] = []
        self._build_kw_candidates()

        # Precompute exact known-symbol prefix set for O(1) lookup (vs O(|symbols|) scan per vocab token)
        self._known_prefixes: Set[str] = set()
        for sym in self._known_symbols:
            for k in range(1, len(sym) + 1):
                self._known_prefixes.add(sym[:k])
        self._step_cache: Dict[Tuple[str, int, int], Dict[int, float]] = {}

    def _build_kw_candidates(self) -> None:
        # Cache decoded candidate pieces per (tokenizer id, vocab_size) so repeated
        # KeywordPlanMasker instantiations take <0.1 ms instead of decoding 4096 tokens each time.
        cache_attr = "_cncm_kw_cand_cache"
        cached = getattr(self.tokenizer, cache_attr, None)
        if cached is not None and cached[0] == self.vocab_size:
            self._vocab_strs, self._kw_candidate_ids = cached[1], cached[2]
            return

        vocab_strs: List[str] = [""] * self.vocab_size
        cand_ids: List[int] = []
        for tid in range(self.vocab_size):
            if tid in self._special_ids:
                continue
            try:
                piece = self.tokenizer.decode([tid], skip_special_tokens=False)
            except Exception:  # noqa: BLE001
                continue
            if not piece or "\ufffd" in piece:
                continue
            # Only allow characters that can appear inside `<kw> kw1 | kw2 </kw>`
            if all(ch.isalnum() or ch in " |_.-+" for ch in piece):
                vocab_strs[tid] = piece
                cand_ids.append(tid)

        self._vocab_strs = vocab_strs
        self._kw_candidate_ids = cand_ids
        try:
            setattr(self.tokenizer, cache_attr, (self.vocab_size, vocab_strs, cand_ids))
        except Exception:  # noqa: BLE001
            pass

    def _extract_plan_tokens(self, seq_ids: List[int]) -> List[int]:
        if not seq_ids:
            return []
        if self.prefix_len is not None and len(seq_ids) >= self.prefix_len:
            start = self.prefix_len
            if self.fim_middle_id is not None and self.fim_middle_id in seq_ids:
                last_idx = len(seq_ids) - 1 - seq_ids[::-1].index(self.fim_middle_id)
                start = max(start, last_idx + 1)
            return seq_ids[start:]
        if self.fim_middle_id is not None and self.fim_middle_id in seq_ids:
            last_idx = len(seq_ids) - 1 - seq_ids[::-1].index(self.fim_middle_id)
            return seq_ids[last_idx + 1 :]
        return seq_ids

    def _matches_known_symbol_prefix(self, partial: str) -> bool:
        if not partial:
            return False
        return partial in self._known_prefixes

    def _compute_step_bias(self, seq_ids: Sequence[int], vocab_dim: int) -> Dict[int, float]:
        """Return `{token_id: additive_bias}` for all allowed tokens at the current plan step."""
        plan_ids = self._extract_plan_tokens(list(seq_ids))

        # 1. Step 0: Must emit one of <mlen_1> .. <mlen_4+>
        if not any(t in self.mlen_ids for t in plan_ids):
            return {tid: 0.0 for tid in self.mlen_ids if 0 <= tid < vocab_dim}

        # 2. If plan already closed via <kw/> or </kw>, only allow </kw>
        if self.kw_empty_id in plan_ids or self.kw_end_id in plan_ids:
            return {self.kw_end_id: 0.0}

        # 3. Step 1 (immediately after <mlen_k>): must emit <kw> or <kw/>
        if self.kw_start_id not in plan_ids:
            step1: Dict[int, float] = {self.kw_start_id: 0.0}
            if self.allow_empty_kw and 0 <= self.kw_empty_id < vocab_dim:
                step1[self.kw_empty_id] = 0.0
            return step1

        # 4. Inside <kw> ... </kw>
        rem_tokens = self.max_plan_tokens - len(plan_ids)
        if rem_tokens <= 1:
            return {self.kw_end_id: 0.0}

        kw_pos = plan_ids.index(self.kw_start_id)
        inner_ids = plan_ids[kw_pos + 1 :]
        inner_text = self.tokenizer.decode(inner_ids, skip_special_tokens=True) if inner_ids else ""

        rem_bucket = 2 if rem_tokens <= 2 else (3 if rem_tokens == 3 else 4)
        cache_key = (inner_text, rem_bucket, vocab_dim)
        cached_biases = self._step_cache.get(cache_key)
        if cached_biases is not None:
            return cached_biases

        _, can_close_now = _is_valid_kw_inner(
            inner_text, max_kw=self.kw_max, max_kw_chars=self.max_kw_chars
        )

        # If token budget is nearly exhausted and we can close cleanly, force </kw>
        if rem_tokens <= 2 and can_close_now:
            res_end = {self.kw_end_id: 0.0}
            self._step_cache[cache_key] = res_end
            return res_end

        biases: Dict[int, float] = {}
        if can_close_now and 0 <= self.kw_end_id < vocab_dim:
            biases[self.kw_end_id] = 0.0

        for tid in self._kw_candidate_ids:
            if tid >= vocab_dim:
                continue
            piece = self._vocab_strs[tid]
            # Avoid opening a new '|' separator when remaining token budget cannot finish another keyword
            if "|" in piece and rem_tokens <= 3:
                continue
            cand = inner_text + piece
            valid_prefix, cand_can_close = _is_valid_kw_inner(
                cand, max_kw=self.kw_max, max_kw_chars=self.max_kw_chars
            )
            if not valid_prefix:
                continue
            if rem_tokens <= 2 and not cand_can_close:
                continue

            # Soft identifier bonus / penalty (PG-FIM Section 4.3)
            active_kw = cand.split("|")[-1].strip()
            bias = 0.0
            if active_kw:
                if active_kw in self._known_prefixes:
                    bias = self.symbol_bonus
                elif active_kw[0].isalpha() or active_kw[0] == "_":
                    bias = -self.unknown_id_penalty
            biases[tid] = bias

        if not biases:
            biases[self.kw_end_id] = 0.0
        self._step_cache[cache_key] = biases
        return biases

    def allowed_token_ids(self, generated_ids: Sequence[int]) -> List[int]:
        return list(self._compute_step_bias(generated_ids, self.vocab_size).keys())

    def __call__(
        self,
        generated_ids: Union[Tensor, Sequence[int], Sequence[Sequence[int]]],
        logits: Tensor,
    ) -> Tensor:
        """Apply hard PG-FIM format mask and soft symbol-table logit biases."""
        orig_shape = logits.shape
        if logits.ndim == 1:
            logits_2d = logits.unsqueeze(0)
        elif logits.ndim == 3 and logits.shape[1] == 1:
            logits_2d = logits.squeeze(1)
        elif logits.ndim == 2:
            logits_2d = logits
        else:
            raise ValueError(f"Expected logits of shape [V], [B, V], or [B, 1, V]; got {orig_shape}")

        batch_size, vocab_dim = logits_2d.shape

        if isinstance(generated_ids, Tensor):
            if generated_ids.ndim == 1:
                seqs = [generated_ids.detach().cpu().tolist()] * batch_size
            elif generated_ids.ndim == 2:
                seqs = generated_ids.detach().cpu().tolist()
            else:
                raise ValueError(f"Expected 1D or 2D generated_ids tensor; got {generated_ids.shape}")
        else:
            if len(generated_ids) > 0 and isinstance(generated_ids[0], (list, tuple)):
                seqs = [list(s) for s in generated_ids]  # type: ignore[arg-type]
            else:
                seqs = [list(generated_ids)] * batch_size  # type: ignore[arg-type]

        mask = torch.full(
            (batch_size, vocab_dim),
            float("-inf"),
            dtype=logits_2d.dtype,
            device=logits_2d.device,
        )
        for b in range(batch_size):
            seq_b = seqs[b] if b < len(seqs) else seqs[0]
            step_biases = self._compute_step_bias(seq_b, vocab_dim)
            if not step_biases:
                fallback_id = min(self.kw_end_id, vocab_dim - 1)
                mask[b, fallback_id] = 0.0
            else:
                ids_t = torch.tensor(list(step_biases.keys()), dtype=torch.long, device=logits_2d.device)
                vals_t = torch.tensor(
                    list(step_biases.values()), dtype=logits_2d.dtype, device=logits_2d.device
                )
                mask[b, ids_t] = vals_t

        return (logits_2d + mask).view(orig_shape)


__all__ = [
    "JSONSchemaConstrainedMasker",
    "KeywordPlanMasker",
    "extract_json_object",
    "parse_plan_tokens",
]
