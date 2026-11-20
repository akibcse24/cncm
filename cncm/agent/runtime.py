"""Dual-Mode Agentic AI Runtime for CNCM.

Executes a single unified ``CNCMModel`` across two complementary decoding modes:

1. **Fast Parallel Concept Mode** (``generate_concept_chunk`` / ``generate_block_ar``
   with dynamic early exit): generates ``<think>...</think>`` planning and prose
   chunks in ``K`` parallel coarse-to-fine steps with zero target KV growth.
2. **Exact-AR Mode** (``generate_exact`` + ``JSONSchemaConstrainedMasker``):
   triggered automatically on ``<tool_call>`` (or explicit/intent tool trigger)
   using the exact same backbone weights, enforcing 100% schema-valid JSON up to
   ``</tool_call>``, executing the local tool, injecting ``<tool_response>...</tool_response>``
   into the causal prefix, and returning to Concept Mode.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from cncm.agent.grammar import JSONSchemaConstrainedMasker, extract_json_object
from cncm.agent.tools import ToolCallResult, ToolRegistry

_MATH_EXPR_RE = re.compile(r"\d+\s*[+\-*/^]\s*\d+")


@dataclass
class AgentStep:
    """Single recorded step in a dual-mode CNCM agent trajectory."""

    turn: int
    mode: str  # "concept" | "exact" | "tool"
    text: str
    token_ids: List[int] = field(default_factory=list)
    num_passes: int = 0
    exit_level: Optional[int] = None
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[ToolCallResult] = None
    latency_ms: float = 0.0


@dataclass
class AgentTrace:
    """Structured execution trace returned by ``CNCMAgent.run``."""

    prompt: str
    steps: List[AgentStep] = field(default_factory=list)
    mode_switches: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[ToolCallResult] = field(default_factory=list)
    concept_passes: int = 0
    exact_passes: int = 0
    concept_tokens: int = 0
    exact_tokens: int = 0
    final_response: str = ""
    total_latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Prompt: {self.prompt}",
            f"Steps: {len(self.steps)} | Mode switches: {len(self.mode_switches)} | Tool calls: {len(self.tool_calls)}",
            f"Passes: concept={self.concept_passes} ({self.concept_tokens} toks), "
            f"exact={self.exact_passes} ({self.exact_tokens} toks) | "
            f"Total latency: {self.total_latency_ms:.1f} ms",
        ]
        for idx, (tc, tr) in enumerate(zip(self.tool_calls, self.tool_results)):
            lines.append(
                f"  Tool[{idx}]: {tc.get('name')}({json.dumps(tc.get('arguments', {}))}) "
                f"-> ok={tr.ok} result={tr.output if tr.ok else tr.error}"
            )
        lines.append(f"Final response: {self.final_response[:200]}")
        return "\n".join(lines)


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    for buf in model.buffers():
        return buf.device
    return torch.device("cpu")


def _extract_new_tokens(out_ids: Tensor, prefix_len: int) -> List[int]:
    """Normalize generation outputs that may contain either new-only or prefix+new IDs."""
    seq = out_ids[0].detach().cpu().tolist() if out_ids.ndim == 2 else out_ids.detach().cpu().tolist()
    if len(seq) > prefix_len:
        # Check if output starts with prefix; if callers return only new tokens of len > prefix_len
        # we handle that at call sites with exact prefix check when possible.
        return seq
    return seq


def _strip_prefix_if_present(full_or_new: List[int], prefix: List[int]) -> List[int]:
    p = len(prefix)
    if p > 0 and len(full_or_new) >= p and full_or_new[:p] == prefix:
        return full_or_new[p:]
    return full_or_new


def _detect_intent_tool(prompt: str, registry: ToolRegistry) -> Optional[str]:
    """Lightweight deterministic edge router for explicit tool requests.

    Used only when ``enable_intent_router=True`` and the model did not organically
    emit ``<tool_call>`` on turn 0 (e.g., tiny edge checkpoints or smoke tests).
    """
    low = prompt.lower()
    names = registry.list_names()
    # Direct mention of a registered tool name
    for name in names:
        if name.lower() in low:
            return name
    # Arithmetic intent -> calculator
    if "calculator" in registry:
        if _MATH_EXPR_RE.search(low) or any(
            kw in low for kw in ("calculate", "compute", "evaluate", "multiply", "divide", "sum of")
        ):
            return "calculator"
    # Clock intent -> clock_utc
    if "clock_utc" in registry and any(
        kw in low for kw in ("utc time", "current time", "what time", "timestamp", "clock")
    ):
        return "clock_utc"
    # Text statistics intent -> text_stats
    if "text_stats" in registry and any(
        kw in low for kw in ("word count", "count words", "character count", "text stats")
    ):
        return "text_stats"
    # Key-value working memory intent -> kv_memory
    if "kv_memory" in registry and any(
        kw in low for kw in ("remember ", "save to memory", "recall ", "kv_memory", "working memory")
    ):
        return "kv_memory"
    return None


class CNCMAgent:
    """Dual-Mode Agentic AI engine built on a single unified ``CNCMModel``."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        registry: ToolRegistry,
        max_turns: int = 4,
        use_grammar_constraints: bool = True,
        quantize_kv: bool = True,
        early_exit_conf: float = 0.92,
        concept_temperature: float = 0.8,
        exact_temperature: float = 0.0,
        max_exact_tokens: int = 96,
        enable_intent_router: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry
        self.max_turns = max(1, int(max_turns))
        self.use_grammar_constraints = bool(use_grammar_constraints)
        self.quantize_kv = bool(quantize_kv)
        self.early_exit_conf = float(early_exit_conf)
        self.concept_temperature = float(concept_temperature)
        self.exact_temperature = float(exact_temperature)
        self.max_exact_tokens = max(32, int(max_exact_tokens))
        self.enable_intent_router = bool(enable_intent_router)
        self.device = device if device is not None else _infer_model_device(model)

        self.pad_id: int = int(getattr(tokenizer, "pad_token_id", 0))
        self.eos_id: int = int(getattr(tokenizer, "eos_token_id", 2))
        self.tool_call_start_id: Optional[int] = getattr(tokenizer, "tool_call_start_id", None)
        self.tool_call_end_id: Optional[int] = getattr(tokenizer, "tool_call_end_id", None)
        self.tool_resp_start_id: Optional[int] = getattr(tokenizer, "tool_resp_start_id", None)
        self.tool_resp_end_id: Optional[int] = getattr(tokenizer, "tool_resp_end_id", None)

    def _format_initial_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        sys_text = system_prompt or (
            "You are CNCM, a compact dual-mode agentic AI. "
            "Reason in <think>...</think> and invoke tools inside <tool_call>...</tool_call>."
        )
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": prompt},
        ]
        tools_schema = self.registry.to_schema_list() if len(self.registry) > 0 else None
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools_schema,
                    add_generation_prompt=True,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        tool_block = f"\n{self.registry.format_prompt_schema()}\n" if tools_schema else "\n"
        return f"System: {sys_text}{tool_block}User: {prompt}\nAssistant: "

    def _trim_prefix(self, ids: List[int], reserve_tokens: int = 128) -> List[int]:
        cfg = getattr(self.model, "cfg", None)
        max_pos = int(getattr(cfg, "max_position_embeddings", 2048)) if cfg is not None else 2048
        limit = max(32, max_pos - reserve_tokens)
        if len(ids) > limit:
            return ids[-limit:]
        return ids

    @torch.no_grad()
    def _run_concept_step(
        self,
        prefix_ids: List[int],
        max_concept_tokens: Optional[int] = None,
    ) -> Tuple[List[int], int, int, float]:
        """Run one parallel Concept Mode turn (chunk or multi-chunk Block-AR)."""
        cfg = getattr(self.model, "cfg", None)
        chunk_size = int(getattr(cfg, "chunk_size", 64)) if cfg is not None else 64
        num_levels = len(getattr(cfg, "levels", (1, 2, 4, 8, 12, 20, 36, 64))) if cfg is not None else 8

        trimmed = self._trim_prefix(prefix_ids, reserve_tokens=max(chunk_size, max_concept_tokens or chunk_size) + 32)
        prefix_tensor = torch.tensor([trimmed], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        if max_concept_tokens is not None and max_concept_tokens > chunk_size and hasattr(
            self.model, "generate_block_ar"
        ):
            out = self.model.generate_block_ar(
                prefix_tensor,
                max_new_tokens=max_concept_tokens,
                temperature=self.concept_temperature,
                early_exit_conf=self.early_exit_conf,
                quantize_kv=self.quantize_kv,
            )
            raw_seq = out["token_ids"][0].detach().cpu().tolist()
            new_ids = _strip_prefix_if_present(raw_seq, trimmed)
            chunks = int(out.get("chunks", max(1, (len(new_ids) + chunk_size - 1) // chunk_size)))
            avg_exit = float(out.get("avg_exit_level", num_levels))
            passes = max(1, int(round(chunks * avg_exit)))
            exit_level = int(round(avg_exit))
        else:
            out = self.model.generate_concept_chunk(
                prefix_tensor,
                temperature=self.concept_temperature,
                early_exit_conf=self.early_exit_conf,
                quantize_kv=self.quantize_kv,
                return_codes=False,
            )
            raw_seq = out["token_ids"][0].detach().cpu().tolist()
            new_ids = _strip_prefix_if_present(raw_seq, trimmed)
            exit_level = int(out.get("exit_level", num_levels))
            # Handle 0-based vs 1-based exit_level reporting cleanly
            passes = exit_level if exit_level >= 1 else num_levels
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return new_ids, passes, exit_level, latency_ms

    @torch.no_grad()
    def generate_tool_call_exact(
        self,
        prefix_ids: Sequence[int],
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[int], Optional[Dict[str, Any]], ToolCallResult, float]:
        """Run Exact-AR Mode with optional JSON-schema logit masking and execute the tool."""
        prefix_list = list(prefix_ids)
        if self.tool_call_start_id is not None and (
            not prefix_list or prefix_list[-1] != self.tool_call_start_id
        ):
            prefix_list.append(self.tool_call_start_id)

        trimmed = self._trim_prefix(prefix_list, reserve_tokens=self.max_exact_tokens + 16)
        prefix_tensor = torch.tensor([trimmed], dtype=torch.long, device=self.device)

        stop_ids = [self.eos_id]
        if self.tool_call_end_id is not None:
            stop_ids.insert(0, self.tool_call_end_id)

        masker: Optional[JSONSchemaConstrainedMasker] = None
        if self.use_grammar_constraints and len(self.registry) > 0:
            masker = JSONSchemaConstrainedMasker(
                tokenizer=self.tokenizer,
                registry=self.registry,
                allowed_tools=allowed_tools,
                stop_token_id=self.tool_call_end_id if self.tool_call_end_id is not None else self.eos_id,
                max_tokens=self.max_exact_tokens,
                prefix_len=len(trimmed),
            )

        t0 = time.perf_counter()
        exact_out = self.model.generate_exact(
            prefix_tensor,
            max_new_tokens=self.max_exact_tokens,
            temperature=self.exact_temperature,
            stop_token_ids=stop_ids,
            logit_mask_fn=masker,
            past_key_values=None,
            quantize_kv=self.quantize_kv,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_exact = exact_out["token_ids"][0].detach().cpu().tolist()
        new_exact_ids = _strip_prefix_if_present(raw_exact, trimmed)

        # Strip stop token from JSON text payload before parsing
        payload_ids = [
            tid
            for tid in new_exact_ids
            if tid not in (self.tool_call_end_id, self.eos_id, self.pad_id)
        ]
        raw_text = self.tokenizer.decode(payload_ids, skip_special_tokens=True)
        parsed = extract_json_object(raw_text)
        if parsed is not None:
            tool_name = str(parsed.get("name", ""))
            tool_args = parsed.get("arguments", {})
            tool_result = self.registry.execute(tool_name, tool_args)
        else:
            tool_result = self.registry.execute_json(raw_text)

        return raw_text, new_exact_ids, parsed, tool_result, latency_ms

    def _encode_tool_response(self, result: ToolCallResult) -> List[int]:
        body_ids = self.tokenizer.encode(result.to_json(), add_special_tokens=False)
        tokens: List[int] = []
        if self.tool_resp_start_id is not None:
            tokens.append(self.tool_resp_start_id)
        tokens.extend(body_ids)
        if self.tool_resp_end_id is not None:
            tokens.append(self.tool_resp_end_id)
        return tokens

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        force_tool: Optional[str] = None,
        max_concept_tokens: Optional[int] = None,
    ) -> AgentTrace:
        """Execute a full Dual-Mode Agentic loop (Concept <-> Exact-AR <-> Tool)."""
        was_training = self.model.training
        self.model.eval()
        t_start = time.perf_counter()

        trace = AgentTrace(prompt=prompt)
        formatted = self._format_initial_prompt(prompt, system_prompt=system_prompt)
        history_ids: List[int] = list(self.tokenizer.encode(formatted, add_special_tokens=False))
        if not history_ids:
            bos = int(getattr(self.tokenizer, "bos_token_id", 1))
            history_ids = [bos]

        try:
            for turn in range(self.max_turns):
                # ----------------------------------------------------------
                # 1. Parallel Concept Mode (<think> / prose planning)
                # ----------------------------------------------------------
                chunk_ids, passes, exit_level, concept_ms = self._run_concept_step(
                    history_ids,
                    max_concept_tokens=max_concept_tokens,
                )
                # Trim at EOS if present before any tool call
                clean_chunk: List[int] = []
                saw_eos = False
                saw_tool_start = False
                for tid in chunk_ids:
                    if tid == self.pad_id:
                        continue
                    if self.tool_call_start_id is not None and tid == self.tool_call_start_id:
                        saw_tool_start = True
                        break
                    if tid == self.eos_id:
                        saw_eos = True
                        break
                    clean_chunk.append(tid)

                chunk_text = self.tokenizer.decode(clean_chunk, skip_special_tokens=False)
                if not saw_tool_start and "<tool_call>" in chunk_text:
                    saw_tool_start = True
                    before_tag = chunk_text.split("<tool_call>", 1)[0]
                    clean_chunk = list(self.tokenizer.encode(before_tag, add_special_tokens=False))
                    chunk_text = before_tag

                trace.concept_passes += passes
                trace.concept_tokens += len(chunk_ids)
                trace.steps.append(
                    AgentStep(
                        turn=turn,
                        mode="concept",
                        text=chunk_text,
                        token_ids=clean_chunk,
                        num_passes=passes,
                        exit_level=exit_level,
                        latency_ms=concept_ms,
                    )
                )
                history_ids.extend(clean_chunk)

                # Decide whether to switch to Exact-AR Mode for a tool call
                trigger_reason: Optional[str] = None
                allowed_tools: Optional[List[str]] = None
                if saw_tool_start and len(self.registry) > 0:
                    trigger_reason = "tool_call_token"
                elif turn == 0 and force_tool is not None and len(self.registry) > 0:
                    trigger_reason = f"force_tool:{force_tool}"
                    if force_tool != "auto" and force_tool in self.registry:
                        allowed_tools = [force_tool]
                elif (
                    turn == 0
                    and self.enable_intent_router
                    and not trace.tool_calls
                    and len(self.registry) > 0
                ):
                    routed = _detect_intent_tool(prompt, self.registry)
                    if routed is not None:
                        trigger_reason = f"intent_router:{routed}"
                        allowed_tools = [routed]

                if trigger_reason is None:
                    trace.final_response = chunk_text.strip()
                    break

                # ----------------------------------------------------------
                # 2. Switch to Exact-AR Mode (JSON-Schema Constrained)
                # ----------------------------------------------------------
                trace.mode_switches.append(
                    {
                        "turn": turn,
                        "from": "concept",
                        "to": "exact",
                        "reason": trigger_reason,
                    }
                )
                if self.tool_call_start_id is not None:
                    history_ids.append(self.tool_call_start_id)

                raw_json, exact_ids, parsed_call, tool_result, exact_ms = (
                    self.generate_tool_call_exact(
                        history_ids,
                        allowed_tools=allowed_tools,
                    )
                )
                trace.exact_passes += len(exact_ids)
                trace.exact_tokens += len(exact_ids)
                trace.steps.append(
                    AgentStep(
                        turn=turn,
                        mode="exact",
                        text=raw_json,
                        token_ids=exact_ids,
                        num_passes=len(exact_ids),
                        tool_call=parsed_call,
                        latency_ms=exact_ms,
                    )
                )
                history_ids.extend(exact_ids)
                if self.tool_call_end_id is not None and (
                    not exact_ids or exact_ids[-1] != self.tool_call_end_id
                ):
                    history_ids.append(self.tool_call_end_id)

                # ----------------------------------------------------------
                # 3. Execute Tool & Append <tool_response> to Causal Prefix
                # ----------------------------------------------------------
                call_record = parsed_call if parsed_call is not None else {
                    "name": tool_result.name,
                    "arguments": tool_result.arguments,
                    "raw": raw_json,
                }
                trace.tool_calls.append(call_record)
                trace.tool_results.append(tool_result)
                trace.steps.append(
                    AgentStep(
                        turn=turn,
                        mode="tool",
                        text=tool_result.to_json(),
                        tool_call=call_record,
                        tool_result=tool_result,
                    )
                )

                resp_tokens = self._encode_tool_response(tool_result)
                history_ids.extend(resp_tokens)
                trace.mode_switches.append(
                    {
                        "turn": turn,
                        "from": "exact",
                        "to": "concept",
                        "reason": "tool_response_injected",
                    }
                )

                # If this was the final allowed turn, synthesize response summary
                if turn == self.max_turns - 1:
                    trace.final_response = tool_result.to_json()
                elif saw_eos and not saw_tool_start:
                    # Continue one more turn so Concept Mode reads <tool_response>
                    continue

            if not trace.final_response and trace.steps:
                trace.final_response = trace.steps[-1].text.strip()
        finally:
            if was_training:
                self.model.train()

        trace.total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        return trace


__all__ = [
    "AgentStep",
    "AgentTrace",
    "CNCMAgent",
]
