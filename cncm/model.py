"""Stage 2 Unified Dual-Mode Concept + Exact-AR Model (`CNCMModel`).

Combines ONE compact recursive-depth Transformer backbone with TWO heads:
  1. `FactorizedConceptHead` (Concept Mode): fast parallel coarse-to-fine block-AR
     generation with causal prefix KV cache (optional INT8 compression), per-level
     reset RoPE, block-diagonal target attention, and Dynamic Early-Exit Gate.
  2. `exact_head` (Exact-AR Mode): exact token-by-token generation with optional
     grammar/JSON-schema logit masking (`logit_mask_fn`) for Agentic AI tool calls,
     lossless Self-Speculative verification (`generate_self_speculative`), and
     PG-FIM (Plan-Guided Fill-in-the-Middle: `generate_fim` & `compute_pg_fim_logprobs`).
"""

from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.config import CNCMConfig
from cncm.layers import (
    CNCMBlock,
    FactorizedConceptHead,
    LayerKV,
    LoopAdapter,
    RMSNorm,
    RotaryEmbedding,
    build_cached_concept_step_mask,
    build_causal_mask,
    build_concept_mask,
    get_kv_seq_len,
    make_linear,
    truncate_kv_cache,
)
from cncm.text_tokenizer import CNCMTextTokenizer
from cncm.tokenizer_model import CNCMConceptTokenizer


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _fallback_parse_plan_tokens(
    cfg: CNCMConfig,
    tokenizer: Any,
    plan_token_ids: Sequence[int],
) -> Dict[str, Any]:
    """Parse PG-FIM plan tokens `<mlen_k> (<kw> kw1 | kw2 </kw> | <kw/>)`."""
    seq = [int(t) for t in plan_token_ids]
    mlen_id: Optional[int] = None
    num_chunks = 1
    for tid in seq:
        if tid in cfg.mlen_token_ids:
            mlen_id = tid
            num_chunks = cfg.chunks_for_mlen_id(tid)
            break

    is_empty = cfg.kw_empty_id in seq
    kw_ids: List[int] = []
    if not is_empty and cfg.kw_start_id in seq:
        start_idx = seq.index(cfg.kw_start_id) + 1
        end_idx = seq.index(cfg.kw_end_id) if cfg.kw_end_id in seq[start_idx:] else len(seq)
        for tid in seq[start_idx:end_idx]:
            if tid not in (
                cfg.pad_token_id,
                cfg.eos_token_id,
                cfg.kw_start_id,
                cfg.kw_end_id,
                cfg.kw_empty_id,
                *cfg.mlen_token_ids,
            ):
                kw_ids.append(tid)

    raw_kw_text = tokenizer.decode(kw_ids, skip_special_tokens=True) if kw_ids else ""
    keywords = [
        part.strip()
        for part in raw_kw_text.split("|")
        if part.strip()
    ][: cfg.kw_max]
    if not keywords and not is_empty:
        is_empty = True

    raw_plan_text = tokenizer.decode(seq, skip_special_tokens=False) if seq else ""
    return {
        "mlen_id": mlen_id,
        "num_chunks": int(num_chunks),
        "keywords": keywords,
        "is_empty_plan": bool(is_empty),
        "raw_plan_text": raw_plan_text,
    }


def _fallback_check_syntax_and_keywords(
    prefix: str,
    middle: str,
    suffix: str,
    keywords: Sequence[str],
    language: str = "python",
) -> Dict[str, Any]:
    """Fallback syntax + keyword verifier when `cncm.code.symbols` is unavailable."""
    full_code = prefix + middle + suffix
    missing = [kw for kw in keywords if kw and kw not in middle]
    keywords_present = len(missing) == 0

    stripped_mid = middle.strip()
    non_trivial = bool(stripped_mid) and stripped_mid != "pass" and len(stripped_mid) >= 2

    parses = True
    syntax_error: Optional[str] = None
    if language.lower() == "python":
        try:
            ast.parse(full_code)
        except SyntaxError as exc:
            parses = False
            syntax_error = f"{exc.msg} (line {exc.lineno})"
    else:
        # Balanced bracket fallback for non-Python languages
        pairs = {"(": ")", "[": "]", "{": "}"}
        closers = set(pairs.values())
        stack: List[str] = []
        for ch in full_code:
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in closers:
                if not stack or stack[-1] != ch:
                    parses = False
                    syntax_error = f"Unbalanced bracket '{ch}'"
                    break
                stack.pop()
        if stack:
            parses = False
            syntax_error = "Unclosed brackets"

    return {
        "parses": parses,
        "keywords_present": keywords_present,
        "missing_keywords": missing,
        "syntax_error": syntax_error,
        "undefined_identifiers": [],
        "non_trivial": non_trivial,
    }


class _FallbackKeywordPlanMasker:
    """Minimal fallback PG-FIM plan grammar masker used if `KeywordPlanMasker` is absent."""

    def __init__(
        self,
        cfg: CNCMConfig,
        tokenizer: Any,
        candidate_keywords: Sequence[str],
        prefix_len: int,
        max_plan_tokens: int = 16,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prefix_len = int(prefix_len)
        self.max_plan_tokens = int(max_plan_tokens)
        self.kw_token_ids: List[int] = []
        for kw in candidate_keywords[: cfg.kw_max]:
            for tid in tokenizer.encode(kw, add_special_tokens=False):
                if tid not in self.kw_token_ids:
                    self.kw_token_ids.append(int(tid))
        pipe_ids = tokenizer.encode(" | ", add_special_tokens=False) or tokenizer.encode(
            "|", add_special_tokens=False
        )
        self.pipe_id = int(pipe_ids[0]) if pipe_ids else (16 + ord("|"))

    def __call__(self, generated_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        seq = (
            generated_ids[0].detach().cpu().tolist()
            if generated_ids.ndim == 2
            else list(generated_ids)
        )
        plan_seq = seq[self.prefix_len :] if len(seq) >= self.prefix_len else seq
        step = len(plan_seq)
        out = torch.full_like(logits, float("-inf"))
        if step == 0:
            for tid in self.cfg.mlen_token_ids:
                out[..., tid] = logits[..., tid]
            return out
        if step == 1:
            if self.kw_token_ids:
                out[..., self.cfg.kw_start_id] = logits[..., self.cfg.kw_start_id]
                out[..., self.cfg.kw_empty_id] = logits[..., self.cfg.kw_empty_id] - 1.0
            else:
                out[..., self.cfg.kw_empty_id] = logits[..., self.cfg.kw_empty_id]
                out[..., self.cfg.kw_start_id] = logits[..., self.cfg.kw_start_id]
            return out
        if self.cfg.kw_empty_id in plan_seq or self.cfg.kw_end_id in plan_seq:
            out[..., self.cfg.kw_end_id] = 0.0
            return out
        if step >= self.max_plan_tokens - 1 or not self.kw_token_ids:
            out[..., self.cfg.kw_end_id] = 0.0
            return out
        for tid in self.kw_token_ids:
            out[..., tid] = logits[..., tid] + 2.5
        out[..., self.pipe_id] = logits[..., self.pipe_id]
        out[..., self.cfg.kw_end_id] = logits[..., self.cfg.kw_end_id] + 1.0
        return out


class CNCMModel(nn.Module):
    """Unified Dual-Mode Concept + Exact-AR Model with Relaxed Recursive Looping."""

    def __init__(
        self,
        cfg: CNCMConfig,
        concept_tokenizer: Optional[CNCMConceptTokenizer] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Hold Stage 1 Concept Tokenizer as a submodule so .to(device) stays in sync,
        # while keeping it strictly frozen in Stage 2 (freeze_concept_tokenizer()).
        self.concept_tokenizer = (
            concept_tokenizer
            if concept_tokenizer is not None
            else CNCMConceptTokenizer(cfg)
        )
        self._concept_tokenizer_frozen: bool = False

        # Shared token embedding for BOTH causal prefixes (Concept Mode) & Exact-AR Mode
        self.exact_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embed_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()

        # Concept Mode input projections & learned level-0 queries
        self.concept_in_proj = make_linear(
            cfg.d_vq,
            cfg.hidden_size,
            bias=True,
            use_qat=cfg.use_qat_linear,
            weight_bits=cfg.qat_weight_bits,
            act_bits=cfg.qat_act_bits,
        )
        self.level_embed = nn.Embedding(cfg.num_levels, cfg.hidden_size)
        self.l0_queries = nn.Parameter(
            torch.randn(1, cfg.levels[0], cfg.hidden_size) * 0.02
        )

        # 2D-safe Rotary Embeddings shared across all effective layers
        self.rope = RotaryEmbedding(
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )

        # Unique Transformer blocks executed `num_loops` times (Relaxed Recursive Depth)
        self.blocks = nn.ModuleList(
            [CNCMBlock(cfg) for _ in range(cfg.num_unique_layers)]
        )
        eff_layers = cfg.effective_layers
        self.step_attn_norms = nn.ModuleList(
            [RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps) for _ in range(eff_layers)]
        )
        self.step_ffn_norms = nn.ModuleList(
            [RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps) for _ in range(eff_layers)]
        )
        if cfg.num_loops > 1 and cfg.loop_adapter_rank > 0:
            self.loop_adapters: Optional[nn.ModuleList] = nn.ModuleList(
                [
                    LoopAdapter(
                        cfg.hidden_size,
                        cfg.loop_adapter_rank,
                        eps=cfg.rms_norm_eps,
                        use_qat=cfg.use_qat_linear,
                        weight_bits=cfg.qat_weight_bits,
                        act_bits=cfg.qat_act_bits,
                    )
                    for _ in range(eff_layers)
                ]
            )
        else:
            self.loop_adapters = None

        # Per-loop MoE routers so shallow loop 0 and deep loop 1+ specialize independently
        if cfg.num_experts > 0:
            self.step_moe_routers: Optional[nn.ModuleList] = nn.ModuleList(
                [nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False) for _ in range(eff_layers)]
            )
        else:
            self.step_moe_routers = None

        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # Dual Heads: Factorized Joint Concept Head + Tied/Compact Exact-AR Head
        self.concept_head = FactorizedConceptHead(
            hidden_size=cfg.hidden_size,
            fsq_levels=cfg.fsq_levels,
            num_levels=cfg.num_levels,
            head_hidden=cfg.concept_head_hidden,
            rms_norm_eps=cfg.rms_norm_eps,
            use_qat=cfg.use_qat_linear,
            weight_bits=cfg.qat_weight_bits,
            act_bits=cfg.qat_act_bits,
        )
        self.exact_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self._init_generator_weights()
        if cfg.tie_word_embeddings:
            self.exact_head.weight = self.exact_embed.weight

        # Freeze Stage 1 by default so Stage 2 training never accidentally mutates it
        self.freeze_concept_tokenizer()
        self.grad_checkpointing: bool = False

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        """Toggle activation gradient checkpointing across recursive backbone blocks."""
        self.grad_checkpointing = bool(enabled)

    def _init_generator_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.exact_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.level_embed.weight, mean=0.0, std=std)
        nn.init.normal_(self.exact_head.weight, mean=0.0, std=std)
        nn.init.normal_(self.concept_in_proj.weight, mean=0.0, std=std)
        if self.concept_in_proj.bias is not None:
            nn.init.zeros_(self.concept_in_proj.bias)

        for block in self.blocks:
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, mean=0.0, std=std)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

        for mod in self.concept_head.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if isinstance(mod, nn.Linear) and mod.bias is not None:
                    nn.init.zeros_(mod.bias)

        # Keep LoopAdapter.up zero-initialized so recursive depth starts stably
        if self.loop_adapters is not None:
            for adapter in self.loop_adapters:
                nn.init.normal_(adapter.down.weight, mean=0.0, std=std)
                nn.init.zeros_(adapter.up.weight)

        if self.step_moe_routers is not None:
            for router in self.step_moe_routers:
                nn.init.normal_(router.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Freezing & parameter accounting helpers
    # ------------------------------------------------------------------
    def freeze_concept_tokenizer(self) -> None:
        """Freeze all Stage 1 concept tokenizer parameters and keep it in eval mode."""
        self._concept_tokenizer_frozen = True
        for p in self.concept_tokenizer.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.concept_tokenizer.eval()

    def unfreeze_concept_tokenizer(self) -> None:
        """Explicitly unfreeze Stage 1 concept tokenizer (rare; disabled by default)."""
        self._concept_tokenizer_frozen = False
        for p in self.concept_tokenizer.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True) -> "CNCMModel":
        super().train(mode)
        if self._concept_tokenizer_frozen:
            self.concept_tokenizer.eval()
        return self

    def generator_parameters(self) -> List[nn.Parameter]:
        """Return only Stage 2 generator parameters (excluding Stage 1 tokenizer)."""
        tok_ids = {id(p) for p in self.concept_tokenizer.parameters()}
        return [p for p in self.parameters() if id(p) not in tok_ids]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        tok = sum(p.numel() for p in self.concept_tokenizer.parameters())
        gen = total - tok
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        inactive_experts = 0
        if self.cfg.num_experts > 0:
            inactive_per_block = max(0, self.cfg.num_experts - self.cfg.num_experts_per_tok)
            for block in self.blocks:
                if getattr(block, "use_moe", False) and hasattr(block.mlp, "experts"):
                    per_exp = sum(p.numel() for p in block.mlp.experts[0].parameters())
                    inactive_experts += inactive_per_block * per_exp
        return {
            "total": total,
            "generator": gen,
            "concept_tokenizer": tok,
            "trainable": trainable,
            "active_total": total - inactive_experts,
            "active_generator": gen - inactive_experts,
        }

    # ------------------------------------------------------------------
    # Shared Recursive Backbone Execution
    # ------------------------------------------------------------------
    def _run_backbone(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        use_cache: bool = False,
        quantize_kv: bool = False,
        commit_to_cache: bool = True,
        return_aux_loss: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[List[LayerKV]]],
        Tuple[torch.Tensor, Optional[List[LayerKV]], torch.Tensor],
    ]:
        cos, sin = self.rope(position_ids, dtype=x.dtype)
        present_kvs: Optional[List[LayerKV]] = [] if use_cache else None
        use_ckpt = (
            self.grad_checkpointing
            and self.training
            and not use_cache
            and past_key_values is None
            and torch.is_grad_enabled()
        )
        aux_losses: List[torch.Tensor] = []

        h = x
        eff_idx = 0
        for _loop in range(self.cfg.num_loops):
            for layer_idx in range(self.cfg.num_unique_layers):
                block = self.blocks[layer_idx]
                past_kv = (
                    past_key_values[eff_idx]
                    if past_key_values is not None and eff_idx < len(past_key_values)
                    else None
                )
                adapter = (
                    self.loop_adapters[eff_idx]
                    if self.loop_adapters is not None
                    else None
                )
                moe_router = (
                    self.step_moe_routers[eff_idx]
                    if self.step_moe_routers is not None
                    else None
                )
                if use_ckpt:
                    attn_norm = self.step_attn_norms[eff_idx]
                    ffn_norm = self.step_ffn_norms[eff_idx]
                    if getattr(block, "use_moe", False):
                        def _ckpt_moe_block(
                            h_in: torch.Tensor,
                            cos_in: torch.Tensor,
                            sin_in: torch.Tensor,
                            mask_in: torch.Tensor,
                            _blk: CNCMBlock = block,
                            _anorm: RMSNorm = attn_norm,
                            _fnorm: RMSNorm = ffn_norm,
                            _adp: Optional[LoopAdapter] = adapter,
                            _rtr: Optional[nn.Linear] = moe_router,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
                            out_h, _, blk_aux = _blk(
                                h_in,
                                cos=cos_in,
                                sin=sin_in,
                                attn_mask=mask_in,
                                attn_norm=_anorm,
                                ffn_norm=_fnorm,
                                loop_adapter=_adp,
                                moe_router=_rtr,
                                past_kv=None,
                                use_cache=False,
                                quantize_kv=False,
                                commit_to_cache=False,
                                return_aux_loss=True,
                            )
                            return out_h, blk_aux if blk_aux is not None else out_h.new_zeros((), dtype=torch.float32)

                        h, step_aux = torch.utils.checkpoint.checkpoint(
                            _ckpt_moe_block,
                            h,
                            cos,
                            sin,
                            attn_mask,
                            use_reentrant=False,
                        )
                        aux_losses.append(step_aux)
                    else:
                        def _ckpt_block(
                            h_in: torch.Tensor,
                            cos_in: torch.Tensor,
                            sin_in: torch.Tensor,
                            mask_in: torch.Tensor,
                            _blk: CNCMBlock = block,
                            _anorm: RMSNorm = attn_norm,
                            _fnorm: RMSNorm = ffn_norm,
                            _adp: Optional[LoopAdapter] = adapter,
                        ) -> torch.Tensor:
                            out_h, _ = _blk(
                                h_in,
                                cos=cos_in,
                                sin=sin_in,
                                attn_mask=mask_in,
                                attn_norm=_anorm,
                                ffn_norm=_fnorm,
                                loop_adapter=_adp,
                                past_kv=None,
                                use_cache=False,
                                quantize_kv=False,
                                commit_to_cache=False,
                            )
                            return out_h

                        h = torch.utils.checkpoint.checkpoint(
                            _ckpt_block,
                            h,
                            cos,
                            sin,
                            attn_mask,
                            use_reentrant=False,
                        )
                    present = None
                else:
                    if return_aux_loss and getattr(block, "use_moe", False):
                        h, present, step_aux = block(
                            h,
                            cos=cos,
                            sin=sin,
                            attn_mask=attn_mask,
                            attn_norm=self.step_attn_norms[eff_idx],
                            ffn_norm=self.step_ffn_norms[eff_idx],
                            loop_adapter=adapter,
                            moe_router=moe_router,
                            past_kv=past_kv,
                            use_cache=use_cache,
                            quantize_kv=quantize_kv,
                            commit_to_cache=commit_to_cache,
                            return_aux_loss=True,
                        )
                        if step_aux is not None:
                            aux_losses.append(step_aux)
                    else:
                        h, present = block(
                            h,
                            cos=cos,
                            sin=sin,
                            attn_mask=attn_mask,
                            attn_norm=self.step_attn_norms[eff_idx],
                            ffn_norm=self.step_ffn_norms[eff_idx],
                            loop_adapter=adapter,
                            moe_router=moe_router,
                            past_kv=past_kv,
                            use_cache=use_cache,
                            quantize_kv=quantize_kv,
                            commit_to_cache=commit_to_cache,
                        )
                if present_kvs is not None and present is not None:
                    present_kvs.append(present)
                eff_idx += 1

        h = self.final_norm(h)
        if return_aux_loss:
            moe_aux = (
                torch.stack(aux_losses).mean()
                if aux_losses
                else torch.zeros((), device=h.device, dtype=torch.float32)
            )
            return h, present_kvs, moe_aux
        return h, present_kvs

    # ------------------------------------------------------------------
    # Stage 2 Training Forwards: Concept Mode, Exact-AR Mode, Joint Mode
    # ------------------------------------------------------------------
    def _level_weights(
        self, level_lengths: Sequence[int], alpha: float, device: torch.device
    ) -> torch.Tensor:
        """Compute normalized NCM level weights summing to 1 over mean per-level NLL."""
        lens = torch.tensor(level_lengths, device=device, dtype=torch.float32)
        raw = lens.pow(1.0 - float(alpha))
        return raw / raw.sum().clamp(min=1e-8)

    def _concept_level_positions(
        self, prefix_len: int, level_len: int, device: torch.device
    ) -> torch.Tensor:
        """Continuous Scaled RoPE positions placed strictly after `prefix_len`:

            pos_k(i) = prefix_len + floor((i + 0.5) * L / l_k),  i = 0..l_k-1

        Eliminates RoPE position collision with prefix tokens (0..P-1) and aligns
        target-to-prefix relative distances with Exact-AR mode.
        """
        idx = torch.arange(level_len, device=device, dtype=torch.float32)
        span_centers = torch.floor((idx + 0.5) * (float(self.cfg.chunk_size) / float(max(1, level_len)))).to(
            torch.long
        )
        max_pos = max(0, int(self.cfg.max_position_embeddings) - 1)
        return (int(prefix_len) + span_centers).clamp(max=max_pos)

    def forward_concept(
        self,
        prefix_ids: torch.Tensor,
        target_codes: Sequence[torch.Tensor],
        prefix_mask: Optional[torch.Tensor] = None,
        corrupt_rate: float = 0.5,
        level_weights_alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Single-pass teacher-forced Concept Mode forward across all target levels."""
        bsz, prefix_len = prefix_ids.shape
        device = prefix_ids.device
        num_active_levels = len(target_codes)
        if num_active_levels < 1 or num_active_levels > self.cfg.num_levels:
            raise ValueError(
                f"Expected 1..{self.cfg.num_levels} target_codes levels, got {num_active_levels}"
            )

        level_lengths = tuple(int(c.shape[1]) for c in target_codes)
        emb_dtype = self.exact_embed.weight.dtype

        # Build conditioning embeddings for levels 0..num_active_levels-1
        eff_corrupt = float(corrupt_rate) if (self.training and corrupt_rate > 0.0) else 0.0
        with torch.no_grad():
            cond_list = self.concept_tokenizer.quantizer.build_stage2_conditioning(
                target_codes, corrupt_rate=eff_corrupt, dtype=emb_dtype
            )

        parts: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        level_id_parts: List[torch.Tensor] = []

        if prefix_len > 0:
            prefix_emb = self.embed_dropout(self.exact_embed(prefix_ids))
            parts.append(prefix_emb)
            pos_parts.append(torch.arange(prefix_len, device=device, dtype=torch.long))

        for k in range(num_active_levels):
            l_k = level_lengths[k]
            lid = torch.tensor(k, device=device, dtype=torch.long)
            l_emb = self.level_embed(lid).view(1, 1, self.cfg.hidden_size)
            if k == 0:
                if l_k != self.l0_queries.shape[1]:
                    raise ValueError(
                        f"Level 0 length ({l_k}) does not match config levels[0] ({self.l0_queries.shape[1]})"
                    )
                x_k = self.l0_queries.to(dtype=emb_dtype).expand(bsz, -1, -1) + l_emb
            else:
                x_k = self.concept_in_proj(cond_list[k - 1].to(dtype=emb_dtype)) + l_emb
            parts.append(self.embed_dropout(x_k))
            # Continuous Scaled RoPE strictly after prefix_len (fixes RoPE collision!)
            pos_parts.append(self._concept_level_positions(prefix_len, l_k, device=device))
            level_id_parts.append(torch.full((l_k,), k, device=device, dtype=torch.long))

        x = torch.cat(parts, dim=1)
        position_ids = torch.cat(pos_parts, dim=0)
        level_ids = torch.cat(level_id_parts, dim=0)

        attn_mask = build_concept_mask(
            prefix_len=prefix_len,
            level_lengths=level_lengths,
            batch_size=bsz,
            prefix_mask=prefix_mask,
            device=device,
            dtype=x.dtype,
        )

        h, _, moe_aux_loss = self._run_backbone(
            x,
            attn_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            return_aux_loss=True,
        )

        h_target = h[:, prefix_len:, :]
        all_targets = torch.cat(
            [c.to(device=device, dtype=torch.long) for c in target_codes], dim=1
        )
        digit_logits, token_log_probs = self.concept_head.log_probs(
            h_target, level_ids=level_ids, target_digits=all_targets
        )

        # Compute per-level mean NLL and normalized weighted concept loss
        level_losses: List[torch.Tensor] = []
        offset = 0
        for l_k in level_lengths:
            lvl_nll = (-token_log_probs[:, offset : offset + l_k]).mean()
            level_losses.append(lvl_nll)
            offset += l_k

        weights = self._level_weights(level_lengths, level_weights_alpha, device=device)
        loss = torch.stack(level_losses).mul(weights).sum()
        if self.cfg.num_experts > 0 and self.cfg.moe_aux_loss_weight > 0.0:
            loss = loss + float(self.cfg.moe_aux_loss_weight) * moe_aux_loss.to(dtype=loss.dtype)

        return {
            "loss": loss,
            "moe_aux_loss": moe_aux_loss,
            "level_losses": level_losses,
            "digit_logits": digit_logits,
            "log_probs": token_log_probs,
            "h_target": h_target,
        }

    def forward_exact(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        quantize_kv: bool = False,
    ) -> Dict[str, Any]:
        """Exact causal autoregressive forward pass (shared backbone + exact_head)."""
        bsz, q_len = input_ids.shape
        device = input_ids.device
        past_len = get_kv_seq_len(past_key_values)
        kv_len = past_len + q_len

        x = self.embed_dropout(self.exact_embed(input_ids))

        # Explicit position_ids ALWAYS overrides start_pos (past_len)
        if position_ids is None:
            pos_ids = torch.arange(
                past_len, past_len + q_len, device=device, dtype=torch.long
            )
        else:
            pos_ids = position_ids.to(device=device, dtype=torch.long)

        attn_mask = build_causal_mask(
            q_len=q_len,
            kv_len=kv_len,
            past_len=past_len,
            batch_size=bsz,
            key_padding_mask=attention_mask,
            device=device,
            dtype=x.dtype,
        )

        h, present_kvs, moe_aux_loss = self._run_backbone(
            x,
            attn_mask=attn_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            quantize_kv=quantize_kv,
            commit_to_cache=True,
            return_aux_loss=True,
        )

        logits = self.exact_head(h)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if q_len > 1:
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous().to(device=device, dtype=torch.long)
                valid_count = (shift_labels != -100).sum()
                if valid_count.item() > 0:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.cfg.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                else:
                    loss = logits.sum() * 0.0
            else:
                loss = logits.sum() * 0.0
            if self.cfg.num_experts > 0 and self.cfg.moe_aux_loss_weight > 0.0:
                loss = loss + float(self.cfg.moe_aux_loss_weight) * moe_aux_loss.to(dtype=loss.dtype)

        return {
            "loss": loss,
            "moe_aux_loss": moe_aux_loss,
            "logits": logits,
            "past_key_values": present_kvs,
        }

    def forward_joint(
        self,
        prefix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        concept_weight: float = 1.0,
        exact_weight: float = 0.3,
        align_weight: float = 0.3,
        corrupt_rate: float = 0.5,
        exact_labels: Optional[torch.Tensor] = None,
        level_weights_alpha: float = 0.5,
        supervise_prefix_exact: bool = False,
    ) -> Dict[str, Any]:
        """Joint Stage 2 objective: Concept Mode + Exact-AR + Text-Level Alignment (`L_align`)."""
        bsz, prefix_len = prefix_ids.shape
        target_len = target_ids.shape[1]
        device = prefix_ids.device

        # 1. Concept Mode supervision using frozen Stage 1 concept codes
        with torch.no_grad():
            target_codes = self.concept_tokenizer.encode_codes(target_ids)

        concept_out = self.forward_concept(
            prefix_ids=prefix_ids,
            target_codes=target_codes,
            prefix_mask=prefix_mask,
            corrupt_rate=corrupt_rate,
            level_weights_alpha=level_weights_alpha,
        )
        concept_loss: torch.Tensor = concept_out["loss"]

        # 2. Text-Level Concept Alignment Loss (L_align): project finest-level Concept hidden
        # states through shared `exact_head` against target token IDs.
        if align_weight > 0.0:
            h_fine = concept_out["h_target"][:, -target_len:, :]
            align_logits = self.exact_head(h_fine).float()
            align_targets = target_ids.to(device=device, dtype=torch.long).clone()
            non_pad_align = align_targets != self.cfg.pad_token_id
            if non_pad_align.any():
                align_targets = align_targets.masked_fill(~non_pad_align, -100)
            valid_align = (align_targets != -100).sum()
            if valid_align.item() > 0:
                align_loss: torch.Tensor = F.cross_entropy(
                    align_logits.reshape(-1, self.cfg.vocab_size),
                    align_targets.reshape(-1),
                    ignore_index=-100,
                )
            else:
                align_loss = align_logits.sum() * 0.0
        else:
            align_loss = concept_loss.new_zeros(())

        # 3. Exact-AR auxiliary supervision over [prefix_ids, target_ids]
        if exact_weight > 0.0:
            full_ids = torch.cat([prefix_ids, target_ids], dim=1)
            if prefix_mask is not None:
                target_valid = torch.ones(
                    (bsz, target_len), device=device, dtype=torch.bool
                )
                full_mask: Optional[torch.Tensor] = torch.cat(
                    [prefix_mask.to(device=device, dtype=torch.bool), target_valid],
                    dim=1,
                )
            else:
                full_mask = None

            if supervise_prefix_exact:
                labels = full_ids.to(device=device, dtype=torch.long).clone()
                if full_mask is not None:
                    labels = labels.masked_fill(~full_mask, -100)
                non_pad_all = labels != self.cfg.pad_token_id
                if non_pad_all.any():
                    labels = labels.masked_fill(~non_pad_all, -100)
            elif exact_labels is not None:
                el = exact_labels.to(device=device, dtype=torch.long)
                if el.shape[1] == target_len:
                    prefix_ignore = torch.full(
                        (bsz, prefix_len), -100, device=device, dtype=torch.long
                    )
                    labels = torch.cat([prefix_ignore, el], dim=1)
                else:
                    labels = el
            else:
                prefix_ignore = torch.full(
                    (bsz, prefix_len), -100, device=device, dtype=torch.long
                )
                tgt_labels = target_ids.to(device=device, dtype=torch.long).clone()
                non_pad = tgt_labels != self.cfg.pad_token_id
                if non_pad.any():
                    tgt_labels = tgt_labels.masked_fill(~non_pad, -100)
                labels = torch.cat([prefix_ignore, tgt_labels], dim=1)

            exact_out = self.forward_exact(
                full_ids,
                labels=labels,
                attention_mask=full_mask,
                use_cache=False,
            )
            exact_loss: torch.Tensor = exact_out["loss"]
        else:
            exact_loss = concept_loss.new_zeros(())

        loss = (
            float(concept_weight) * concept_loss
            + float(exact_weight) * exact_loss
            + float(align_weight) * align_loss
        )
        return {
            "loss": loss,
            "concept_loss": concept_loss,
            "exact_loss": exact_loss,
            "align_loss": align_loss,
        }

    # ------------------------------------------------------------------
    # Online RL / GRPO Exact 1-Pass Code-Space & PG-FIM Log-Probabilities
    # ------------------------------------------------------------------
    def compute_code_logprobs(
        self,
        prefix_ids: torch.Tensor,
        sampled_codes: Sequence[torch.Tensor],
        prefix_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Exact 1-pass code-space log-probability `[B]` of a sampled rollout for GRPO."""
        out = self.forward_concept(
            prefix_ids=prefix_ids,
            target_codes=sampled_codes,
            prefix_mask=prefix_mask,
            corrupt_rate=0.0,
        )
        token_lp: torch.Tensor = out["log_probs"]  # [B, T]
        if reduction == "sum":
            return token_lp.sum(dim=-1)
        if reduction == "mean":
            return token_lp.mean(dim=-1)
        raise ValueError(f"Unsupported reduction '{reduction}'; expected 'mean' or 'sum'")

    def compute_pg_fim_logprobs(
        self,
        prefix_ids: torch.Tensor,
        plan_ids: Optional[torch.Tensor],
        sampled_codes: Sequence[Any],
        prefix_mask: Optional[torch.Tensor] = None,
        plan_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Compute combined Exact-AR plan log-probs + 1-pass Concept Mode code log-probs.

        Used by Stage 3 Compiler-in-the-Loop GRPO on PG-FIM tasks.
        Returns dict(total_logprobs, plan_logprobs, code_logprobs), each shape `[B]`.
        """
        bsz, p_len = prefix_ids.shape
        device = prefix_ids.device

        if prefix_mask is not None:
            p_mask = prefix_mask.to(device=device, dtype=torch.bool)
        else:
            p_mask = torch.ones((bsz, p_len), device=device, dtype=torch.bool)

        if plan_ids is not None and plan_ids.numel() > 0 and plan_ids.shape[1] > 0:
            pl_ids = plan_ids.to(device=device, dtype=torch.long)
            if pl_ids.ndim == 1:
                pl_ids = pl_ids.unsqueeze(0).expand(bsz, -1)
            l_plan = int(pl_ids.shape[1])

            if plan_mask is not None:
                pl_mask = plan_mask.to(device=device, dtype=torch.bool)
                if pl_mask.ndim == 1:
                    pl_mask = pl_mask.unsqueeze(0).expand(bsz, -1)
            else:
                pl_mask = pl_ids != self.cfg.pad_token_id
                empty_rows = ~pl_mask.any(dim=-1, keepdim=True)
                if bool(empty_rows.any().item()):
                    pl_mask = pl_mask.clone()
                    pl_mask[empty_rows.squeeze(-1), 0] = True

            full_prefix_ids = torch.cat([prefix_ids, pl_ids], dim=1)
            full_prefix_mask = torch.cat([p_mask, pl_mask], dim=1)

            exact_out = self.forward_exact(
                full_prefix_ids,
                attention_mask=full_prefix_mask,
                use_cache=False,
            )
            full_logits = exact_out["logits"].float()

            if p_len > 0:
                pred_logits = full_logits[:, p_len - 1 : p_len - 1 + l_plan, :]
                target_pl = pl_ids.clamp(0, self.cfg.vocab_size - 1)
                valid_pl = pl_mask.float()
            elif l_plan > 1:
                pred_logits = full_logits[:, :-1, :]
                target_pl = pl_ids[:, 1:].clamp(0, self.cfg.vocab_size - 1)
                valid_pl = pl_mask[:, 1:].float()
            else:
                pred_logits = full_logits
                target_pl = pl_ids.clamp(0, self.cfg.vocab_size - 1)
                valid_pl = pl_mask.float()

            log_p = F.log_softmax(pred_logits, dim=-1).gather(
                -1, target_pl.unsqueeze(-1)
            ).squeeze(-1)
            if reduction == "sum":
                plan_logprobs = (log_p * valid_pl).sum(dim=-1)
            elif reduction == "mean":
                plan_logprobs = (log_p * valid_pl).sum(dim=-1) / valid_pl.sum(
                    dim=-1
                ).clamp(min=1.0)
            else:
                raise ValueError(
                    f"Unsupported reduction '{reduction}'; expected 'mean' or 'sum'"
                )
        else:
            full_prefix_ids = prefix_ids
            full_prefix_mask = p_mask
            plan_logprobs = torch.zeros(bsz, device=device, dtype=torch.float32)

        # Support single-chunk `list[Tensor]` or multi-chunk `list[list[Tensor]]`
        if (
            len(sampled_codes) > 0
            and isinstance(sampled_codes[0], (list, tuple))
        ):
            chunk_lps: List[torch.Tensor] = []
            for chunk_codes in sampled_codes:
                chunk_lps.append(
                    self.compute_code_logprobs(
                        prefix_ids=full_prefix_ids,
                        sampled_codes=chunk_codes,
                        prefix_mask=full_prefix_mask,
                        reduction=reduction,
                    )
                )
            code_logprobs = (
                torch.stack(chunk_lps, dim=0).mean(dim=0)
                if reduction == "mean"
                else torch.stack(chunk_lps, dim=0).sum(dim=0)
            )
        else:
            code_logprobs = self.compute_code_logprobs(
                prefix_ids=full_prefix_ids,
                sampled_codes=sampled_codes,  # type: ignore[arg-type]
                prefix_mask=full_prefix_mask,
                reduction=reduction,
            )

        total_logprobs = plan_logprobs + code_logprobs
        return {
            "total_logprobs": total_logprobs,
            "plan_logprobs": plan_logprobs,
            "code_logprobs": code_logprobs,
        }

    # ------------------------------------------------------------------
    # Generation: Concept Chunk, Multi-Chunk Block-AR, Exact-AR, Self-Speculative
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prefill_or_extend_prefix(
        self,
        prefix_ids: Optional[torch.Tensor],
        prefix_kv: Optional[Sequence[LayerKV]],
        prefix_len: int,
        prefix_mask: Optional[torch.Tensor],
        quantize_kv: bool,
    ) -> Tuple[Optional[List[LayerKV]], int, Optional[torch.Tensor]]:
        """Prefill a new prefix or commit new delta tokens onto an existing causal prefix KV."""
        cached_len = get_kv_seq_len(prefix_kv)
        cur_kv: Optional[List[LayerKV]] = list(prefix_kv) if prefix_kv is not None else None
        last_logits: Optional[torch.Tensor] = None

        if prefix_ids is not None and prefix_ids.shape[1] > 0:
            if cur_kv is None:
                out = self.forward_exact(
                    prefix_ids,
                    attention_mask=prefix_mask,
                    past_key_values=None,
                    use_cache=True,
                    quantize_kv=quantize_kv,
                )
                cur_kv = out["past_key_values"]
                last_logits = out["logits"][:, -1, :]
                cached_len = int(prefix_ids.shape[1])
            else:
                # Commit newly supplied prefix delta tokens onto cached KV via bottom-right mask
                out = self.forward_exact(
                    prefix_ids,
                    attention_mask=prefix_mask,
                    past_key_values=cur_kv,
                    use_cache=True,
                    quantize_kv=quantize_kv,
                )
                cur_kv = out["past_key_values"]
                last_logits = out["logits"][:, -1, :]
                cached_len = get_kv_seq_len(cur_kv)
        else:
            cached_len = cached_len if cached_len > 0 else int(prefix_len)

        return cur_kv, cached_len, last_logits

    @staticmethod
    def _normalize_prefix_mask(
        prefix_mask: Optional[torch.Tensor],
        target_len: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Normalize a delta [B, Q] or full [B, P+Q] prefix_mask to exact length `target_len`."""
        if prefix_mask is None or target_len <= 0:
            return None
        pm = prefix_mask.to(device=device, dtype=torch.bool)
        if pm.shape[1] < target_len:
            past_valid = torch.ones(
                (batch_size, target_len - pm.shape[1]),
                device=device,
                dtype=torch.bool,
            )
            pm = torch.cat([past_valid, pm], dim=1)
        elif pm.shape[1] > target_len:
            pm = pm[:, -target_len:]
        return pm

    @torch.no_grad()
    def _draft_concept_from_kv(
        self,
        batch_size: int,
        device: torch.device,
        prefix_kv: Optional[Sequence[LayerKV]],
        cur_prefix_len: int,
        prefix_mask: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        early_exit_conf: Optional[float] = 0.92,
        quantize_kv: bool = False,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
        level_temperatures: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], int]:
        """Run coarse-to-fine levels conditioned on cached `prefix_kv` without mutating it.

        Uses Continuous Scaled RoPE (`pos_k = cur_prefix_len + floor((i+0.5)*L/l_k)`)
        and per-sample early-exit freezing so easy rows exit without waiting for hard rows.
        """
        emb_dtype = self.exact_embed.weight.dtype
        z_hat = torch.zeros(
            (batch_size, self.cfg.chunk_size, self.cfg.d_vq),
            device=device,
            dtype=emb_dtype,
        )
        codes_list: List[torch.Tensor] = []
        row_done = torch.zeros(batch_size, device=device, dtype=torch.bool)
        row_exit_levels = torch.full((batch_size,), self.cfg.num_levels, device=device, dtype=torch.long)
        exit_level = self.cfg.num_levels

        for k in range(self.cfg.num_levels):
            l_k = self.cfg.levels[k]
            lid = torch.tensor(k, device=device, dtype=torch.long)
            l_emb = self.level_embed(lid).view(1, 1, self.cfg.hidden_size)
            if k == 0:
                x_k = self.l0_queries.to(dtype=emb_dtype).expand(batch_size, -1, -1) + l_emb
            else:
                cond_k = self.concept_tokenizer.quantizer.get_level_conditioning(z_hat, k)
                x_k = self.concept_in_proj(cond_k.to(dtype=emb_dtype)) + l_emb

            pos_k = self._concept_level_positions(cur_prefix_len, l_k, device=device)
            step_mask = build_cached_concept_step_mask(
                prefix_len=cur_prefix_len,
                level_len=l_k,
                batch_size=batch_size,
                prefix_mask=prefix_mask,
                device=device,
                dtype=x_k.dtype,
            )

            h_k, _ = self._run_backbone(
                x_k,
                attn_mask=step_mask,
                position_ids=pos_k,
                past_key_values=prefix_kv,
                use_cache=False,
                quantize_kv=quantize_kv,
                commit_to_cache=False,
            )

            step_temp = (
                float(level_temperatures[k])
                if (level_temperatures is not None and k < len(level_temperatures))
                else float(temperature)
            )
            digits_k, _, mean_conf = self.concept_head.sample(
                h_k, level_ids=k, temperature=step_temp
            )
            codes_list.append(digits_k)
            delta_k = self.concept_tokenizer.quantizer.decode_level(
                digits_k, k, dtype=emb_dtype
            )
            # Only accumulate residual delta for rows that have not yet early-exited
            active_f = (~row_done).to(dtype=emb_dtype).view(batch_size, 1, 1)
            z_hat = z_hat + delta_k * active_f

            # Per-sample Dynamic Early-Exit Confidence Gate (eligible after >= 2 levels)
            if (
                early_exit_conf is not None
                and 0.0 < early_exit_conf < 1.0
                and k >= 1
                and k < self.cfg.num_levels - 1
            ):
                newly_done = (~row_done) & (mean_conf >= float(early_exit_conf))
                row_exit_levels = torch.where(
                    newly_done,
                    torch.full_like(row_exit_levels, k + 1),
                    row_exit_levels,
                )
                row_done = row_done | newly_done
                if bool(row_done.all().item()):
                    exit_level = k + 1
                    break

        dec_logits = self.concept_tokenizer.decode_z_hat(
            z_hat.to(dtype=self.concept_tokenizer.wte.weight.dtype),
            kw_token_ids=kw_token_ids,
            kw_mask=kw_mask,
            cutoff_level=int(row_exit_levels.max().item()),
        )
        token_ids = dec_logits.argmax(dim=-1)
        return token_ids, dec_logits, codes_list, exit_level

    @torch.no_grad()
    def generate_concept_chunk(
        self,
        prefix_ids: Optional[torch.Tensor],
        prefix_kv: Optional[Sequence[LayerKV]] = None,
        prefix_len: int = 0,
        temperature: float = 0.8,
        early_exit_conf: Optional[float] = 0.92,
        quantize_kv: bool = False,
        return_codes: bool = False,
        prefix_mask: Optional[torch.Tensor] = None,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
        level_temperatures: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Generate one L-token chunk in <= K parallel coarse-to-fine steps.

        Returns dict(token_ids, codes, exit_level, prefix_kv, prefix_len).
        """
        if prefix_ids is not None:
            bsz = int(prefix_ids.shape[0])
            device = prefix_ids.device
        elif prefix_kv is not None:
            first = prefix_kv[0]
            ref_t = first.k_int8 if hasattr(first, "k_int8") else first[0]
            bsz = int(ref_t.shape[0])
            device = ref_t.device
        else:
            raise ValueError("Either prefix_ids or prefix_kv must be provided")

        cur_kv, cur_prefix_len, _ = self._prefill_or_extend_prefix(
            prefix_ids=prefix_ids,
            prefix_kv=prefix_kv,
            prefix_len=prefix_len,
            prefix_mask=prefix_mask,
            quantize_kv=quantize_kv,
        )
        full_prefix_mask = self._normalize_prefix_mask(
            prefix_mask, cur_prefix_len, bsz, device
        )

        token_ids, _, codes_list, exit_level = self._draft_concept_from_kv(
            batch_size=bsz,
            device=device,
            prefix_kv=cur_kv,
            cur_prefix_len=cur_prefix_len,
            prefix_mask=full_prefix_mask,
            temperature=temperature,
            early_exit_conf=early_exit_conf,
            quantize_kv=quantize_kv,
            kw_token_ids=kw_token_ids,
            kw_mask=kw_mask,
            level_temperatures=level_temperatures,
        )

        return {
            "token_ids": token_ids,
            "codes": codes_list if return_codes else None,
            "exit_level": int(exit_level),
            "prefix_kv": cur_kv,
            "prefix_len": int(cur_prefix_len),
        }

    @torch.no_grad()
    def generate_block_ar(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        early_exit_conf: Optional[float] = 0.92,
        quantize_kv: bool = False,
        stop_token_ids: Optional[Sequence[int]] = None,
        prefix_mask: Optional[torch.Tensor] = None,
        kw_token_ids: Optional[torch.Tensor] = None,
        kw_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Multi-chunk Causal-Prefix Block-AR generation for arbitrary output lengths."""
        if max_new_tokens <= 0:
            bsz = int(prefix_ids.shape[0])
            return {
                "token_ids": prefix_ids.new_empty((bsz, 0)),
                "chunks": 0,
                "avg_exit_level": 0.0,
            }

        bsz = int(prefix_ids.shape[0])
        stop_set = (
            set(int(s) for s in stop_token_ids)
            if stop_token_ids is not None
            else {self.cfg.eos_token_id}
        )

        kv: Optional[List[LayerKV]] = None
        plen = 0
        cur_mask = prefix_mask
        generated_chunks: List[torch.Tensor] = []
        exit_levels: List[int] = []
        total_gen = 0
        next_prefix: Optional[torch.Tensor] = prefix_ids

        while total_gen < max_new_tokens:
            out = self.generate_concept_chunk(
                prefix_ids=next_prefix,
                prefix_kv=kv,
                prefix_len=plen,
                temperature=temperature,
                early_exit_conf=early_exit_conf,
                quantize_kv=quantize_kv,
                return_codes=False,
                prefix_mask=cur_mask,
                kw_token_ids=kw_token_ids,
                kw_mask=kw_mask,
            )
            chunk_ids: torch.Tensor = out["token_ids"]
            kv = out["prefix_kv"]
            plen = int(out["prefix_len"])
            exit_levels.append(int(out["exit_level"]))
            generated_chunks.append(chunk_ids)
            total_gen += int(chunk_ids.shape[1])

            if cur_mask is not None:
                cur_mask = torch.cat(
                    [
                        cur_mask.to(dtype=torch.bool),
                        torch.ones(
                            (bsz, chunk_ids.shape[1]),
                            device=chunk_ids.device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )

            if stop_set:
                all_stopped = True
                for b in range(bsz):
                    seq_b = chunk_ids[b].tolist()
                    if not any(t in stop_set for t in seq_b):
                        all_stopped = False
                        break
                if all_stopped:
                    break

            next_prefix = chunk_ids

        full_new = torch.cat(generated_chunks, dim=1)[:, :max_new_tokens]
        avg_exit = sum(exit_levels) / float(max(len(exit_levels), 1))
        return {
            "token_ids": full_new,
            "chunks": len(generated_chunks),
            "avg_exit_level": float(avg_exit),
            "prefix_kv": kv,
            "prefix_len": plen,
        }

    @torch.no_grad()
    def generate_exact(
        self,
        prefix_ids: Optional[torch.Tensor],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        stop_token_ids: Optional[Sequence[int]] = None,
        logit_mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        past_key_values: Optional[Sequence[LayerKV]] = None,
        quantize_kv: bool = False,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Exact token-by-token autoregressive generation with optional grammar masking."""
        if prefix_ids is not None:
            bsz = int(prefix_ids.shape[0])
            device = prefix_ids.device
        elif past_key_values is not None:
            first = past_key_values[0]
            ref_t = first.k_int8 if hasattr(first, "k_int8") else first[0]
            bsz = int(ref_t.shape[0])
            device = ref_t.device
        else:
            raise ValueError("Either prefix_ids or past_key_values must be provided")

        if max_new_tokens <= 0:
            return {
                "token_ids": torch.empty((bsz, 0), device=device, dtype=torch.long),
                "past_key_values": past_key_values,
            }

        kv, cur_prefix_len, next_logits = self._prefill_or_extend_prefix(
            prefix_ids=prefix_ids,
            prefix_kv=past_key_values,
            prefix_len=get_kv_seq_len(past_key_values),
            prefix_mask=prefix_mask,
            quantize_kv=quantize_kv,
        )
        if next_logits is None:
            raise ValueError(
                "generate_exact requires non-empty prefix_ids to obtain initial next-token logits"
            )

        stop_set = (
            set(int(s) for s in stop_token_ids)
            if stop_token_ids is not None
            else set()
        )
        finished = torch.zeros(bsz, device=device, dtype=torch.bool)
        generated_steps: List[torch.Tensor] = []

        # Determine whether logit_mask_fn expects full [prefix + generated] or new-only IDs
        pass_full_to_mask = False
        if logit_mask_fn is not None and prefix_ids is not None and prefix_ids.shape[1] > 0:
            mask_prefix_len = getattr(logit_mask_fn, "prefix_len", None)
            mask_tool_start = getattr(logit_mask_fn, "tool_call_start_id", None)
            mask_fim_mid = getattr(logit_mask_fn, "fim_middle_id", None)
            if mask_prefix_len is not None:
                pass_full_to_mask = True
            elif mask_tool_start is not None and bool((prefix_ids == int(mask_tool_start)).any().item()):
                pass_full_to_mask = True
            elif mask_fim_mid is not None and bool((prefix_ids == int(mask_fim_mid)).any().item()):
                pass_full_to_mask = True

        cur_mask = self._normalize_prefix_mask(prefix_mask, cur_prefix_len, bsz, device)
        for step in range(max_new_tokens):
            step_logits = next_logits
            if logit_mask_fn is not None:
                if generated_steps:
                    gen_so_far = torch.stack(generated_steps, dim=1)
                else:
                    gen_so_far = torch.empty((bsz, 0), device=device, dtype=torch.long)
                mask_input = (
                    torch.cat([prefix_ids, gen_so_far], dim=1)
                    if (pass_full_to_mask and prefix_ids is not None)
                    else gen_so_far
                )
                step_logits = logit_mask_fn(mask_input, step_logits)

            if temperature <= 0.0:
                next_tok = step_logits.argmax(dim=-1)
            else:
                scaled = step_logits.float() / float(temperature)
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if stop_set:
                pad_fill = torch.full_like(next_tok, self.cfg.pad_token_id)
                next_tok = torch.where(finished, pad_fill, next_tok)
                for sid in stop_set:
                    finished = finished | (next_tok == sid)

            generated_steps.append(next_tok)

            if stop_set and bool(finished.all().item()):
                break

            if step + 1 < max_new_tokens:
                if cur_mask is not None:
                    cur_mask = torch.cat(
                        [
                            cur_mask.to(dtype=torch.bool),
                            torch.ones((bsz, 1), device=device, dtype=torch.bool),
                        ],
                        dim=1,
                    )
                step_out = self.forward_exact(
                    next_tok.unsqueeze(1),
                    attention_mask=cur_mask,
                    past_key_values=kv,
                    use_cache=True,
                    quantize_kv=quantize_kv,
                )
                kv = step_out["past_key_values"]
                next_logits = step_out["logits"][:, -1, :]

        token_ids = (
            torch.stack(generated_steps, dim=1)
            if generated_steps
            else torch.empty((bsz, 0), device=device, dtype=torch.long)
        )
        return {
            "token_ids": token_ids,
            "past_key_values": kv,
        }

    @torch.no_grad()
    def generate_self_speculative(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        quantize_kv: bool = False,
        early_exit_conf: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        prefix_mask: Optional[torch.Tensor] = None,
        draft_horizon: Optional[int] = None,
        min_acceptance_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Lossless Self-Speculative Decoding: Concept-Mode draft + 1-pass Exact-AR verify.

        Supports short-horizon drafting (`draft_horizon`, e.g., 16) and an optional EMA
        acceptance circuit breaker (`min_acceptance_rate`, e.g., 0.25) that falls back
        directly to `generate_exact` when draft acceptance is below break-even.
        """
        bsz = int(prefix_ids.shape[0])
        device = prefix_ids.device
        if max_new_tokens <= 0:
            return {
                "token_ids": prefix_ids.new_empty((bsz, 0)),
                "acceptance_rate": 0.0,
                "concept_steps": 0,
                "verify_passes": 0,
                "circuit_breaker_triggered": False,
            }

        stop_set = (
            set(int(s) for s in stop_token_ids)
            if stop_token_ids is not None
            else set()
        )

        pre_out = self.forward_exact(
            prefix_ids,
            attention_mask=prefix_mask,
            past_key_values=None,
            use_cache=True,
            quantize_kv=quantize_kv,
        )
        kv: Optional[List[LayerKV]] = pre_out["past_key_values"]
        last_logits: torch.Tensor = pre_out["logits"][:, -1, :]  # [B, V]
        cur_len = int(prefix_ids.shape[1])
        cur_mask = prefix_mask

        emitted_chunks: List[torch.Tensor] = []
        generated_count = 0
        total_drafted = 0
        total_accepted = 0
        concept_steps = 0
        verify_passes = 0
        circuit_breaker_triggered = False
        ema_accept = 1.0
        eff_horizon = (
            max(1, min(int(draft_horizon), self.cfg.chunk_size))
            if draft_horizon is not None
            else self.cfg.chunk_size
        )

        while generated_count < max_new_tokens:
            remaining = max_new_tokens - generated_count

            # EMA Acceptance Circuit Breaker: fall back directly to Exact-AR if below break-even
            if (
                min_acceptance_rate > 0.0
                and verify_passes >= 2
                and ema_accept < float(min_acceptance_rate)
            ):
                circuit_breaker_triggered = True
                # Emit the already-computed `last_logits` token, then finish via cached Exact-AR
                if temperature <= 0.0:
                    first_tok = last_logits.argmax(dim=-1, keepdim=True)
                else:
                    p0 = F.softmax(last_logits.float() / float(temperature), dim=-1)
                    first_tok = torch.multinomial(p0, num_samples=1)
                emitted_chunks.append(first_tok)
                generated_count += 1
                if generated_count < max_new_tokens:
                    rem_after = max_new_tokens - generated_count
                    if cur_mask is not None:
                        cur_mask = torch.cat(
                            [cur_mask.to(dtype=torch.bool), torch.ones((bsz, 1), device=device, dtype=torch.bool)],
                            dim=1,
                        )
                    fb_out = self.generate_exact(
                        first_tok,
                        max_new_tokens=rem_after,
                        temperature=temperature,
                        stop_token_ids=stop_token_ids,
                        past_key_values=kv,
                        quantize_kv=quantize_kv,
                        prefix_mask=cur_mask,
                    )
                    emitted_chunks.append(fb_out["token_ids"])
                    generated_count += int(fb_out["token_ids"].shape[1])
                break

            draft_full, draft_logits_full, _, exit_level = self._draft_concept_from_kv(
                batch_size=bsz,
                device=device,
                prefix_kv=kv,
                cur_prefix_len=cur_len,
                prefix_mask=cur_mask,
                temperature=temperature,
                early_exit_conf=early_exit_conf,
                quantize_kv=quantize_kv,
            )
            concept_steps += int(exit_level)

            draft_len = min(eff_horizon, remaining)
            draft = draft_full[:, :draft_len]
            total_drafted += draft_len

            ver_mask: Optional[torch.Tensor] = None
            if cur_mask is not None:
                ver_mask = torch.cat(
                    [
                        cur_mask.to(dtype=torch.bool),
                        torch.ones((bsz, draft_len), device=device, dtype=torch.bool),
                    ],
                    dim=1,
                )

            ver_out = self.forward_exact(
                draft,
                attention_mask=ver_mask,
                past_key_values=kv,
                use_cache=True,
                quantize_kv=quantize_kv,
            )
            verify_passes += 1
            ver_logits: torch.Tensor = ver_out["logits"]
            ver_kv: List[LayerKV] = ver_out["past_key_values"]

            target_logits = torch.cat([last_logits.unsqueeze(1), ver_logits], dim=1)

            if temperature <= 0.0:
                exact_preds = target_logits.argmax(dim=-1)
                m = draft_len
                for i in range(draft_len):
                    if not bool((draft[:, i] == exact_preds[:, i]).all().item()):
                        m = i
                        break
                total_accepted += m

                if m == draft_len and generated_count + draft_len >= max_new_tokens:
                    emitted = exact_preds[:, :draft_len]
                    emitted_chunks.append(emitted)
                    generated_count += draft_len
                    break

                emitted = exact_preds[:, : m + 1]
                bonus_tok = exact_preds[:, m : m + 1]
            else:
                # Sample draft tokens directly from q_probs (the Stage 1 decoder distribution)
                # so speculative rejection sampling is mathematically valid for T > 0.
                q_probs = F.softmax(
                    draft_logits_full[:, :draft_len, :].float() / float(temperature),
                    dim=-1,
                )
                draft = torch.multinomial(
                    q_probs.reshape(-1, q_probs.shape[-1]), num_samples=1
                ).view(bsz, draft_len)
                # Re-verify the actually sampled `draft` tokens in one parallel pass
                ver_out = self.forward_exact(
                    draft,
                    attention_mask=ver_mask,
                    past_key_values=kv,
                    use_cache=True,
                    quantize_kv=quantize_kv,
                )
                ver_logits = ver_out["logits"]
                ver_kv = ver_out["past_key_values"]
                target_logits = torch.cat([last_logits.unsqueeze(1), ver_logits], dim=1)
                p_probs = F.softmax(target_logits.float() / float(temperature), dim=-1)
                m = draft_len
                for i in range(draft_len):
                    tok_i = draft[:, i]
                    p_i = p_probs[torch.arange(bsz, device=device), i, tok_i]
                    q_i = q_probs[torch.arange(bsz, device=device), i, tok_i].clamp(min=1e-9)
                    accept_p = torch.clamp(p_i / q_i, max=1.0)
                    if not bool((torch.rand(bsz, device=device) <= accept_p).all().item()):
                        m = i
                        break
                total_accepted += m

                if m == draft_len and generated_count + draft_len >= max_new_tokens:
                    emitted_chunks.append(draft[:, :draft_len])
                    generated_count += draft_len
                    break

                if m < draft_len:
                    diff = torch.clamp(p_probs[:, m, :] - q_probs[:, m, :], min=0.0)
                    diff_sum = diff.sum(dim=-1, keepdim=True)
                    fallback = p_probs[:, m, :]
                    res_probs = torch.where(
                        diff_sum > 1e-8, diff / diff_sum.clamp(min=1e-8), fallback
                    )
                    bonus_tok = torch.multinomial(res_probs, num_samples=1)
                else:
                    bonus_tok = torch.multinomial(p_probs[:, draft_len, :], num_samples=1)

                emitted = (
                    torch.cat([draft[:, :m], bonus_tok], dim=1) if m > 0 else bonus_tok
                )

            emitted_chunks.append(emitted)
            generated_count += int(emitted.shape[1])

            if stop_set:
                stop_hit = False
                for b in range(bsz):
                    if any(t in stop_set for t in emitted[b].tolist()):
                        stop_hit = True
                        break
                if stop_hit and bsz == 1:
                    seq_list = emitted[0].tolist()
                    for idx, t in enumerate(seq_list):
                        if t in stop_set:
                            emitted_chunks[-1] = emitted[:, : idx + 1]
                            break
                    break

            if generated_count >= max_new_tokens:
                break

            kv_accepted = truncate_kv_cache(ver_kv, cur_len + m)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [
                        cur_mask.to(dtype=torch.bool),
                        torch.ones((bsz, m + 1), device=device, dtype=torch.bool),
                    ],
                    dim=1,
                )
            commit_out = self.forward_exact(
                bonus_tok,
                attention_mask=cur_mask,
                past_key_values=kv_accepted,
                use_cache=True,
                quantize_kv=quantize_kv,
            )
            kv = commit_out["past_key_values"]
            last_logits = commit_out["logits"][:, -1, :]
            cur_len = cur_len + m + 1
            step_acc = float(m) / float(max(draft_len, 1))
            ema_accept = 0.5 * ema_accept + 0.5 * step_acc

        token_ids = torch.cat(emitted_chunks, dim=1)[:, :max_new_tokens]
        acceptance_rate = total_accepted / float(max(total_drafted, 1))
        return {
            "token_ids": token_ids,
            "acceptance_rate": float(acceptance_rate),
            "concept_steps": int(concept_steps),
            "verify_passes": int(verify_passes),
            "circuit_breaker_triggered": bool(circuit_breaker_triggered),
        }

    # ------------------------------------------------------------------
    # PG-FIM: Plan-Guided Fill-in-the-Middle Generation
    # ------------------------------------------------------------------
    def _build_syntactic_keyword_repair(
        self,
        prefix_str: str,
        current_middle: str,
        suffix_str: str,
        keywords: Sequence[str],
        language: str,
        check_fn: Callable[..., Dict[str, Any]],
    ) -> str:
        """Synthesize a clean AST-valid infill containing all planned keywords."""
        # Determine indentation from prefix trailing lines
        indent = ""
        lines = prefix_str.splitlines()
        if lines:
            last_line = lines[-1]
            lead = len(last_line) - len(last_line.lstrip(" \t"))
            base_indent = last_line[:lead]
            if last_line.rstrip().endswith(":"):
                indent = base_indent + "    "
            else:
                indent = base_indent

        needs_leading_nl = prefix_str.endswith(":")
        needs_trailing_nl = bool(suffix_str) and not suffix_str.startswith(("\n", "\r"))
        lead_nl = "\n" if needs_leading_nl else ""
        trail_nl = "\n" if needs_trailing_nl else ""

        clean_kws = [k.strip() for k in keywords if k and k.strip()]
        ident_kws = [
            k for k in clean_kws
            if _IDENT_RE.fullmatch(k) and k not in ("return", "for", "if", "while", "def", "class", "pass")
        ]

        candidates: List[str] = []
        # 1. Try appending missing keywords to current_middle if it already parses partially
        if current_middle.strip():
            missing = [k for k in clean_kws if k not in current_middle]
            if missing:
                joined_missing = " ".join(missing)
                candidates.append(f"{current_middle.rstrip()}  # {joined_missing}{trail_nl}")

        # 2. Construct valid Python expressions/statements using the planned keywords
        if not clean_kws:
            candidates.extend([
                f"{lead_nl}{indent}return None{trail_nl}",
                "0",
                f"{lead_nl}{indent}_res = 0{trail_nl}",
            ])
        else:
            has_return = "return" in clean_kws
            expr_terms = ident_kws if ident_kws else [k for k in clean_kws if k != "return"]
            expr_str = " + ".join(expr_terms) if expr_terms else "0"
            comment_all = " | ".join(clean_kws)
            if has_return:
                candidates.extend([
                    f"{lead_nl}{indent}return {expr_str}{trail_nl}",
                    f"return {expr_str}",
                    f"{lead_nl}{indent}return 0  # {comment_all}{trail_nl}",
                ])
            else:
                target_var = ident_kws[0] if ident_kws else "_val"
                rhs = " + ".join(ident_kws[1:]) if len(ident_kws) > 1 else expr_str
                candidates.extend([
                    f"{expr_str}",
                    f"{lead_nl}{indent}{target_var} = {rhs}{trail_nl}",
                    f"{lead_nl}{indent}return {expr_str}  # {comment_all}{trail_nl}",
                    f"({expr_str})  # {comment_all}",
                ])

        for cand in candidates:
            chk = check_fn(prefix_str, cand, suffix_str, clean_kws, language=language)
            if chk.get("parses", False) and chk.get("keywords_present", False):
                return cand

        # Universal fallback: inline expression or comment-annotated valid statement
        fallback_comment = " ".join(clean_kws)
        for base_stmt in (
            f"{lead_nl}{indent}_ok = 1  # {fallback_comment}\n",
            f"0  # {fallback_comment}\n",
            f"\n{indent}_ok = 1  # {fallback_comment}\n",
        ):
            chk = check_fn(prefix_str, base_stmt, suffix_str, clean_kws, language=language)
            if chk.get("parses", False) and chk.get("keywords_present", False):
                return base_stmt
        return candidates[0] if candidates else current_middle

    @torch.no_grad()
    def generate_fim(
        self,
        prefix: Union[str, torch.Tensor],
        suffix: Union[str, torch.Tensor],
        tokenizer: Optional[Any] = None,
        max_chunks: int = 4,
        temperature: float = 0.0,
        early_exit_conf: Optional[float] = 0.92,
        quantize_kv: bool = False,
        use_plan: bool = True,
        enable_repair: bool = True,
        allow_template_fallback: bool = False,
        language: str = "python",
        max_plan_tokens: int = 16,
        oracle_keywords: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Plan-Guided Fill-in-the-Middle (PG-FIM) generation.

        Pipeline:
          1. Format & prefill `<fim_prefix> P <fim_suffix> S <fim_middle>` into Causal Prefix KV.
          2. Exact-AR Plan Stage (`use_plan=True`): emit `<mlen_k>` + `<kw> kw1 | kw2 </kw>` (or `<kw/>`)
             under `KeywordPlanMasker` (hard format rules + soft symbol-table identifier bias).
          3. Parallel Concept Mode Middle Generation: generate `planned_chunks` chunks in <= K
             coarse-to-fine steps each (passing `kw_token_ids` to Phase B Copy-Aware Decoder if active).
          4. Verify & 1-Pass Self-Speculative Exact-AR Span Repair (Phase A): verify `P + M + S`
             parses and contains all planned keywords; if not and `enable_repair=True`, run 1-pass
             parallel Exact-AR verification/repair conditioned on the cached prefix+plan KV.
        """
        was_training = self.training
        self.eval()

        try:
            device = self.exact_embed.weight.device
            tok = (
                tokenizer
                if tokenizer is not None
                else CNCMTextTokenizer(vocab_size=self.cfg.vocab_size)
            )

            # --------------------------------------------------------------
            # Step 1: Normalize prefix & suffix into PSM causal prompt
            # --------------------------------------------------------------
            if isinstance(prefix, str) and isinstance(suffix, str):
                prefix_str = prefix
                suffix_str = suffix
                _, psm_list = tok.format_fim_prompt(
                    prefix_str,
                    suffix_str,
                    prefix_budget=self.cfg.fim_prefix_budget,
                    suffix_budget=self.cfg.fim_suffix_budget,
                )
                psm_ids = torch.tensor([psm_list], dtype=torch.long, device=device)
            else:
                if isinstance(prefix, str):
                    prefix_str = prefix
                    p_ids_list = tok.encode(prefix_str, add_special_tokens=False)
                    p_t = torch.tensor([p_ids_list], dtype=torch.long, device=device)
                else:
                    p_t = prefix.to(device=device, dtype=torch.long)
                    if p_t.ndim == 1:
                        p_t = p_t.unsqueeze(0)
                    prefix_str = tok.decode(p_t[0].tolist(), skip_special_tokens=True)

                if isinstance(suffix, str):
                    suffix_str = suffix
                    s_ids_list = tok.encode(suffix_str, add_special_tokens=False)
                    s_t = torch.tensor([s_ids_list], dtype=torch.long, device=device)
                else:
                    s_t = suffix.to(device=device, dtype=torch.long)
                    if s_t.ndim == 1:
                        s_t = s_t.unsqueeze(0)
                    suffix_str = tok.decode(s_t[0].tolist(), skip_special_tokens=True)

                bsz_in = int(p_t.shape[0])
                if s_t.shape[0] != bsz_in and s_t.shape[0] == 1:
                    s_t = s_t.expand(bsz_in, -1)

                # If caller already passed a formatted PSM tensor in `prefix`
                if (
                    p_t.shape[1] >= 3
                    and int(p_t[0, 0].item()) == self.cfg.fim_prefix_id
                    and int(p_t[0, -1].item()) == self.cfg.fim_middle_id
                    and s_t.shape[1] == 0
                ):
                    psm_ids = p_t
                else:
                    if self.cfg.fim_prefix_budget > 0 and p_t.shape[1] > self.cfg.fim_prefix_budget:
                        p_t = p_t[:, -self.cfg.fim_prefix_budget :]
                    if self.cfg.fim_suffix_budget > 0 and s_t.shape[1] > self.cfg.fim_suffix_budget:
                        s_t = s_t[:, : self.cfg.fim_suffix_budget]

                    fp_col = torch.full((bsz_in, 1), self.cfg.fim_prefix_id, device=device, dtype=torch.long)
                    fs_col = torch.full((bsz_in, 1), self.cfg.fim_suffix_id, device=device, dtype=torch.long)
                    fm_col = torch.full((bsz_in, 1), self.cfg.fim_middle_id, device=device, dtype=torch.long)
                    psm_ids = torch.cat([fp_col, p_t, fs_col, s_t, fm_col], dim=1)

            bsz = int(psm_ids.shape[0])

            # Resolve code symbol analyzer & grammar utilities (if available)
            symbol_info: Any = None
            check_fn: Callable[..., Dict[str, Any]] = _fallback_check_syntax_and_keywords
            try:
                from cncm.code.symbols import (  # type: ignore[import-not-found]
                    CodeSymbolTable,
                    check_syntax_and_keywords as _ext_check,
                )

                check_fn = _ext_check
                sym_table = CodeSymbolTable()
                symbol_info = sym_table.extract_symbols(
                    prefix_str, suffix_str, language=language
                )
            except Exception:
                symbol_info = None

            # Extract candidate identifiers from prefix + suffix for fallback plan masking
            cand_idents: List[str] = []
            if symbol_info is not None and hasattr(symbol_info, "all_identifiers"):
                cand_idents = sorted(list(getattr(symbol_info, "all_identifiers")))
            if not cand_idents:
                seen_id: set[str] = set()
                for m in _IDENT_RE.finditer(prefix_str + "\n" + suffix_str):
                    w = m.group(0)
                    if w not in seen_id and w not in ("def", "class", "if", "else", "for", "while", "in", "pass"):
                        seen_id.add(w)
                        cand_idents.append(w)

            # --------------------------------------------------------------
            # Step 2: Exact-AR Plan Stage (<mlen_k> + <kw> ... </kw> | <kw/>)
            # --------------------------------------------------------------
            plan_ids = torch.empty((bsz, 0), device=device, dtype=torch.long)
            plan_steps = 0
            planned_chunks = 1
            keywords: List[str] = list(oracle_keywords) if oracle_keywords is not None else []
            plan_kv: Optional[List[LayerKV]] = None
            plan_last_logits: Optional[torch.Tensor] = None

            if use_plan:
                masker: Any = None
                try:
                    from cncm.agent.grammar import KeywordPlanMasker  # type: ignore[import-not-found]

                    sig = inspect.signature(KeywordPlanMasker.__init__)
                    candidate_kwargs: Dict[str, Any] = {
                        "tokenizer": tok,
                        "cfg": self.cfg,
                        "config": self.cfg,
                        "symbol_info": symbol_info,
                        "symbols": symbol_info,
                        "prefix": prefix_str,
                        "suffix": suffix_str,
                        "language": language,
                        "kw_max": self.cfg.kw_max,
                        "max_plan_tokens": max_plan_tokens,
                        "max_tokens": max_plan_tokens,
                        "prefix_len": int(psm_ids.shape[1]),
                    }
                    accepts_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                    if accepts_var_kw:
                        init_kwargs = candidate_kwargs
                    else:
                        init_kwargs = {
                            k: v
                            for k, v in candidate_kwargs.items()
                            if k in sig.parameters and v is not None
                        }
                    masker = KeywordPlanMasker(**init_kwargs)
                except Exception:
                    masker = _FallbackKeywordPlanMasker(
                        cfg=self.cfg,
                        tokenizer=tok,
                        candidate_keywords=keywords if keywords else cand_idents,
                        prefix_len=int(psm_ids.shape[1]),
                        max_plan_tokens=max_plan_tokens,
                    )

                plan_stop_ids = [
                    self.cfg.kw_end_id,
                    self.cfg.kw_empty_id,
                    self.cfg.eos_token_id,
                ]
                plan_out = self.generate_exact(
                    prefix_ids=psm_ids,
                    max_new_tokens=max_plan_tokens,
                    temperature=temperature,
                    stop_token_ids=plan_stop_ids,
                    logit_mask_fn=masker,
                    past_key_values=None,
                    quantize_kv=quantize_kv,
                )
                plan_ids = plan_out["token_ids"]
                plan_kv = plan_out["past_key_values"]
                plan_steps = int(plan_ids.shape[1])

                # Commit any uncached trailing plan token(s) into plan_kv so both
                # Concept Mode and Phase A 1-pass repair share the exact full prefix+plan KV
                full_psm_plan = torch.cat([psm_ids, plan_ids], dim=1)
                cached_so_far = get_kv_seq_len(plan_kv)
                if cached_so_far < full_psm_plan.shape[1]:
                    tail_plan = full_psm_plan[:, cached_so_far:]
                    plan_kv, _, plan_last_logits = self._prefill_or_extend_prefix(
                        prefix_ids=tail_plan,
                        prefix_kv=plan_kv,
                        prefix_len=cached_so_far,
                        prefix_mask=None,
                        quantize_kv=quantize_kv,
                    )

                # Parse plan tokens into (planned_chunks, keywords)
                parsed_plan: Optional[Dict[str, Any]] = None
                try:
                    from cncm.agent.grammar import parse_plan_tokens  # type: ignore[import-not-found]

                    parsed_plan = parse_plan_tokens(tok, plan_ids[0].tolist())
                except Exception:
                    parsed_plan = _fallback_parse_plan_tokens(
                        self.cfg, tok, plan_ids[0].tolist()
                    )

                if parsed_plan is not None:
                    raw_chunks = int(parsed_plan.get("num_chunks", 1))
                    planned_chunks = max(1, min(raw_chunks, int(max_chunks)))
                    if oracle_keywords is None:
                        keywords = [
                            str(k).strip()
                            for k in parsed_plan.get("keywords", [])
                            if str(k).strip()
                        ][: self.cfg.kw_max]
            else:
                # Prefill PSM context directly into plan_kv
                plan_kv, _, plan_last_logits = self._prefill_or_extend_prefix(
                    prefix_ids=psm_ids,
                    prefix_kv=None,
                    prefix_len=0,
                    prefix_mask=None,
                    quantize_kv=quantize_kv,
                )
                planned_chunks = 1

            # Encode planned keywords into kw_token_ids for Phase B Copy-Aware Stage 1 Decoder
            kw_token_ids: Optional[torch.Tensor] = None
            if keywords:
                flat_kw_toks: List[int] = []
                for kw in keywords:
                    for tid in tok.encode(kw, add_special_tokens=False):
                        if int(tid) not in flat_kw_toks and int(tid) != self.cfg.pad_token_id:
                            flat_kw_toks.append(int(tid))
                if flat_kw_toks:
                    kw_token_ids = (
                        torch.tensor(flat_kw_toks, dtype=torch.long, device=device)
                        .unsqueeze(0)
                        .expand(bsz, -1)
                    )

            # --------------------------------------------------------------
            # Step 3: Parallel Concept Mode Middle Generation
            # --------------------------------------------------------------
            cur_kv = plan_kv
            cur_plen = get_kv_seq_len(cur_kv)
            next_prefix: Optional[torch.Tensor] = None
            middle_chunks: List[torch.Tensor] = []
            concept_steps = 0
            fim_stop_ids = {self.cfg.fim_end_id, self.cfg.eos_token_id}

            for _chunk_idx in range(planned_chunks):
                c_out = self.generate_concept_chunk(
                    prefix_ids=next_prefix,
                    prefix_kv=cur_kv,
                    prefix_len=cur_plen,
                    temperature=temperature,
                    early_exit_conf=early_exit_conf,
                    quantize_kv=quantize_kv,
                    return_codes=False,
                    kw_token_ids=kw_token_ids,
                )
                chunk_ids: torch.Tensor = c_out["token_ids"]
                cur_kv = c_out["prefix_kv"]
                cur_plen = int(c_out["prefix_len"])
                concept_steps += int(c_out["exit_level"])

                # Cut at <fim_end> or <|eos|> if emitted inside this chunk
                seq0 = chunk_ids[0].tolist()
                cut_pos: Optional[int] = None
                for idx, tid in enumerate(seq0):
                    if int(tid) in fim_stop_ids:
                        cut_pos = idx
                        break

                if cut_pos is not None:
                    if cut_pos > 0:
                        middle_chunks.append(chunk_ids[:, :cut_pos])
                    break

                middle_chunks.append(chunk_ids)
                next_prefix = chunk_ids

            if middle_chunks:
                middle_ids = torch.cat(middle_chunks, dim=1)
            else:
                middle_ids = torch.empty((bsz, 0), device=device, dtype=torch.long)

            clean_mid_list = [
                int(t)
                for t in middle_ids[0].tolist()
                if int(t) not in (self.cfg.pad_token_id, self.cfg.fim_end_id, self.cfg.eos_token_id)
            ]
            middle_str = tok.decode(clean_mid_list, skip_special_tokens=True)

            # --------------------------------------------------------------
            # Step 4: Verify & 1-Pass Self-Speculative Exact-AR Span Repair (Phase A)
            # --------------------------------------------------------------
            chk = check_fn(prefix_str, middle_str, suffix_str, keywords, language=language)
            parses = bool(chk.get("parses", False))
            keywords_present = bool(chk.get("keywords_present", True))
            repair_triggered = False
            repair_passes = 0
            repair_source = "none"
            model_generated = True

            if enable_repair and (not parses or not keywords_present):
                repair_triggered = True

                # True Local-Span Autoregressive Repair (#6):
                # 1. Locate the failing line/offset in `middle_str` via ast.parse on `P + M + S`
                # 2. Keep the valid prefix of `middle_str` up to the last safe newline before the fault
                # 3. Roll back KV cache to `plan_kv` + valid prefix and run `generate_exact` for
                #    up to `16` tokens so every repaired token is autoregressively conditioned on
                #    previous repaired tokens (fixing the broken non-AR parallel splice!).
                safe_prefix_str = ""
                if middle_str and language.lower() in ("python", "py"):
                    try:
                        import ast as _ast

                        _ast.parse(prefix_str + middle_str + suffix_str)
                    except SyntaxError as syn_err:
                        p_lines = prefix_str.count("\n")
                        err_line_in_mid = (int(syn_err.lineno or 1) - 1) - p_lines
                        mid_lines = middle_str.splitlines(keepends=True)
                        if 0 < err_line_in_mid <= len(mid_lines):
                            safe_prefix_str = "".join(mid_lines[: err_line_in_mid])
                    except Exception:
                        safe_prefix_str = ""

                safe_prefix_toks = (
                    tok.encode(safe_prefix_str, add_special_tokens=False)
                    if safe_prefix_str
                    else []
                )
                repair_kv = plan_kv
                if safe_prefix_toks:
                    sp_t = (
                        torch.tensor(safe_prefix_toks, dtype=torch.long, device=device)
                        .unsqueeze(0)
                        .expand(bsz, -1)
                    )
                    repair_kv, _, _ = self._prefill_or_extend_prefix(
                        prefix_ids=sp_t,
                        prefix_kv=plan_kv,
                        prefix_len=get_kv_seq_len(plan_kv),
                        prefix_mask=None,
                        quantize_kv=quantize_kv,
                    )
                    repair_seed = sp_t[:, -1:]
                    repair_kv_for_gen = truncate_kv_cache(repair_kv, max(0, get_kv_seq_len(repair_kv) - 1))
                else:
                    # Seed from the last token already in psm_ids + plan_ids
                    full_cond = torch.cat([psm_ids, plan_ids], dim=1) if plan_ids.shape[1] > 0 else psm_ids
                    repair_seed = full_cond[:, -1:]
                    repair_kv_for_gen = truncate_kv_cache(plan_kv, max(0, get_kv_seq_len(plan_kv) - 1))

                # Soft keyword logit bias during local-span AR repair
                kw_bias_ids = kw_token_ids[0].tolist() if (kw_token_ids is not None and kw_token_ids.numel() > 0) else []

                def _kw_boost_mask(_gen_ids: torch.Tensor, logits_in: torch.Tensor) -> torch.Tensor:
                    if not kw_bias_ids:
                        return logits_in
                    out_l = logits_in.clone()
                    for tid in kw_bias_ids:
                        if 0 <= int(tid) < out_l.shape[-1]:
                            out_l[..., int(tid)] = out_l[..., int(tid)] + 1.5
                    return out_l

                ar_rep_budget = min(16, max(8, self.cfg.chunk_size // 4))
                ar_rep_out = self.generate_exact(
                    prefix_ids=repair_seed,
                    max_new_tokens=ar_rep_budget,
                    temperature=0.0,
                    stop_token_ids=[self.cfg.fim_end_id, self.cfg.eos_token_id],
                    logit_mask_fn=_kw_boost_mask if kw_bias_ids else None,
                    past_key_values=repair_kv_for_gen,
                    quantize_kv=quantize_kv,
                )
                rep_tail_ids: torch.Tensor = ar_rep_out["token_ids"]
                repair_passes = int(rep_tail_ids.shape[1])

                if safe_prefix_toks:
                    sp_t_full = (
                        torch.tensor(safe_prefix_toks, dtype=torch.long, device=device)
                        .unsqueeze(0)
                        .expand(bsz, -1)
                    )
                    spliced_ids = torch.cat([sp_t_full, rep_tail_ids], dim=1)
                else:
                    spliced_ids = rep_tail_ids

                spliced_list = [
                    int(t)
                    for t in spliced_ids[0].tolist()
                    if int(t) not in (self.cfg.pad_token_id, self.cfg.fim_end_id, self.cfg.eos_token_id)
                ]
                spliced_str = tok.decode(spliced_list, skip_special_tokens=True)
                chk_spliced = check_fn(
                    prefix_str, spliced_str, suffix_str, keywords, language=language
                )

                if chk_spliced.get("parses", False) and chk_spliced.get("keywords_present", True):
                    middle_ids = spliced_ids
                    middle_str = spliced_str
                    parses = True
                    keywords_present = True
                    repair_source = "exact_ar_model"
                elif allow_template_fallback:
                    # Explicit opt-in syntactic + keyword template upper-bound fallback
                    repaired_str = self._build_syntactic_keyword_repair(
                        prefix_str=prefix_str,
                        current_middle=spliced_str if spliced_str.strip() else middle_str,
                        suffix_str=suffix_str,
                        keywords=keywords,
                        language=language,
                        check_fn=check_fn,
                    )
                    chk_rep = check_fn(
                        prefix_str, repaired_str, suffix_str, keywords, language=language
                    )
                    rep_toks = tok.encode(repaired_str, add_special_tokens=False)
                    if rep_toks:
                        middle_ids = (
                            torch.tensor(rep_toks, dtype=torch.long, device=device)
                            .unsqueeze(0)
                            .expand(bsz, -1)
                        )
                    middle_str = repaired_str
                    parses = bool(chk_rep.get("parses", False))
                    keywords_present = bool(chk_rep.get("keywords_present", True))
                    repair_source = "template_fallback"
                    model_generated = False
                else:
                    # Honest model-only output: keep local AR repair if it improved syntax or keywords
                    if chk_spliced.get("parses", False) or (
                        not keywords_present and chk_spliced.get("keywords_present", False)
                    ):
                        middle_ids = spliced_ids
                        middle_str = spliced_str
                        parses = bool(chk_spliced.get("parses", False))
                        keywords_present = bool(chk_spliced.get("keywords_present", True))
                        repair_source = "exact_ar_model"
                    else:
                        repair_source = "exact_ar_model_unrepaired"

            sequential_passes = int(plan_steps + concept_steps + repair_passes)
            ar_baseline_passes = float(
                max(
                    int(middle_ids.shape[1]),
                    self.cfg.chunk_size * max(1, len(middle_chunks)),
                )
            )
            speedup_vs_ar = ar_baseline_passes / float(max(sequential_passes, 1))

            return {
                "token_ids": middle_ids,
                "middle_text": middle_str,
                "full_text": prefix_str + middle_str + suffix_str,
                "plan_token_ids": plan_ids,
                "planned_chunks": int(planned_chunks),
                "keywords": list(keywords),
                "parses": bool(parses),
                "keywords_present": bool(keywords_present),
                "repair_triggered": bool(repair_triggered),
                "repair_source": repair_source,
                "model_generated": bool(model_generated),
                "sequential_passes": int(sequential_passes),
                "concept_steps": int(concept_steps),
                "plan_steps": int(plan_steps),
                "repair_passes": int(repair_passes),
                "speedup_vs_ar": float(speedup_vs_ar),
            }
        finally:
            if was_training:
                self.train(True)
