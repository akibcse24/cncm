"""Core pure-PyTorch layers for CNCM (CPU, DirectML, MPS, CUDA, ONNX 17+ safe).

Includes all verified architectural & engineering fixes:
  - 2D-safe RoPE (`cos.unsqueeze(-2)` / `sin.unsqueeze(-2)` on [B, S, H, D] layout)
  - Explicit 4D additive SDPA masks (`is_causal=False`, zero `torch.all(mask==1)`)
  - Exact bottom-right causal chunk-commit mask (`k_idx <= past_len + q_idx`)
  - Pre-RMSNorm + QK-RMSNorm + GQA + SwiGLU + Relaxed Recursive Loop Adapters
  - Optional per-token per-head INT8 KV cache compression (`QuantizedKV`)
  - Optional QAT fake-quantized `BitQuantLinear(nn.Linear)`
  - `FactorizedConceptHead`: chained micro-AR head over `d_q` FSQ digits
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cncm.config import CNCMConfig


# ----------------------------------------------------------------------
# Normalization & QAT Linear
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Pure-PyTorch Root Mean Square Layer Normalization (ONNX opset 17 safe)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


class BitQuantLinear(nn.Linear):
    """QAT fake-quantized Linear (inherits nn.Linear for seamless PTQ/INT4 export)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits) if act_bits is not None else None
        self.qat_enabled: bool = True

    @staticmethod
    def _fake_quant_sym(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
        q_max = float((1 << (bits - 1)) - 1)
        scale = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-6) / q_max
        x_q = torch.clamp(torch.round(x / scale), -q_max, q_max) * scale
        return x + (x_q - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.qat_enabled:
            return F.linear(x, self.weight, self.bias)
        w = self._fake_quant_sym(self.weight, self.weight_bits, dim=-1)
        if self.act_bits is not None:
            x = self._fake_quant_sym(x, self.act_bits, dim=-1)
        return F.linear(x, w, self.bias)


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    use_qat: bool = False,
    weight_bits: int = 8,
    act_bits: Optional[int] = 8,
) -> nn.Linear:
    if use_qat:
        return BitQuantLinear(
            in_features,
            out_features,
            bias=bias,
            weight_bits=weight_bits,
            act_bits=act_bits,
        )
    return nn.Linear(in_features, out_features, bias=bias)


# ----------------------------------------------------------------------
# 2D-Safe Rotary Positional Embeddings
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Rotary position embedding supporting arbitrary 1D [S] or 2D [B, S] position_ids."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        theta: float = 100000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_position_embeddings = int(max_position_embeddings)
        self.theta = float(theta)

        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape [*position_ids.shape, head_dim].

        Supports 1D [S] -> [S, D] and 2D [B, S] -> [B, S, D].
        """
        pos = position_ids.to(device=self.inv_freq.device, dtype=torch.long)
        # During ONNX export or out-of-range positions, compute directly from inv_freq
        if torch.onnx.is_in_onnx_export():
            freqs = pos.float().unsqueeze(-1) * self.inv_freq
            emb = torch.cat([freqs, freqs], dim=-1)
            cos, sin = emb.cos(), emb.sin()
        else:
            max_pos = int(pos.max().item()) if pos.numel() > 0 else 0
            if max_pos >= self.cos_cached.shape[0]:
                new_len = max(max_pos + 1, self.cos_cached.shape[0] * 2)
                self._build_cache(new_len)
            cos = self.cos_cached[pos]
            sin = self.sin_cached[pos]

        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q [B, S, H_q, D] and k [B, S, H_kv, D].

    CRITICAL 2D RoPE FIX: Uses `cos.unsqueeze(-2)` / `sin.unsqueeze(-2)` so:
      - 1D cos [S, D]    -> [S, 1, D]    broadcasts onto [B, S, H, D]
      - 2D cos [B, S, D] -> [B, S, 1, D] broadcasts onto [B, S, H, D]
    Never silently swaps S and H axes even when S == num_attention_heads!
    """
    cos_b = cos.unsqueeze(-2)
    sin_b = sin.unsqueeze(-2)
    q_embed = (q * cos_b) + (rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (rotate_half(k) * sin_b)
    return q_embed, k_embed


# ----------------------------------------------------------------------
# KV Cache (FP and Optional Per-Token Per-Head INT8 Compression)
# ----------------------------------------------------------------------
@dataclass
class QuantizedKV:
    """Symmetric per-token per-head INT8 KV cache (~3.7x smaller than FP32)."""

    k_int8: torch.Tensor  # [B, H_kv, P, D], int8
    k_scale: torch.Tensor  # [B, H_kv, P, 1], float16
    v_int8: torch.Tensor  # [B, H_kv, P, D], int8
    v_scale: torch.Tensor  # [B, H_kv, P, 1], float16

    @property
    def seq_len(self) -> int:
        return int(self.k_int8.shape[2])

    def dequantize(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.k_int8.to(dtype) * self.k_scale.to(dtype)
        v = self.v_int8.to(dtype) * self.v_scale.to(dtype)
        return k, v

    def truncate(self, length: int) -> "QuantizedKV":
        return QuantizedKV(
            k_int8=self.k_int8[:, :, :length, :],
            k_scale=self.k_scale[:, :, :length, :],
            v_int8=self.v_int8[:, :, :length, :],
            v_scale=self.v_scale[:, :, :length, :],
        )

    def nbytes(self) -> int:
        return (
            self.k_int8.numel() * self.k_int8.element_size()
            + self.k_scale.numel() * self.k_scale.element_size()
            + self.v_int8.numel() * self.v_int8.element_size()
            + self.v_scale.numel() * self.v_scale.element_size()
        )


LayerKV = Union[Tuple[torch.Tensor, torch.Tensor], QuantizedKV]


def quantize_kv_pair(k: torch.Tensor, v: torch.Tensor) -> QuantizedKV:
    """Quantize [B, H_kv, S, D] key/value tensors into symmetric INT8 + FP16 scales."""
    k_scale = (k.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0).to(
        torch.float16
    )
    v_scale = (v.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0).to(
        torch.float16
    )
    k_int8 = torch.clamp(torch.round(k.float() / k_scale.float()), -127, 127).to(
        torch.int8
    )
    v_int8 = torch.clamp(torch.round(v.float() / v_scale.float()), -127, 127).to(
        torch.int8
    )
    return QuantizedKV(k_int8=k_int8, k_scale=k_scale, v_int8=v_int8, v_scale=v_scale)


def append_kv_cache(
    past_kv: Optional[LayerKV],
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    quantize_kv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, LayerKV]:
    """Concatenate past KV with new [B, H_kv, Q, D] tensors and build updated cache."""
    if past_kv is None:
        k_full, v_full = k_new, v_new
        if quantize_kv:
            present: LayerKV = quantize_kv_pair(k_new, v_new)
        else:
            present = (k_full, v_full)
        return k_full, v_full, present

    if isinstance(past_kv, QuantizedKV):
        k_past, v_past = past_kv.dequantize(k_new.dtype)
        k_full = torch.cat([k_past, k_new], dim=2)
        v_full = torch.cat([v_past, v_new], dim=2)
        if quantize_kv:
            new_q = quantize_kv_pair(k_new, v_new)
            present = QuantizedKV(
                k_int8=torch.cat([past_kv.k_int8, new_q.k_int8], dim=2),
                k_scale=torch.cat([past_kv.k_scale, new_q.k_scale], dim=2),
                v_int8=torch.cat([past_kv.v_int8, new_q.v_int8], dim=2),
                v_scale=torch.cat([past_kv.v_scale, new_q.v_scale], dim=2),
            )
        else:
            present = (k_full, v_full)
        return k_full, v_full, present

    k_past, v_past = past_kv
    k_full = torch.cat([k_past, k_new], dim=2)
    v_full = torch.cat([v_past, v_new], dim=2)
    if quantize_kv:
        present = quantize_kv_pair(k_full, v_full)
    else:
        present = (k_full, v_full)
    return k_full, v_full, present


def get_kv_seq_len(past_key_values: Optional[Sequence[LayerKV]]) -> int:
    if not past_key_values:
        return 0
    first = past_key_values[0]
    if isinstance(first, QuantizedKV):
        return first.seq_len
    return int(first[0].shape[2])


def truncate_kv_cache(
    past_key_values: Optional[Sequence[LayerKV]], length: int
) -> Optional[List[LayerKV]]:
    if past_key_values is None:
        return None
    if length <= 0:
        return None
    out: List[LayerKV] = []
    for entry in past_key_values:
        if isinstance(entry, QuantizedKV):
            out.append(entry.truncate(length))
        else:
            k, v = entry
            out.append((k[:, :, :length, :], v[:, :, :length, :]))
    return out


def kv_cache_nbytes(past_key_values: Optional[Sequence[LayerKV]]) -> int:
    if not past_key_values:
        return 0
    total = 0
    for entry in past_key_values:
        if isinstance(entry, QuantizedKV):
            total += entry.nbytes()
        else:
            k, v = entry
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


# ----------------------------------------------------------------------
# Explicit 4D Additive Attention Masks (Zero `torch.all(mask==1)`)
# ----------------------------------------------------------------------
def _bool_to_additive_mask(
    allowed: torch.Tensor,
    dtype: torch.dtype,
    self_col_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert [B, 1, Q, K] bool mask (True=attend) into NaN-safe additive float mask."""
    # Ensure no query row is all-False (prevents NaN softmax on fully padded rows)
    if self_col_indices is not None:
        row_empty = ~allowed.any(dim=-1, keepdim=True)
        k_idx = torch.arange(allowed.shape[-1], device=allowed.device)
        diag = k_idx.view(1, 1, 1, -1) == self_col_indices.view(1, 1, -1, 1)
        allowed = allowed | (row_empty & diag)

    neg_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    zero = torch.zeros((), device=allowed.device, dtype=dtype)
    neg = torch.full((), neg_val, device=allowed.device, dtype=dtype)
    return torch.where(allowed, zero, neg)


def build_causal_mask(
    q_len: int,
    kv_len: int,
    past_len: int,
    batch_size: int,
    key_padding_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build explicit 4D bottom-right causal mask [B, 1, Q, K] (`k_idx <= past_len + q_idx`).

    Handles arbitrary chunk commits onto cached prefixes (`kv_len == past_len + q_len`).
    """
    q_idx = torch.arange(q_len, device=device, dtype=torch.long)
    k_idx = torch.arange(kv_len, device=device, dtype=torch.long)
    causal = k_idx.unsqueeze(0) <= (past_len + q_idx.unsqueeze(1))  # [Q, K]
    allowed = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, kv_len)

    if key_padding_mask is not None:
        kpm = key_padding_mask.to(device=device, dtype=torch.bool)
        if kpm.shape[1] == q_len and past_len > 0:
            past_valid = torch.ones(
                (batch_size, past_len), device=device, dtype=torch.bool
            )
            kpm = torch.cat([past_valid, kpm], dim=1)
        allowed = allowed & kpm[:, None, None, :]

    self_cols = (past_len + q_idx).clamp(max=kv_len - 1)
    return _bool_to_additive_mask(allowed, dtype=dtype, self_col_indices=self_cols)


def build_concept_mask(
    prefix_len: int,
    level_lengths: Sequence[int],
    batch_size: int,
    prefix_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build 4D Causal-Prefix + Block-Diagonal Target mask [B, 1, P+T, P+T].

    - Prefix [0:P] attends CAUSALLY to prefix [0:P] only (enables incremental KV reuse).
    - Target level k attends to all valid prefix tokens [0:P] + bidirectionally within level k.
    - Cross-level target attention is strictly blocked (zero target KV cache growth).
    """
    total_target = sum(level_lengths)
    total_len = prefix_len + total_target
    allowed = torch.zeros((total_len, total_len), device=device, dtype=torch.bool)

    if prefix_len > 0:
        p_idx = torch.arange(prefix_len, device=device)
        allowed[:prefix_len, :prefix_len] = p_idx.unsqueeze(0) <= p_idx.unsqueeze(1)
        allowed[prefix_len:, :prefix_len] = True

    offset = prefix_len
    for l_k in level_lengths:
        allowed[offset : offset + l_k, offset : offset + l_k] = True
        offset += l_k

    allowed_4d = allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total_len, total_len)

    if prefix_mask is not None and prefix_len > 0:
        pm = prefix_mask.to(device=device, dtype=torch.bool)
        target_valid = torch.ones(
            (batch_size, total_target), device=device, dtype=torch.bool
        )
        key_valid = torch.cat([pm, target_valid], dim=1)
        allowed_4d = allowed_4d & key_valid[:, None, None, :]

    self_cols = torch.arange(total_len, device=device, dtype=torch.long)
    return _bool_to_additive_mask(allowed_4d, dtype=dtype, self_col_indices=self_cols)


def build_cached_concept_step_mask(
    prefix_len: int,
    level_len: int,
    batch_size: int,
    prefix_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build 4D mask [B, 1, l_k, P + l_k] for a cached concept level step."""
    kv_len = prefix_len + level_len
    allowed = torch.ones((batch_size, 1, level_len, kv_len), device=device, dtype=torch.bool)
    if prefix_mask is not None and prefix_len > 0:
        pm = prefix_mask.to(device=device, dtype=torch.bool)
        target_valid = torch.ones(
            (batch_size, level_len), device=device, dtype=torch.bool
        )
        key_valid = torch.cat([pm, target_valid], dim=1)
        allowed = allowed & key_valid[:, None, None, :]
    self_cols = torch.arange(prefix_len, kv_len, device=device, dtype=torch.long)
    return _bool_to_additive_mask(allowed, dtype=dtype, self_col_indices=self_cols)


def build_bidirectional_mask(
    seq_len: int,
    batch_size: int,
    attention_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build 4D bidirectional mask [B, 1, L, L] for the Stage 1 Concept Decoder."""
    allowed = torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.bool)
    if attention_mask is not None:
        kpm = attention_mask.to(device=device, dtype=torch.bool)
        allowed = allowed & kpm[:, None, None, :]
    self_cols = torch.arange(seq_len, device=device, dtype=torch.long)
    return _bool_to_additive_mask(allowed, dtype=dtype, self_col_indices=self_cols)


# ----------------------------------------------------------------------
# Attention, SwiGLU, Loop Adapter & Transformer Blocks
# ----------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads [B, H_kv, S, D] -> [B, H_kv * n_rep, S, D]."""
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def scaled_dot_product_attn_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Explicit-4D SDPA (`is_causal=False`) with ONNX opset 17 matmul fallback."""
    if torch.onnx.is_in_onnx_export():
        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_mask
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(probs, v)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (bias-free)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_qat: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.gate_proj = make_linear(
            hidden_size, intermediate_size, bias=False, use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits
        )
        self.up_proj = make_linear(
            hidden_size, intermediate_size, bias=False, use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits
        )
        self.down_proj = make_linear(
            intermediate_size, hidden_size, bias=False, use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """DeepSeek-style Fine-Grained Mixture-of-Experts FFN (1 Shared + N Routed Experts, Top-K).

    Supports external per-loop routers (`router` argument in `forward`) so recursive loops
    can route tokens to different experts on shallow vs. deep passes without adding extra
    active FLOPs. Always executes router softmax and load-balancing loss in FP32 for T4 FP16 safety.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        expert_intermediate_size: int = 768,
        shared_intermediate_size: int = 768,
        use_internal_router: bool = False,
        use_qat: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_intermediate_size = int(expert_intermediate_size)
        self.shared_intermediate_size = int(shared_intermediate_size)

        self.shared_expert: Optional[SwiGLU] = (
            SwiGLU(
                hidden_size=self.hidden_size,
                intermediate_size=self.shared_intermediate_size,
                use_qat=use_qat,
                weight_bits=weight_bits,
                act_bits=act_bits,
            )
            if self.shared_intermediate_size > 0
            else None
        )
        self.experts = nn.ModuleList(
            [
                SwiGLU(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.expert_intermediate_size,
                    use_qat=use_qat,
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                )
                for _ in range(self.num_experts)
            ]
        )
        self.default_router: Optional[nn.Linear] = (
            nn.Linear(self.hidden_size, self.num_experts, bias=False)
            if use_internal_router
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        router: Optional[nn.Linear] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)
        n_tok = x_flat.shape[0]

        gate_proj = router if router is not None else self.default_router
        if gate_proj is None:
            raise RuntimeError("MoEFFN requires either an external `router` or `use_internal_router=True`.")

        # FP32 router softmax prevents FP16 underflow/collapse on T4 GPUs
        router_logits = F.linear(x_flat.float(), gate_proj.weight.float())  # [N, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]

        topk_vals, topk_idx = torch.topk(router_probs, k=self.top_k, dim=-1)  # [N, K]
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Switch / DeepSeek Load-Balancing Auxiliary Loss (FP32, minimum = 1.0 when perfectly balanced)
        one_hot = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # [N, K, E]
        f_e = one_hot.sum(dim=(0, 1)) / float(max(1, n_tok * self.top_k))
        p_e = router_probs.mean(dim=0)
        aux_loss = float(self.num_experts) * torch.sum(f_e * p_e)

        out = self.shared_expert(x_flat) if self.shared_expert is not None else torch.zeros_like(x_flat)
        routed_out = torch.zeros_like(x_flat)

        # Single-sort sparse dispatch: 1 index_select + 1 index_add_ instead of E separate nonzero() syncs
        flat_exp = topk_idx.reshape(-1)
        flat_w = topk_weights.reshape(-1)
        order = torch.argsort(flat_exp, stable=False)
        sorted_exp = flat_exp.index_select(0, order)
        sorted_tok = torch.div(order, self.top_k, rounding_mode="floor")
        sorted_w = flat_w.index_select(0, order).to(dtype=routed_out.dtype).unsqueeze(-1)
        counts = torch.bincount(sorted_exp, minlength=self.num_experts).tolist()

        x_chunks = x_flat.index_select(0, sorted_tok).split(counts, dim=0)
        out_chunks: List[torch.Tensor] = []
        for e_idx, (expert, c_n) in enumerate(zip(self.experts, counts)):
            if c_n == 0:
                continue
            out_chunks.append(expert(x_chunks[e_idx]).to(dtype=routed_out.dtype))
        if out_chunks:
            y_sorted = torch.cat(out_chunks, dim=0) * sorted_w
            routed_out.index_add_(0, sorted_tok, y_sorted)

        return (out.to(dtype=routed_out.dtype) + routed_out).to(dtype=x.dtype).view(bsz, seq_len, dim), aux_loss


class LoopAdapter(nn.Module):
    """Lightweight low-rank residual adapter specializing each recursive loop pass."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        eps: float = 1e-6,
        use_qat: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.down = make_linear(
            hidden_size, rank, bias=False, use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits
        )
        self.up = make_linear(
            rank, hidden_size, bias=False, use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits
        )
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(self.norm(x))))


class CNCMAttention(nn.Module):
    """Grouped-Query Attention with pre-RoPE QK-RMSNorm, 2D-safe RoPE, and INT8 KV support."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_qat: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_attention_heads)
        self.num_kv_heads = int(num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = float(dropout)

        self.q_proj = make_linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False,
            use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
        )
        self.k_proj = make_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
        )
        self.v_proj = make_linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False,
            use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
        )
        self.o_proj = make_linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False,
            use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
        )

        # Pre-RoPE QK-RMSNorm for training stability across recursive loops
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        past_kv: Optional[LayerKV] = None,
        use_cache: bool = False,
        quantize_kv: bool = False,
        commit_to_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[LayerKV]]:
        bsz, seq_len, _ = x.shape

        # Project and reshape to [B, S, H, D] BEFORE RoPE so cos.unsqueeze(-2) is 2D-safe
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose to [B, H, S, D] for KV cache & SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        present_kv: Optional[LayerKV] = None
        if past_kv is not None:
            if commit_to_cache:
                k_full, v_full, updated = append_kv_cache(
                    past_kv, k, v, quantize_kv=quantize_kv
                )
                if use_cache:
                    present_kv = updated
            else:
                # Attend to cached prefix + current level WITHOUT mutating prefix_kv
                if isinstance(past_kv, QuantizedKV):
                    k_past, v_past = past_kv.dequantize(k.dtype)
                else:
                    k_past, v_past = past_kv
                k_full = torch.cat([k_past, k], dim=2)
                v_full = torch.cat([v_past, v], dim=2)
                if use_cache:
                    present_kv = past_kv
        else:
            k_full, v_full = k, v
            if use_cache and commit_to_cache:
                present_kv = (
                    quantize_kv_pair(k_full, v_full) if quantize_kv else (k_full, v_full)
                )

        k_rep = repeat_kv(k_full, self.num_kv_groups)
        v_rep = repeat_kv(v_full, self.num_kv_groups)

        drop_p = self.dropout if self.training else 0.0
        attn_out = scaled_dot_product_attn_4d(
            q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=drop_p
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_out), present_kv


class CNCMBlock(nn.Module):
    """Shared Transformer block executed across Relaxed Recursive Loops (Dense SwiGLU or Fine-Grained MoE)."""

    def __init__(self, cfg: CNCMConfig) -> None:
        super().__init__()
        self.use_moe = bool(cfg.num_experts > 0)
        self.attn = CNCMAttention(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            rms_norm_eps=cfg.rms_norm_eps,
            dropout=cfg.dropout,
            use_qat=cfg.use_qat_linear,
            weight_bits=cfg.qat_weight_bits,
            act_bits=cfg.qat_act_bits,
        )
        if self.use_moe:
            self.mlp: nn.Module = MoEFFN(
                hidden_size=cfg.hidden_size,
                num_experts=cfg.num_experts,
                top_k=cfg.num_experts_per_tok,
                expert_intermediate_size=cfg.moe_expert_intermediate_size,
                shared_intermediate_size=cfg.moe_shared_intermediate_size,
                use_internal_router=False,
                use_qat=cfg.use_qat_linear,
                weight_bits=cfg.qat_weight_bits,
                act_bits=cfg.qat_act_bits,
            )
        else:
            self.mlp = SwiGLU(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                use_qat=cfg.use_qat_linear,
                weight_bits=cfg.qat_weight_bits,
                act_bits=cfg.qat_act_bits,
            )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_norm: RMSNorm,
        ffn_norm: RMSNorm,
        loop_adapter: Optional[LoopAdapter] = None,
        moe_router: Optional[nn.Linear] = None,
        past_kv: Optional[LayerKV] = None,
        use_cache: bool = False,
        quantize_kv: bool = False,
        commit_to_cache: bool = True,
        return_aux_loss: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[LayerKV]], Tuple[torch.Tensor, Optional[LayerKV], Optional[torch.Tensor]]]:
        h_attn, present_kv = self.attn(
            attn_norm(x),
            cos=cos,
            sin=sin,
            attn_mask=attn_mask,
            past_kv=past_kv,
            use_cache=use_cache,
            quantize_kv=quantize_kv,
            commit_to_cache=commit_to_cache,
        )
        x = x + h_attn
        aux_loss: Optional[torch.Tensor] = None
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(ffn_norm(x), router=moe_router)
            x = x + mlp_out
        else:
            x = x + self.mlp(ffn_norm(x))
        if loop_adapter is not None:
            x = loop_adapter(x)
        if return_aux_loss:
            return x, present_kv, aux_loss
        return x, present_kv


class CNCMDecoderBlock(nn.Module):
    """Lightweight bidirectional Transformer block for the Stage 1 Concept Decoder."""

    def __init__(self, cfg: CNCMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.d_vq, eps=cfg.rms_norm_eps)
        self.attn = CNCMAttention(
            hidden_size=cfg.d_vq,
            num_attention_heads=cfg.dec_num_heads,
            num_key_value_heads=cfg.dec_num_heads,
            rms_norm_eps=cfg.rms_norm_eps,
            dropout=cfg.dropout,
            use_qat=False,
        )
        self.mlp = SwiGLU(
            hidden_size=cfg.d_vq,
            intermediate_size=cfg.dec_intermediate_size,
            use_qat=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_attn, _ = self.attn(
            self.attn_norm(x),
            cos=cos,
            sin=sin,
            attn_mask=attn_mask,
            past_kv=None,
            use_cache=False,
        )
        x = x + h_attn
        x = x + self.mlp(self.ffn_norm(x))
        return x


# ----------------------------------------------------------------------
# Factorized Joint Concept Head
# ----------------------------------------------------------------------
class FactorizedConceptHead(nn.Module):
    """Chained autoregressive micro-head over `d_q` FSQ digits:

        p(c_0, ..., c_{d_q-1} | h, k) = prod_{j=0}^{d_q-1} p(c_j | h, k, c_{<j})

    Captures full joint correlation across all prod(fsq_levels) (e.g. 4096)
    implicit codes at <0.3M params and >40x fewer MACs than a flat 4096-way head.
    """

    def __init__(
        self,
        hidden_size: int,
        fsq_levels: Tuple[int, ...],
        num_levels: int,
        head_hidden: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        use_qat: bool = False,
        weight_bits: int = 8,
        act_bits: Optional[int] = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.fsq_levels = tuple(int(v) for v in fsq_levels)
        self.d_q = len(self.fsq_levels)
        self.num_levels = int(num_levels)
        self.head_hidden = (
            int(head_hidden)
            if head_hidden is not None
            else min(160, max(48, self.hidden_size // 4))
        )
        bottleneck = max(24, self.head_hidden // 2)

        self.ctx_proj = make_linear(
            self.hidden_size, self.head_hidden, bias=False,
            use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
        )
        self.level_emb = nn.Embedding(self.num_levels, self.head_hidden)
        self.ctx_norm = RMSNorm(self.head_hidden, eps=rms_norm_eps)

        self.digit_embs = nn.ModuleList(
            [nn.Embedding(self.fsq_levels[j], self.head_hidden) for j in range(self.d_q - 1)]
        )
        self.step_norms = nn.ModuleList(
            [RMSNorm(self.head_hidden, eps=rms_norm_eps) for _ in range(self.d_q - 1)]
        )
        self.step_down = nn.ModuleList(
            [
                make_linear(
                    self.head_hidden, bottleneck, bias=False,
                    use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
                )
                for _ in range(self.d_q - 1)
            ]
        )
        self.step_up = nn.ModuleList(
            [
                make_linear(
                    bottleneck, self.head_hidden, bias=False,
                    use_qat=use_qat, weight_bits=weight_bits, act_bits=act_bits,
                )
                for _ in range(self.d_q - 1)
            ]
        )
        self.digit_heads = nn.ModuleList(
            [
                nn.Linear(self.head_hidden, self.fsq_levels[j], bias=True)
                for j in range(self.d_q)
            ]
        )

    def _build_context(
        self, h: torch.Tensor, level_ids: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        proj = self.ctx_proj(h)
        if isinstance(level_ids, int):
            lid = torch.tensor(level_ids, device=h.device, dtype=torch.long)
            l_emb = self.level_emb(lid).view(1, 1, self.head_hidden)
        else:
            lid = level_ids.to(device=h.device, dtype=torch.long)
            l_emb = self.level_emb(lid)
            if l_emb.ndim == 2:
                l_emb = l_emb.unsqueeze(0)
        return F.silu(self.ctx_norm(proj + l_emb))

    def _advance_state(
        self, state: torch.Tensor, prev_digit: torch.Tensor, step_idx: int
    ) -> torch.Tensor:
        mixed = state + self.digit_embs[step_idx](prev_digit)
        return mixed + self.step_up[step_idx](
            F.silu(self.step_down[step_idx](self.step_norms[step_idx](mixed)))
        )

    def forward(
        self,
        h: torch.Tensor,
        level_ids: Union[int, torch.Tensor],
        target_digits: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Teacher-forced vectorized prediction of all `d_q` digit distributions.

        Args:
            h: [B, S, hidden_size]
            level_ids: scalar int or [S] / [B, S] int64 level indices
            target_digits: [B, S, d_q] int64 ground-truth FSQ digits
        Returns:
            list of d_q tensors, each [B, S, L_j]
        """
        state = self._build_context(h, level_ids)
        logits_list: List[torch.Tensor] = [self.digit_heads[0](state)]
        for j in range(1, self.d_q):
            state = self._advance_state(state, target_digits[..., j - 1], j - 1)
            logits_list.append(self.digit_heads[j](state))
        return logits_list

    def log_probs(
        self,
        h: torch.Tensor,
        level_ids: Union[int, torch.Tensor],
        target_digits: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return (digit_logits_list, joint_log_probs [B, S])."""
        digit_logits = self.forward(h, level_ids, target_digits)
        total_lp = h.new_zeros(h.shape[:2], dtype=torch.float32)
        for j, logits_j in enumerate(digit_logits):
            lp_j = F.log_softmax(logits_j.float(), dim=-1)
            gathered = lp_j.gather(-1, target_digits[..., j : j + 1]).squeeze(-1)
            total_lp = total_lp + gathered
        return digit_logits, total_lp

    @torch.no_grad()
    def sample(
        self,
        h: torch.Tensor,
        level_ids: Union[int, torch.Tensor],
        temperature: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressively sample the `d_q` FSQ digits in parallel across all S positions.

        Returns:
            digits: [B, S, d_q] int64
            joint_log_probs: [B, S] float32
            mean_confidence: [B] float32 (mean max-digit probability, used for early exit)
        """
        state = self._build_context(h, level_ids)
        bsz, seq_len, _ = h.shape
        sampled: List[torch.Tensor] = []
        total_lp = h.new_zeros((bsz, seq_len), dtype=torch.float32)
        conf_acc = h.new_zeros((bsz, seq_len), dtype=torch.float32)

        for j in range(self.d_q):
            if j > 0:
                state = self._advance_state(state, sampled[-1], j - 1)
            logits_j = self.digit_heads[j](state).float()
            probs_raw = F.softmax(logits_j, dim=-1)
            conf_acc = conf_acc + probs_raw.amax(dim=-1)

            if temperature <= 0.0:
                digit_j = logits_j.argmax(dim=-1)
                lp_j = F.log_softmax(logits_j, dim=-1)
            else:
                scaled = logits_j / float(temperature)
                probs = F.softmax(scaled, dim=-1)
                digit_j = torch.multinomial(
                    probs.reshape(-1, self.fsq_levels[j]), num_samples=1
                ).view(bsz, seq_len)
                lp_j = F.log_softmax(logits_j, dim=-1)

            total_lp = total_lp + lp_j.gather(-1, digit_j.unsqueeze(-1)).squeeze(-1)
            sampled.append(digit_j)

        digits = torch.stack(sampled, dim=-1)
        mean_conf = (conf_acc / float(self.d_q)).mean(dim=-1)  # [B]
        return digits, total_lp, mean_conf
